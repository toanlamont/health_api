"""Read-only tools the chat model can call to look at the user's own data.

Every query is scoped to the user making the request; the model only chooses
which tool and how far back to look, never whose data.
"""

from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import nutrition
from app.models import (
    HealthGoal, HealthProfile, HeartRateRecord, MealRecord, Reminder, User,
    WeightRecord,
)


def _days_param(default: int, maximum: int) -> dict:
    return {
        "type": "integer",
        "description": f"Số ngày gần nhất cần xem (mặc định {default}, tối đa {maximum}).",
        "minimum": 1,
        "maximum": maximum,
    }


def _tool(name: str, description: str, properties: dict | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties or {}},
        },
    }


TOOLS = [
    _tool(
        "get_profile",
        "Hồ sơ người dùng: tên, tuổi, giới tính, chiều cao, cân nặng, BMI, nhóm máu, "
        "và mức calo khuyến nghị mỗi ngày.",
    ),
    _tool(
        "get_heart_rate",
        "Các lần đo nhịp tim gần đây (bpm, hoàn cảnh đo, thời điểm) kèm trung bình/thấp nhất/cao nhất.",
        {"days": _days_param(30, 365)},
    ),
    _tool(
        "get_weight",
        "Lịch sử cân nặng và BMI, kèm mức thay đổi trong khoảng thời gian.",
        {"days": _days_param(90, 365)},
    ),
    _tool(
        "get_meals",
        "Các bữa ăn đã ghi (tên, calo, macro) và tổng calo theo từng ngày, so với mục tiêu calo.",
        {"days": _days_param(7, 30)},
    ),
    _tool("get_goals", "Các mục tiêu sức khoẻ người dùng đặt ra và tiến độ."),
    _tool("get_reminders", "Các lời nhắc người dùng đã đặt (uống nước, đo nhịp tim...)."),
]


def _since(args: dict, default: int, maximum: int) -> datetime:
    try:
        days = int(args.get("days") or default)
    except (TypeError, ValueError):
        days = default
    days = max(1, min(days, maximum))
    return datetime.now() - timedelta(days=days)


def _ts(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M")


async def _profile(db: AsyncSession, user: User) -> HealthProfile | None:
    result = await db.execute(
        select(HealthProfile)
        .where(HealthProfile.user_id == user.id)
        .order_by(HealthProfile.updated_at.desc())
        .limit(1)
    )
    return result.scalars().first()


async def get_profile(db: AsyncSession, user: User, args: dict) -> dict:
    p = await _profile(db, user)
    return {
        "full_name": user.full_name,
        "age": nutrition.age_from(user.date_of_birth),
        "gender": user.gender,
        "height_cm": p.height_cm if p else None,
        "weight_kg": p.weight_kg if p else None,
        "bmi": p.bmi if p else None,
        "blood_type": p.blood_type if p else None,
        "daily_calorie_target": nutrition.daily_target(
            weight_kg=p.weight_kg if p else None,
            height_cm=p.height_cm if p else None,
            born=user.date_of_birth,
            gender=user.gender,
        ),
    }


async def get_heart_rate(db: AsyncSession, user: User, args: dict) -> dict:
    rows = (await db.execute(
        select(HeartRateRecord)
        .where(HeartRateRecord.user_id == user.id,
               HeartRateRecord.measured_at >= _since(args, 30, 365))
        .order_by(HeartRateRecord.measured_at.desc())
    )).scalars().all()
    if not rows:
        return {"count": 0, "records": []}
    bpms = [r.bpm for r in rows]
    return {
        "count": len(rows),
        "avg_bpm": round(sum(bpms) / len(bpms), 1),
        "min_bpm": min(bpms),
        "max_bpm": max(bpms),
        # Newest first; capped so a heavy user doesn't blow up the prompt.
        "records": [
            {"bpm": r.bpm, "context": r.context, "measured_at": _ts(r.measured_at)}
            for r in rows[:30]
        ],
    }


async def get_weight(db: AsyncSession, user: User, args: dict) -> dict:
    rows = (await db.execute(
        select(WeightRecord)
        .where(WeightRecord.user_id == user.id,
               WeightRecord.recorded_at >= _since(args, 90, 365))
        .order_by(WeightRecord.recorded_at.desc())
    )).scalars().all()
    if not rows:
        return {"count": 0, "records": []}
    return {
        "count": len(rows),
        "latest_kg": rows[0].weight_kg,
        "change_kg": round(rows[0].weight_kg - rows[-1].weight_kg, 1),
        "records": [
            {"weight_kg": r.weight_kg, "bmi": r.bmi, "recorded_at": _ts(r.recorded_at)}
            for r in rows[:30]
        ],
    }


async def get_meals(db: AsyncSession, user: User, args: dict) -> dict:
    rows = (await db.execute(
        select(MealRecord)
        .where(MealRecord.user_id == user.id,
               MealRecord.eaten_at >= _since(args, 7, 30))
        .order_by(MealRecord.eaten_at.desc())
    )).scalars().all()
    per_day: dict[str, int] = {}
    for m in rows:
        day = m.eaten_at.date().isoformat()
        per_day[day] = per_day.get(day, 0) + m.calories
    p = await _profile(db, user)
    target = nutrition.daily_target(
        weight_kg=p.weight_kg if p else None,
        height_cm=p.height_cm if p else None,
        born=user.date_of_birth,
        gender=user.gender,
    )
    return {
        "daily_target": target,
        "calories_per_day": per_day,
        "meals": [
            {
                "name": m.name, "meal_type": m.meal_type, "calories": m.calories,
                "protein_g": m.protein_g, "carbs_g": m.carbs_g, "fat_g": m.fat_g,
                "eaten_at": _ts(m.eaten_at),
            }
            for m in rows[:50]
        ],
    }


async def get_goals(db: AsyncSession, user: User, args: dict) -> dict:
    rows = (await db.execute(
        select(HealthGoal).where(HealthGoal.user_id == user.id)
    )).scalars().all()
    return {"goals": [
        {
            "type": g.type, "target_value": g.target_value, "current_value": g.current_value,
            "unit": g.unit, "deadline": g.deadline.isoformat() if g.deadline else None,
            "status": g.status,
        }
        for g in rows
    ]}


async def get_reminders(db: AsyncSession, user: User, args: dict) -> dict:
    rows = (await db.execute(
        select(Reminder).where(Reminder.user_id == user.id)
    )).scalars().all()
    return {"reminders": [
        {"type": r.type, "title": r.title, "schedule_cron": r.schedule_cron, "enabled": r.is_enabled}
        for r in rows
    ]}


_HANDLERS = {
    "get_profile": get_profile,
    "get_heart_rate": get_heart_rate,
    "get_weight": get_weight,
    "get_meals": get_meals,
    "get_goals": get_goals,
    "get_reminders": get_reminders,
}


async def run_tool(name: str, args: dict, db: AsyncSession, user: User) -> dict:
    handler = _HANDLERS.get(name)
    if handler is None:
        return {"error": f"unknown tool {name}"}
    return await handler(db, user, args if isinstance(args, dict) else {})
