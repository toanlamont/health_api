from fastapi import APIRouter, Depends, HTTPException
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import select, func
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta, date as date_type
import logging
import numpy as np
from app import captures, nutrition, ppg
from app.database import get_db
from app.models import (
    User, HealthProfile, HeartRateRecord, WeightRecord, Reminder, HealthGoal,
    MealRecord,
)
from app.schemas import (
    HealthProfileUpdate, HealthProfileOut,
    HeartRateCreate, HeartRateOut,
    WeightCreate, WeightOut,
    ReminderCreate, ReminderOut,
    GoalCreate, GoalOut, GoalUpdate,
    DashboardOut,
    SignalAnalyze, SignalAnalyzeOut,
    MealCreate, MealOut, NutritionDay, NutritionSummary,
)
from app.auth import get_current_user

router = APIRouter(prefix="/api/v1", tags=["health"])
log = logging.getLogger("heartbeat.ppg")

# --- Health Profile ---
async def _find_profile(db: AsyncSession, user_id: str) -> HealthProfile | None:
    # user_id was not unique, so concurrent first requests could each insert a
    # row. Take the most recently updated one instead of failing on duplicates.
    result = await db.execute(
        select(HealthProfile)
        .where(HealthProfile.user_id == user_id)
        .order_by(HealthProfile.updated_at.desc())
        .limit(1)
    )
    return result.scalars().first()

async def _get_or_create_profile(db: AsyncSession, user_id: str) -> HealthProfile:
    profile = await _find_profile(db, user_id)
    if profile:
        return profile
    profile = HealthProfile(user_id=user_id)
    db.add(profile)
    try:
        await db.flush()
    except IntegrityError:
        # Lost a race with a concurrent request that created it first.
        await db.rollback()
        return await _find_profile(db, user_id)
    return profile

@router.get("/health-profile", response_model=HealthProfileOut)
async def get_profile(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    profile = await _get_or_create_profile(db, user.id)
    await db.commit()
    await db.refresh(profile)
    return profile

@router.put("/health-profile", response_model=HealthProfileOut)
async def update_profile(data: HealthProfileUpdate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    profile = await _get_or_create_profile(db, user.id)
    if data.height_cm is not None:
        profile.height_cm = data.height_cm
    if data.weight_kg is not None:
        profile.weight_kg = data.weight_kg
    if data.blood_type is not None:
        profile.blood_type = data.blood_type
    if profile.height_cm and profile.weight_kg:
        profile.bmi = round(profile.weight_kg / (profile.height_cm / 100) ** 2, 1)
    await db.commit()
    await db.refresh(profile)
    return profile

# --- Heart Rate ---
@router.post("/heart-rate", response_model=HeartRateOut, status_code=201)
async def create_heart_rate(data: HeartRateCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    record = HeartRateRecord(user_id=user.id, bpm=data.bpm, confidence=data.confidence, measurement_duration=data.measurement_duration, context=data.context)
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record

@router.post("/heart-rate/analyze", response_model=SignalAnalyzeOut)
async def analyze_signal(data: SignalAnalyze, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    times = np.asarray(data.times, dtype=float)
    regions = np.asarray(data.values, dtype=float).reshape(len(data.times), data.region_count, 3)

    # Filtering and spectral analysis are CPU-bound; keep them off the loop.
    result = await run_in_threadpool(ppg.estimate, times, regions, data.source)

    duration = int(round(float(times[-1] - times[0])))

    # Only the final analysis is archived. While a capture is running the app
    # polls this endpoint every couple of seconds for a running estimate, and
    # archiving those would bury the finished capture.
    if data.save:
        saved = captures.save(
            data.source, data.times, data.region_count, data.values,
            {} if result is None else {
                "bpm": result.bpm, "snr_db": result.snr_db, "method": result.method,
                "region": result.region, "fps": result.fps, "quality": result.quality,
            "agreement": result.agreement, "candidates": result.candidates,
            },
        )
        if saved is not None:
            log.info("capture saved to %s", saved.name)

    log.info(
        "%s source=%s frames=%d regions=%d span=%ds -> %s",
        "FINAL " if data.save else "live  ",
        data.source, len(data.times), data.region_count, duration,
        "unusable capture" if result is None else
        f"{result.bpm:.1f}bpm snr={result.snr_db:.2f}dB amp={result.amplitude:.4f} "
        f"agree={result.agreement}/{result.candidates} "
        f"sqi={'-' if result.quality is None else f'{result.quality:.2f}'} "
        f"via {result.method} region={result.region} fps={result.fps:.1f}",
    )

    if result is None:
        return SignalAnalyzeOut(message="Capture too short or unusable")

    # The rate always comes back, however weak the signal. Judging it is what
    # snr_db is for, and that judgement needs a reference heart rate to be
    # worth anything -- see the note at the top of app/ppg.py.
    bpm = int(round(result.bpm))

    if data.save:
        db.add(HeartRateRecord(
            user_id=user.id,
            bpm=bpm,
            confidence=result.confidence,
            measurement_duration=duration,
            context=data.context,
        ))
        await db.commit()

    return SignalAnalyzeOut(
        bpm=bpm,
        confidence=round(result.confidence, 3),
        snr_db=round(result.snr_db, 2),
        amplitude=round(result.amplitude, 4),
        method=result.method,
        region=result.region,
        fps=round(result.fps, 2),
        agreement=result.agreement,
        candidates=result.candidates,
        quality=None if result.quality is None else round(result.quality, 3),
    )

@router.get("/heart-rate", response_model=list[HeartRateOut])
async def list_heart_rate(limit: int = 20, offset: int = 0, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(HeartRateRecord).where(HeartRateRecord.user_id == user.id)
        .order_by(HeartRateRecord.measured_at.desc()).limit(limit).offset(offset)
    )
    return result.scalars().all()

@router.get("/heart-rate/latest", response_model=HeartRateOut | None)
async def latest_heart_rate(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(HeartRateRecord).where(HeartRateRecord.user_id == user.id)
        .order_by(HeartRateRecord.measured_at.desc()).limit(1)
    )
    return result.scalar_one_or_none()


# --- Meals and calories ---
@router.post("/meals", response_model=MealOut, status_code=201)
async def create_meal(data: MealCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    record = MealRecord(
        user_id=user.id, name=data.name, calories=data.calories,
        meal_type=data.meal_type, protein_g=data.protein_g,
        carbs_g=data.carbs_g, fat_g=data.fat_g,
        eaten_at=data.eaten_at or datetime.now(),
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record

@router.get("/meals", response_model=list[MealOut])
async def list_meals(day: date_type | None = None, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Meals for one day, newest first. Defaults to today."""
    day = day or datetime.now().date()
    start = datetime.combine(day, datetime.min.time())
    result = await db.execute(
        select(MealRecord).where(
            MealRecord.user_id == user.id,
            MealRecord.eaten_at >= start,
            MealRecord.eaten_at < start + timedelta(days=1),
        ).order_by(MealRecord.eaten_at.desc())
    )
    return result.scalars().all()

@router.delete("/meals/{meal_id}", status_code=204)
async def delete_meal(meal_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(MealRecord).where(MealRecord.id == meal_id, MealRecord.user_id == user.id))
    meal = result.scalar_one_or_none()
    if not meal:
        raise HTTPException(status_code=404, detail="Meal not found")
    await db.delete(meal)
    await db.commit()

@router.get("/nutrition/summary", response_model=NutritionSummary)
async def nutrition_summary(days: int = 14, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    """Today's intake against the target, with the preceding days for a chart."""
    days = max(1, min(days, 90))
    today = datetime.now().date()
    since = datetime.combine(today - timedelta(days=days - 1), datetime.min.time())

    result = await db.execute(
        select(MealRecord).where(MealRecord.user_id == user.id, MealRecord.eaten_at >= since)
    )
    meals = result.scalars().all()

    by_day: dict[date_type, dict] = {}
    for meal in meals:
        bucket = by_day.setdefault(
            meal.eaten_at.date(),
            {"calories": 0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "meals": 0},
        )
        bucket["calories"] += meal.calories
        bucket["protein_g"] += meal.protein_g or 0.0
        bucket["carbs_g"] += meal.carbs_g or 0.0
        bucket["fat_g"] += meal.fat_g or 0.0
        bucket["meals"] += 1

    # Every day in the window, including the empty ones: a chart with gaps
    # silently closed over reads as if nothing was missed.
    history = []
    for offset in range(days - 1, -1, -1):
        current = today - timedelta(days=offset)
        bucket = by_day.get(current, {"calories": 0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "meals": 0})
        history.append(NutritionDay(day=current, **bucket))

    profile = await _find_profile(db, user.id)
    target = nutrition.daily_target(
        weight_kg=profile.weight_kg if profile else None,
        height_cm=profile.height_cm if profile else None,
        born=user.date_of_birth,
        gender=user.gender,
    )

    todays = by_day.get(today, {"calories": 0, "protein_g": 0.0, "carbs_g": 0.0, "fat_g": 0.0, "meals": 0})
    return NutritionSummary(day=today, target=target, history=history, **todays)

# --- Weight ---
@router.post("/weight", response_model=WeightOut, status_code=201)
async def create_weight(data: WeightCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    profile = await _find_profile(db, user.id)
    bmi = None
    if profile and profile.height_cm:
        bmi = round(data.weight_kg / (profile.height_cm / 100) ** 2, 1)
        profile.weight_kg = data.weight_kg
        profile.bmi = bmi
    record = WeightRecord(user_id=user.id, weight_kg=data.weight_kg, bmi=bmi)
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record

@router.get("/weight", response_model=list[WeightOut])
async def list_weight(limit: int = 20, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(WeightRecord).where(WeightRecord.user_id == user.id)
        .order_by(WeightRecord.recorded_at.desc()).limit(limit)
    )
    return result.scalars().all()

# --- Reminders ---
@router.post("/reminders", response_model=ReminderOut, status_code=201)
async def create_reminder(data: ReminderCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    reminder = Reminder(user_id=user.id, type=data.type, title=data.title, schedule_cron=data.schedule_cron)
    db.add(reminder)
    await db.commit()
    await db.refresh(reminder)
    return reminder

@router.get("/reminders", response_model=list[ReminderOut])
async def list_reminders(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Reminder).where(Reminder.user_id == user.id))
    return result.scalars().all()

@router.delete("/reminders/{reminder_id}", status_code=204)
async def delete_reminder(reminder_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Reminder).where(Reminder.id == reminder_id, Reminder.user_id == user.id))
    reminder = result.scalar_one_or_none()
    if not reminder:
        raise HTTPException(status_code=404, detail="Reminder not found")
    await db.delete(reminder)
    await db.commit()

# --- Goals ---
@router.post("/goals", response_model=GoalOut, status_code=201)
async def create_goal(data: GoalCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    goal = HealthGoal(user_id=user.id, type=data.type, target_value=data.target_value, unit=data.unit, deadline=data.deadline)
    db.add(goal)
    await db.commit()
    await db.refresh(goal)
    return goal

@router.get("/goals", response_model=list[GoalOut])
async def list_goals(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(HealthGoal).where(HealthGoal.user_id == user.id))
    return result.scalars().all()

@router.put("/goals/{goal_id}", response_model=GoalOut)
async def update_goal(goal_id: str, data: GoalUpdate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(HealthGoal).where(HealthGoal.id == goal_id, HealthGoal.user_id == user.id))
    goal = result.scalar_one_or_none()
    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")
    if data.current_value is not None:
        goal.current_value = data.current_value
    if data.status is not None:
        goal.status = data.status
    await db.commit()
    await db.refresh(goal)
    return goal

# --- Dashboard ---
@router.get("/dashboard", response_model=DashboardOut)
async def dashboard(user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)):
    latest = await db.execute(
        select(HeartRateRecord).where(HeartRateRecord.user_id == user.id)
        .order_by(HeartRateRecord.measured_at.desc()).limit(1)
    )
    latest_hr = latest.scalar_one_or_none()

    week_ago = datetime.now() - timedelta(days=7)
    avg_result = await db.execute(
        select(func.avg(HeartRateRecord.bpm)).where(
            HeartRateRecord.user_id == user.id, HeartRateRecord.measured_at >= week_ago
        )
    )
    avg_bpm = avg_result.scalar()

    count_result = await db.execute(
        select(func.count()).select_from(HeartRateRecord).where(HeartRateRecord.user_id == user.id)
    )
    count = count_result.scalar()

    profile = await _find_profile(db, user.id)

    goals_result = await db.execute(
        select(func.count()).select_from(HealthGoal).where(HealthGoal.user_id == user.id, HealthGoal.status == "active")
    )
    active_goals = goals_result.scalar()

    return DashboardOut(
        latest_bpm=latest_hr.bpm if latest_hr else None,
        avg_bpm_7d=round(avg_bpm, 1) if avg_bpm else None,
        current_bmi=profile.bmi if profile else None,
        active_goals=active_goals or 0,
        measurement_count=count or 0,
    )
