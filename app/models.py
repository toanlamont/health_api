import uuid
from datetime import datetime, date
from sqlalchemy import String, Integer, Float, Boolean, DateTime, Date, Text, Enum as SAEnum
from sqlalchemy.orm import Mapped, mapped_column
from app.database import Base

def gen_uuid() -> str:
    return str(uuid.uuid4())

class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(100))
    date_of_birth: Mapped[date | None] = mapped_column(Date, nullable=True)
    gender: Mapped[str | None] = mapped_column(String(10), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    # New accounts start unverified; init_db marks pre-existing ones verified.
    email_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

class HealthProfile(Base):
    __tablename__ = "health_profiles"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    user_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    height_cm: Mapped[float | None] = mapped_column(Float, nullable=True)
    weight_kg: Mapped[float | None] = mapped_column(Float, nullable=True)
    bmi: Mapped[float | None] = mapped_column(Float, nullable=True)
    blood_type: Mapped[str | None] = mapped_column(String(5), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, onupdate=datetime.now)

class HeartRateRecord(Base):
    __tablename__ = "heart_rate_records"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    bpm: Mapped[int] = mapped_column(Integer)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    measurement_duration: Mapped[int] = mapped_column(Integer, default=15)
    context: Mapped[str] = mapped_column(String(20), default="resting")
    measured_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

class WeightRecord(Base):
    __tablename__ = "weight_records"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    weight_kg: Mapped[float] = mapped_column(Float)
    bmi: Mapped[float | None] = mapped_column(Float, nullable=True)
    recorded_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

class Reminder(Base):
    __tablename__ = "reminders"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    type: Mapped[str] = mapped_column(String(20))
    title: Mapped[str] = mapped_column(String(200))
    schedule_cron: Mapped[str] = mapped_column(String(50))
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)

class HealthGoal(Base):
    __tablename__ = "health_goals"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    type: Mapped[str] = mapped_column(String(20))
    target_value: Mapped[float] = mapped_column(Float)
    current_value: Mapped[float] = mapped_column(Float, default=0.0)
    unit: Mapped[str] = mapped_column(String(20))
    deadline: Mapped[date | None] = mapped_column(Date, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="active")

class MealRecord(Base):
    __tablename__ = "meal_records"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    name: Mapped[str] = mapped_column(String(200))
    calories: Mapped[int] = mapped_column(Integer)
    # breakfast, lunch, dinner or snack
    meal_type: Mapped[str] = mapped_column(String(20), default="snack")
    protein_g: Mapped[float | None] = mapped_column(Float, nullable=True)
    carbs_g: Mapped[float | None] = mapped_column(Float, nullable=True)
    fat_g: Mapped[float | None] = mapped_column(Float, nullable=True)
    eaten_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)

class ChatConversation(Base):
    __tablename__ = "chat_conversations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    title: Mapped[str] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    # Set by hand on every new message, so the list can be sorted by activity.
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    conversation_id: Mapped[str] = mapped_column(String(36), index=True)
    # user or assistant. Tool calls are not stored: they are re-run on demand.
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)

class EmailVerification(Base):
    """The pending sign-up code for one user; only its hash is stored."""
    __tablename__ = "email_verifications"
    user_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    code_hash: Mapped[str] = mapped_column(String(64))
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    sent_at: Mapped[datetime] = mapped_column(DateTime)

class ChatUsage(Base):
    """One row per message sent to the model, for the per-user rate limit.

    Kept apart from chat_messages so deleting a conversation does not hand
    the quota back.
    """
    __tablename__ = "chat_usage"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, index=True)
