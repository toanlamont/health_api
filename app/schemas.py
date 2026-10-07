from datetime import datetime, date
from math import isfinite
from typing import Literal

from pydantic import BaseModel, EmailStr, model_validator

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserOut(BaseModel):
    id: str
    email: str
    full_name: str
    date_of_birth: date | None = None
    gender: str | None = None

class UserUpdate(BaseModel):
    date_of_birth: date | None = None
    gender: str | None = None

class VerifyEmail(BaseModel):
    email: EmailStr
    code: str

class ResendCode(BaseModel):
    email: EmailStr

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class HealthProfileUpdate(BaseModel):
    height_cm: float | None = None
    weight_kg: float | None = None
    blood_type: str | None = None

class HealthProfileOut(BaseModel):
    height_cm: float | None = None
    weight_kg: float | None = None
    bmi: float | None = None
    blood_type: str | None = None

class HeartRateCreate(BaseModel):
    bpm: int
    confidence: float = 0.0
    measurement_duration: int = 15
    context: str = "resting"

class HeartRateOut(BaseModel):
    id: str
    bpm: int
    confidence: float
    measurement_duration: int
    context: str
    measured_at: datetime

class SignalAnalyze(BaseModel):
    """Per-frame colour means captured on the device.

    `values` is flattened row-major: frame, then region, then R, G, B.
    """

    source: Literal["finger", "face"] = "finger"
    times: list[float]
    region_count: int = 1
    values: list[float]
    context: str = "resting"
    save: bool = True

    @model_validator(mode="after")
    def _check_shape(self):
        frames = len(self.times)
        if frames < 2:
            raise ValueError("need at least 2 frames")
        if frames > 5000:
            raise ValueError("too many frames")
        if not 1 <= self.region_count <= 64:
            raise ValueError("region_count must be between 1 and 64")
        expected = frames * self.region_count * 3
        if len(self.values) != expected:
            raise ValueError(f"expected {expected} values, got {len(self.values)}")
        if any(not isfinite(t) for t in self.times):
            raise ValueError("times must be finite")
        return self


class SignalAnalyzeOut(BaseModel):
    bpm: int | None = None
    confidence: float = 0.0
    snr_db: float | None = None
    amplitude: float | None = None
    method: str | None = None
    region: int | None = None
    fps: float | None = None
    # How many independent readings of the capture landed on this rate, out of
    # how many were taken. On captures checked against a watch, the ones that
    # matched scored every reading in agreement and the wrong ones scored a
    # third or less.
    agreement: int | None = None
    candidates: int | None = None
    # How alike the individual beats looked, on the fingertip path only.
    quality: float | None = None
    message: str | None = None


class WeightCreate(BaseModel):
    weight_kg: float

class WeightOut(BaseModel):
    id: str
    weight_kg: float
    bmi: float | None
    recorded_at: datetime

class ReminderCreate(BaseModel):
    type: str
    title: str
    schedule_cron: str

class ReminderOut(BaseModel):
    id: str
    type: str
    title: str
    schedule_cron: str
    is_enabled: bool

class GoalCreate(BaseModel):
    type: str
    target_value: float
    unit: str
    deadline: date | None = None

class GoalOut(BaseModel):
    id: str
    type: str
    target_value: float
    current_value: float
    unit: str
    deadline: date | None
    status: str

class GoalUpdate(BaseModel):
    current_value: float | None = None
    status: str | None = None

class DashboardOut(BaseModel):
    latest_bpm: int | None = None
    avg_bpm_7d: float | None = None
    current_bmi: float | None = None
    active_goals: int = 0
    measurement_count: int = 0

class MealCreate(BaseModel):
    name: str
    calories: int
    meal_type: Literal["breakfast", "lunch", "dinner", "snack"] = "snack"
    protein_g: float | None = None
    carbs_g: float | None = None
    fat_g: float | None = None
    eaten_at: datetime | None = None

class MealOut(BaseModel):
    id: str
    name: str
    calories: int
    meal_type: str
    protein_g: float | None
    carbs_g: float | None
    fat_g: float | None
    eaten_at: datetime

class NutritionDay(BaseModel):
    day: date
    calories: int
    protein_g: float
    carbs_g: float
    fat_g: float
    meals: int

class NutritionSummary(BaseModel):
    """Today against the target, plus enough history to draw a chart."""
    day: date
    calories: int
    protein_g: float
    carbs_g: float
    fat_g: float
    meals: int
    # None when the profile is missing height, weight or date of birth.
    target: dict | None = None
    history: list[NutritionDay] = []
