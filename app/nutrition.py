"""Daily energy needs from the figures already in a user's profile.

Mifflin-St Jeor (1990) rather than Harris-Benedict: it was fitted on a more
recent population and predicts resting energy expenditure a few percent closer
on average, and it is what most clinical calculators use now.

    Mifflin MD, St Jeor ST, Hill LA, Scott BJ, Daugherty SA, Koh YO (1990).
    A new predictive equation for resting energy expenditure in healthy
    individuals. Am J Clin Nutr 51(2):241-7.

Nothing here is a medical recommendation. A predicted target is a starting
point that can be out by a couple of hundred kcal for any given person, which
is why the API returns it alongside the inputs it came from.
"""

from __future__ import annotations

from datetime import date

# Multipliers on the resting rate. The usual published ladder; "light" is the
# default because an app someone opens on a phone cannot tell how much they
# actually move.
ACTIVITY_FACTORS = {
    "sedentary": 1.2,
    "light": 1.375,
    "moderate": 1.55,
    "active": 1.725,
    "very_active": 1.9,
}
DEFAULT_ACTIVITY = "light"

# Rough split used to turn a calorie target into gram targets: 20% of energy
# from protein, 50% carbohydrate, 30% fat, at 4, 4 and 9 kcal per gram.
MACRO_SPLIT = {"protein": (0.20, 4.0), "carbs": (0.50, 4.0), "fat": (0.30, 9.0)}


def age_from(born: date | None, today: date | None = None) -> int | None:
    if born is None:
        return None
    today = today or date.today()
    years = today.year - born.year
    if (today.month, today.day) < (born.month, born.day):
        years -= 1
    return years if 0 < years < 130 else None


def basal_rate(weight_kg: float, height_cm: float, age: int, gender: str | None) -> float:
    """Resting energy expenditure in kcal per day."""
    base = 10.0 * weight_kg + 6.25 * height_cm - 5.0 * age
    # The equation has separate constants per sex. With nothing recorded the
    # midpoint is used, so the answer is wrong by about 80 kcal either way
    # rather than confidently wrong in one direction.
    if gender and gender.lower().startswith("m"):
        return base + 5.0
    if gender and gender.lower().startswith("f"):
        return base - 161.0
    return base - 78.0


def daily_target(
    weight_kg: float | None,
    height_cm: float | None,
    born: date | None,
    gender: str | None,
    activity: str = DEFAULT_ACTIVITY,
) -> dict | None:
    """Calories and macronutrients for a day, or None if the profile is short
    of what the equation needs."""
    age = age_from(born)
    if not weight_kg or not height_cm or age is None:
        return None

    resting = basal_rate(weight_kg, height_cm, age, gender)
    factor = ACTIVITY_FACTORS.get(activity, ACTIVITY_FACTORS[DEFAULT_ACTIVITY])
    calories = resting * factor

    return {
        "calories": int(round(calories)),
        "basal_calories": int(round(resting)),
        "activity": activity,
        "protein_g": int(round(calories * MACRO_SPLIT["protein"][0] / MACRO_SPLIT["protein"][1])),
        "carbs_g": int(round(calories * MACRO_SPLIT["carbs"][0] / MACRO_SPLIT["carbs"][1])),
        "fat_g": int(round(calories * MACRO_SPLIT["fat"][0] / MACRO_SPLIT["fat"][1])),
        # Returned so the client can say what the number was worked out from
        # instead of presenting it as fact.
        "from_weight_kg": weight_kg,
        "from_height_cm": height_cm,
        "from_age": age,
        "from_gender": gender,
    }
