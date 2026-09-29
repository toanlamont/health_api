from datetime import date

import pytest

from app import nutrition


def test_age_counts_whole_years_only():
    born = date(2000, 6, 15)
    assert nutrition.age_from(born, date(2026, 6, 14)) == 25
    assert nutrition.age_from(born, date(2026, 6, 15)) == 26


def test_age_is_none_without_a_birthday():
    assert nutrition.age_from(None) is None


@pytest.mark.parametrize("born", [date(1800, 1, 1), date(2030, 1, 1)])
def test_implausible_birthdays_are_refused(born):
    # A typo in a date of birth would otherwise feed a nonsense age straight
    # into the equation and come back out as a confident calorie target.
    assert nutrition.age_from(born, date(2026, 1, 1)) is None


def test_matches_the_published_equation_for_a_man():
    # Mifflin-St Jeor: 10*kg + 6.25*cm - 5*age + 5
    # 10*80 + 6.25*180 - 5*30 + 5 = 800 + 1125 - 150 + 5
    assert nutrition.basal_rate(80, 180, 30, "male") == pytest.approx(1780)


def test_matches_the_published_equation_for_a_woman():
    # 10*60 + 6.25*165 - 5*30 - 161 = 600 + 1031.25 - 150 - 161
    assert nutrition.basal_rate(60, 165, 30, "female") == pytest.approx(1320.25)


def test_an_unrecorded_sex_lands_between_the_two():
    male = nutrition.basal_rate(70, 170, 30, "male")
    female = nutrition.basal_rate(70, 170, 30, "female")
    unknown = nutrition.basal_rate(70, 170, 30, None)
    assert female < unknown < male


def test_activity_raises_the_target_above_the_resting_rate():
    target = nutrition.daily_target(80, 180, date(1996, 1, 1), "male", "moderate")
    assert target is not None
    assert target["calories"] > target["basal_calories"]
    assert target["activity"] == "moderate"


def test_an_unknown_activity_level_falls_back_rather_than_failing():
    target = nutrition.daily_target(80, 180, date(1996, 1, 1), "male", "nonsense")
    assert target is not None
    assert target["calories"] > 0


@pytest.mark.parametrize(
    "weight,height,born",
    [(None, 180, date(1996, 1, 1)), (80, None, date(1996, 1, 1)), (80, 180, None)],
)
def test_returns_nothing_when_the_profile_is_incomplete(weight, height, born):
    # Better no target at all than one built on a guessed weight.
    assert nutrition.daily_target(weight, height, born, "male") is None


def test_macronutrients_add_back_up_to_the_calorie_target():
    target = nutrition.daily_target(80, 180, date(1996, 1, 1), "male")
    assert target is not None
    from_macros = (
        target["protein_g"] * 4 + target["carbs_g"] * 4 + target["fat_g"] * 9
    )
    assert from_macros == pytest.approx(target["calories"], rel=0.02)


def test_says_what_the_target_was_worked_out_from():
    # The number is a prediction with a couple of hundred kcal of slack, so the
    # client has to be able to show what went into it.
    target = nutrition.daily_target(72.5, 175, date(1990, 3, 2), "female")
    assert target is not None
    assert target["from_weight_kg"] == 72.5
    assert target["from_height_cm"] == 175
    assert target["from_gender"] == "female"
    assert target["from_age"] == nutrition.age_from(date(1990, 3, 2))
