"""
preprocess_data

process democracyGPS survey monkey data on assault weapons

adds columns for:
    is_gun_owner
    has_gun_in_house
    political_lean
    leans_left
    leans_right
    leans_center
    for_assault_weapon_ban
    for_large_magazine_ban
    for_background_checks
    for_safe_storage_laws

creates metadata json file with:
    list of outcomes
    list of predictors
    crosstabs of predictors and outcomes

{'all_outcomes': ['for_assault_weapon_ban',
                  'for_large_magazine_ban',
                  'for_background_checks',
                  'for_safe_storage_laws'],
 'all_predictors': ['is_gun_owner',
                    'has_gun_in_house',
                    'leans_left',
                    'leans_right',
                    'leans_center'],
 'crosstabs': [{'axis_names': {'columns': 'is_gun_owner',
                               'rows': 'political_lean'},
                'data': {False: {'center': 94, 'left': 159, 'right': 80},
                         True: {'center': 54, 'left': 85, 'right': 88}}},
               {'axis_names': {'columns': 'has_gun_in_house',
                               'rows': 'political_lean'},
                'data': {False: {'center': 77, 'left': 135, 'right': 64},
                         True: {'center': 71, 'left': 109, 'right': 104}}}]}

"""

import pandas as pd

import json
import pprint

# filtering data by COLLECTOR_ID removes test data
from constants import (
    SURVEY_INPUT_DATA_PATH,
    SURVEY_CODED_DATA_PATH,
    SURVEY_METADATA_PATH,
    COLLECTOR_ID,
)

# maps from survey answers to boolean outcomes
outcome_maps = [
    {
        "label": "for_assault_weapon_ban",
        "question": "On Assault Weapons: Should owning guns that fire quickly be:",
        "answers": [
            "Limited to military?",
            "Limited to military and police?",
        ],
    },
    {
        "label": "for_large_magazine_ban",
        "question": "On Magazines Holding Over Ten Bullets: Who should have them?",
        "answers": [
            "Just military?",
            "Just military and police?",
        ],
    },
    {
        "label": "for_background_checks",
        "question": (
            "On Background Checks for Assault Weapons: Should there be "
            "checks to ensure owners are allowed to have them?"
        ),
        "answers": [
            "Yes, for all owners",
        ],
    },
    {
        "label": "for_safe_storage_laws",
        "question": (
            "On Keeping Assault Weapons Away from Children: "
            "Should it be illegal to leave these guns where kids can find them?"
        ),
        "answers": [
            "Yes",
        ],
    },
]


def add_boolean_columns(df):
    """
    takes a dataframe and adds columns for boolean outcomes.
    also returns a list of all predictors and outcomes

    returns: dict
    {"df": df, "all_predictors": list, "all_outcomes": list}
    """
    all_predictors = []
    all_outcomes = []

    # define predictor columns
    def map_function(row):
        return (
            row["Do You Own a Gun? (Excluding paintball, BB, or pellet guns)"] == "Yes"
        )

    new_column_name = "is_gun_owner"
    df[new_column_name] = df.apply(map_function, axis=1)
    all_predictors.append(new_column_name)

    def map_function(row):
        return (
            row["Do You Own a Gun? (Excluding paintball, BB, or pellet guns)"] == "Yes"
            or row[
                "Does Someone in Your Home Own a Gun? (Excluding paintball, BB, or pellet guns)"
            ]
            == "Yes"
        )

    new_column_name = "has_gun_in_house"
    df[new_column_name] = df.apply(map_function, axis=1)
    all_predictors.append(new_column_name)

    def map_function(row):
        if row["Do you think of yourself as"] == "A Republican?":
            return "right"
        if row["Do you think of yourself as"] == "A Democrat?":
            return "left"
        if (
            row["Do you see yourself as"]
            == "More open to new ideas and changes (liberal)?"
        ):
            return "left"
        if (
            row["Do you see yourself as"]
            == "Preferring traditional ways and less change (conservative)?"
        ):
            return "right"
        return "center"

    new_column_name = "political_lean"  # left, right, center
    df[new_column_name] = df.apply(map_function, axis=1)

    def map_function(row):
        return row["political_lean"] == "left"

    new_column_name = "leans_left"  # left, right, center
    df[new_column_name] = df.apply(map_function, axis=1)
    all_predictors.append(new_column_name)

    def map_function(row):
        return row["political_lean"] == "right"

    new_column_name = "leans_right"
    df[new_column_name] = df.apply(map_function, axis=1)
    all_predictors.append(new_column_name)

    def map_function(row):
        return row["political_lean"] == "center"

    new_column_name = "leans_center"
    df[new_column_name] = df.apply(map_function, axis=1)
    all_predictors.append(new_column_name)

    # define outcomes

    for outcome_map in outcome_maps:
        label = outcome_map["label"]
        column = outcome_map["question"]
        answers = outcome_map["answers"]
        all_outcomes.append(label)

        def map_function(row):
            return row[column] in answers

        df[label] = df.apply(map_function, axis=1)
    return {"df": df, "all_predictors": all_predictors, "all_outcomes": all_outcomes}


def make_crosstab(crosstab_column_one, crosstab_column_two):
    crosstab_df = pd.crosstab(df[crosstab_column_one], df[crosstab_column_two])
    return {
        "data": crosstab_df.to_dict(),
        "axis_names": {"rows": crosstab_column_one, "columns": crosstab_column_two},
    }


if __name__ == "__main__":
    # read data
    df = pd.read_csv(SURVEY_INPUT_DATA_PATH)
    df = df[df["Collector ID"] == COLLECTOR_ID]

    # add boolean columns
    result = add_boolean_columns(df)
    df = result["df"]
    all_predictors = result["all_predictors"]
    all_outcomes = result["all_outcomes"]

    # write data
    df.to_csv(SURVEY_CODED_DATA_PATH, index=False)
    print(f"data written to: {SURVEY_CODED_DATA_PATH}")

    # construct and write metadata
    metadata = {
        "all_predictors": all_predictors,
        "all_outcomes": all_outcomes,
        "crosstabs": [
            make_crosstab("political_lean", "is_gun_owner"),
            make_crosstab("political_lean", "has_gun_in_house"),
        ],
    }

    with open(SURVEY_METADATA_PATH, "w") as json_file:
        json.dump(metadata, json_file)
    print(f"metadata written to: {SURVEY_METADATA_PATH}")
    pprint.pprint(metadata)
