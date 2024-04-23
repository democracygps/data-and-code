import pandas as pd

from collections import defaultdict
import os

from utils import get_fractions, calculate_margin_of_error
from constants import PROCESSED_DATA_DIR, SURVEY_CODED_DATA_PATH

FILE_DICT = {
    "baseline": "house_district_data_baseline_modelled_fast.csv",
    "right_only": "house_district_data_right_only_modelled_fast.csv",
    # "without_center": "house_district_data_without_center_modelled_fast.csv",
}

EXCEL_FILE_PATH = os.path.join(PROCESSED_DATA_DIR, "assault_weapon_summary.xlsx")

COLUMNS_IN_ORDER = [
    "state_name",
    "body",
    "district",
    "partisan_lean (%)",
    "gun_owner_fraction_right_only",
    "for_assault_weapon_ban_True_full_sample_mean",
    "for_assault_weapon_ban_True_full_sample_stderr",
    "for_large_magazine_ban_True_full_sample_mean",
    "for_large_magazine_ban_True_full_sample_stderr",
    "for_background_checks_True_full_sample_mean",
    "for_background_checks_True_full_sample_stderr",
    "for_safe_storage_laws_True_full_sample_mean",
    "for_safe_storage_laws_True_full_sample_stderr",
    "Republican/lean Rep.",
    "No lean",
    "Democrat/lean Dem.",
    "lean sample size",
    "firearm ownership",
    "firearm ownership margin of error",
    "has_gun_in_house_True",
    "has_gun_in_house_False",
    "is_gun_owner_True",
    "is_gun_owner_False",
    "political_lean_center",
    "political_lean_right",
    "political_lean_left",
    "lean_sum",
    "right_only_flag",
    "model",
    "errors_short",
]

RENAMED_COLUMNS = {
    "state_name": "State",
    "body": "Body",
    "district": "District",
    "partisan_lean (%)": "Partisan Lean (%)",
    "gun_owner_fraction_right_only": "Estimated Gun Owner Fraction",
    "for_assault_weapon_ban_True_full_sample_mean": "For Assault Weapon Ban",
    "for_assault_weapon_ban_True_full_sample_stderr": "For Assault Weapon Ban Error",
    "for_large_magazine_ban_True_full_sample_mean": "For Large Magazine Ban",
    "for_large_magazine_ban_True_full_sample_stderr": "For Large Magazine Ban Error",
    "for_background_checks_True_full_sample_mean": "For Background Checks",
    "for_background_checks_True_full_sample_stderr": "For Background Checks Error",
    "for_safe_storage_laws_True_full_sample_mean": "For Safe Storage Laws",
    "for_safe_storage_laws_True_full_sample_stderr": "For Safe Storage Laws Error",
    "Republican/lean Rep.": "State Republican/lean Rep.",
    "No lean": "State No lean",
    "Democrat/lean Dem.": "State Democrat/lean Dem.",
    "lean sample size": "State Lean Sample Size",
    "firearm ownership": "State Firearm Ownership",
    "firearm ownership margin of error": "State Firearm Ownership Margin of Error",
}


def make_pivot_table(df_poll):
    outcomes = [
        "for_assault_weapon_ban",
        "for_large_magazine_ban",
        "for_background_checks",
        "for_safe_storage_laws",
    ]
    outcome_lookup = {
        "for_assault_weapon_ban": "For Assault Weapon Ban",
        "for_large_magazine_ban": "For Large Magazine Ban",
        "for_background_checks": "For Background Checks",
        "for_safe_storage_laws": "For Safe Storage Laws",
    }
    samples = [
        {
            "predictors": {"political_lean": "right", "is_gun_owner": True},
            "label": ("Republican or conservative", "gun owner"),
        },
        {
            "predictors": {"political_lean": "right", "is_gun_owner": False},
            "label": ("Republican or conservative", "non-owner"),
        },
        {
            "predictors": {"political_lean": "left", "is_gun_owner": True},
            "label": ("Democrat or liberal", "gun owner"),
        },
        {
            "predictors": {"political_lean": "left", "is_gun_owner": False},
            "label": ("Democrat or liberal", "non-owner"),
        },
    ]
    results = defaultdict(dict)
    for sample in samples:
        df_sample = df_poll.copy()
        for key, value in sample["predictors"].items():
            df_sample = df_sample[df_sample[key] == value]
        N = df_sample.shape[0]
        results["Number of responses"][sample["label"]] = N
        for outcome in outcomes:
            result = get_fractions(df_sample, [outcome], as_dict=True)
            error = calculate_margin_of_error(result[True], N)
            result_true = result[True]
            result_string = f"{round(result_true * 100)} ± {round(error * 100)}%"
            results[outcome_lookup[outcome]][sample["label"]] = result_string
    result_df = pd.DataFrame(results)
    result_df = result_df.T
    return result_df


def sort_columns(df):
    df = df[COLUMNS_IN_ORDER]
    return df


def rename_columns(df):
    df = df.rename(columns=RENAMED_COLUMNS)
    return df


if __name__ == "__main__":
    df_poll = pd.read_csv(SURVEY_CODED_DATA_PATH)
    df_pivot_table = make_pivot_table(df_poll)
    print(df_pivot_table)
    df_list = [df_pivot_table]
    for key, value in FILE_DICT.items():
        filepath = os.path.join(PROCESSED_DATA_DIR, value)
        df = pd.read_csv(filepath)
        df["partisan_lean (%)"] = df["partisan_lean (%)"].apply(lambda x: x / 100)
        df = sort_columns(df)
        df = rename_columns(df)
        df_list.append(df)
    sheet_names = ["Summary", "All Voters", "Right-Leaning Voters"]
    with pd.ExcelWriter(EXCEL_FILE_PATH, engine="xlsxwriter") as writer:
        for df, sheet_name in zip(df_list, sheet_names):
            df.to_excel(writer, sheet_name=sheet_name)
    print(f"Dataframes written successfully to {EXCEL_FILE_PATH}")
