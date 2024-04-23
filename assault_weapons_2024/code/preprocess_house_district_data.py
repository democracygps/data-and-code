"""
preprocess_data

combines data from multiple sources to create datasets for house districts

three models are supported:
    baseline:
        uses Pew and fivethirtyeight data to estimate left, center, and right
        populations
    without_center:
        uses only fifethirtyeight data to estimate left and right populations
    right_only:
        assumes 100% of population is Republican/lean Rep.
        only uses Rand data for firearm ownership

columns include:
    district
        district partisan_lean (%)

    state_name and state lean
        Republican/lean Rep.
        No lean
        Democrat/lean Dem.
        lean sample size

    state firearm ownership
        firearm ownership margin of error

    boolean variables
        has_gun_in_house_True
        has_gun_in_house_False
        is_gun_owner_True
        is_gun_owner_False
        political_lean_center
        political_lean_right
        political_lean_left

    data sources:
        https://www.rand.org/content/dam/rand/pubs/tools/TL300/TL354/RAND_TL354.database.zip
        https://www.pewresearch.org/religion/religious-landscape-study/compare/party-affiliation/by/state#party-affiliation
        https://raw.githubusercontent.com/fivethirtyeight/data/master/partisan-lean/fivethirtyeight_partisan_lean_DISTRICTS.csv
        https://raw.githubusercontent.com/fivethirtyeight/data/master/gop-delegate-benchmarks-2024/delegate_targets.csv

"""

import pandas as pd
from icecream import ic

import os

from constants import INPUT_DATA_DIR, PROCESSED_DATA_DIR

# The right only flag is used during population building
# so that gun ownership is not underestimated.
# (A population first collected with state data on lean and gun ownership.
#  then the center and left samples are removed.)
model_dict = {
    "baseline": {
        "political_model": "with_center",  # "with_center", "without_center", "right_only",
        "right_only_flag": False,
    },
    "without_center": {
        "political_model": "without_center",  # "with_center", "without_center", "right_only",
        "right_only_flag": False,
    },
    "right_only": {
        "political_model": "with_center",  # "with_center", "without_center", "right_only",
        "right_only_flag": True,
    },
}


def import_data():
    df_maps = {}
    # Rand state-level data is used to estimate firearm ownership
    # We assume that firearm ownership is the same for all districts in a state
    file_path = os.path.join(
        INPUT_DATA_DIR,
        "TL-354-State-Level Estimates of Household Firearm Ownership.xlsx",
    )
    sheet_name = "State-Level Data & Factor Score"
    df_rand_firearm = pd.read_excel(file_path, sheet_name=sheet_name)
    df_rand_firearm = df_rand_firearm[df_rand_firearm["Year"] == 2016]
    df_map = {
        "df": df_rand_firearm,
        "label": "rand_firearm",
        "file_path": file_path,
        "data_source": "https://www.rand.org/content/dam/rand/pubs/tools/TL300/TL354/RAND_TL354.database.zip",
    }
    df_maps[df_map["label"]] = df_map

    # Pew state-level data is used to estimate population in the center
    file_path = os.path.join(INPUT_DATA_DIR, "pew_party_affiliation_2014.xlsx")
    sheet_name = "data"
    df_pew_party_affiliation = pd.read_excel(file_path, sheet_name=sheet_name)
    df_map = {
        "df": df_pew_party_affiliation,
        "label": "pew_party_affiliation",
        "file_path": file_path,
        "data_source": "https://www.pewresearch.org/religion/religious-landscape-study/compare/party-affiliation/by/state#party-affiliation",
    }
    df_maps[df_map["label"]] = df_map

    # fivethirtyeight data is used to estimate partisan lean
    file_path = os.path.join(
        INPUT_DATA_DIR, "fivethirtyeight_partisan_lean_DISTRICTS.csv"
    )
    df_partisan_lean = pd.read_csv(file_path)
    df_map = {
        "df": df_partisan_lean,
        "label": "partisan_lean",
        "file_path": file_path,
        "data_source": "https://raw.githubusercontent.com/fivethirtyeight/data/master/partisan-lean/fivethirtyeight_partisan_lean_DISTRICTS.csv",
    }
    df_maps[df_map["label"]] = df_map

    # fivethirtyeight data is used for state abbreviations lookups
    file_path = os.path.join(INPUT_DATA_DIR, "delegate_targets.csv")
    df_state_abbrev = pd.read_csv(file_path)
    df_map = {
        "df": df_state_abbrev,
        "label": "state_abbrev",
        "file_path": file_path,
        "data_source": "https://raw.githubusercontent.com/fivethirtyeight/data/master/gop-delegate-benchmarks-2024/delegate_targets.csv",
    }
    df_maps[df_map["label"]] = df_map
    return df_maps


def combined_data(df_map):
    # merge dataframes, starting with partisan lean by district
    df = df_maps["partisan_lean"]["df"].copy()
    df.columns = ["district", "partisan_lean (%)"]

    # add state name
    abbrev_to_state_lookup = (
        df_maps["state_abbrev"]["df"].set_index("state_abb").to_dict()["state_name"]
    )
    df["state_name"] = df["district"].apply(lambda x: abbrev_to_state_lookup[x[:2]])

    # add Pew party affiliation/lean
    df = pd.merge(
        df,
        df_maps["pew_party_affiliation"]["df"],
        left_on="state_name",
        right_on="State",
        how="inner",
    )
    df = df.drop("State", axis=1)
    df = df.rename(columns={"Sample size": "lean sample size"})

    # add Rand firearm ownership
    rand_map = {
        "STATE": "state_name",
        "HFR": "firearm ownership",
        "HFR_se": "firearm ownership margin of error",
    }
    df_rand_firearm = df_maps["rand_firearm"]["df"].rename(columns=rand_map)
    df_rand_firearm = df_rand_firearm[list(rand_map.values())]
    df = pd.merge(df, df_rand_firearm, on="state_name", how="inner")
    return df


def add_model_variables(
    df: pd.DataFrame, model_dict: dict, model: str = "baseline"
) -> pd.DataFrame:
    gun_variable_list = []
    for key in ["has_gun_in_house", "is_gun_owner"]:
        value = str(True)
        variable = f"{key}_{value}"
        gun_variable_list.append(variable)
        df[variable] = df["firearm ownership"]
        value = str(False)
        variable = f"{key}_{value}"
        gun_variable_list.append(variable)
        df[variable] = 1 - df["firearm ownership"]

    lean_map = {
        "right": "Republican/lean Rep.",
        "center": "No lean",
        "left": "Democrat/lean Dem.",
    }

    lean_variable_list = []
    key = "political_lean"
    value = "center"
    variable = f"{key}_{value}"
    lean_variable_list.append(variable)
    if model_dict[model]["political_model"] == "with_center":
        df[variable] = df[lean_map["center"]]
    else:
        df[variable] = 0.0

    value = "right"
    variable = f"{key}_{value}"
    lean_variable_list.append(variable)
    if model_dict[model]["political_model"] == "right_only":
        df[variable] = 1.0
    else:
        if model_dict[model]["political_model"] == "with_center":
            df[variable] = (1 - df[lean_map["center"]]) / 2 - df[
                "partisan_lean (%)"
            ] / 100 / 2
        else:
            df[variable] = 0.5 - df["partisan_lean (%)"] / 100 / 2

    value = "left"
    variable = f"{key}_{value}"
    lean_variable_list.append(variable)
    df[variable] = (1 - df[lean_map["center"]]) / 2 + df["partisan_lean (%)"] / 100 / 2
    if model_dict[model]["political_model"] == "right_only":
        df[variable] = 0.0
    else:
        if model_dict[model]["political_model"] == "with_center":
            df[variable] = (1 - df[lean_map["center"]]) / 2 + df[
                "partisan_lean (%)"
            ] / 100 / 2
        else:
            df[variable] = 0.5 + df["partisan_lean (%)"] / 100 / 2

    for variable in gun_variable_list + lean_variable_list:
        min_value = df[variable].min()
        if min_value < 0:
            raise ValueError(f"{variable} has negative values ({min_value})")
        max_value = df[variable].max()
        if max_value > 1:
            raise ValueError(f"{variable} has values greater than 1 ({max_value})")

    df["lean_sum"] = df[lean_variable_list].sum(axis=1)
    df["lean_sum"] = df["lean_sum"] - 1.0
    df["lean_sum"] = df["lean_sum"].abs()
    max_lean_sum = df["lean_sum"].max()
    if max_lean_sum > 1e-10:
        raise ValueError(
            f"sum of lean variables is not close to 1. max difference is: {df['lean_sum'].max()}"
        )
    df["right_only_flag"] = model_dict[model]["right_only_flag"]
    return df


def add_state_rows(df: pd.DataFrame) -> pd.DataFrame:
    # add rows for each state

    def custom_aggregator(group):
        agg_dict = {}
        for column in group.columns:
            if pd.api.types.is_numeric_dtype(group[column].dtype):
                agg_dict[column] = group[column].mean()
            elif pd.api.types.is_bool_dtype(group[column].dtype):
                agg_dict[column] = group[column].mean()
            elif pd.api.types.is_string_dtype(group[column].dtype):
                mode_val = group[column].mode()
                agg_dict[column] = mode_val.iloc[0] if not mode_val.empty else None
            else:
                agg_dict[column] = (
                    group[column].mode().iloc[0]
                    if not group[column].mode().empty
                    else None
                )
        return pd.Series(agg_dict)

    df["body"] = "house"

    print("adding state rows")
    # state_df = df.groupby("state_name", as_index=False, include_groups=False).apply(
    state_df = df.groupby("state_name", as_index=False).apply(
        custom_aggregator, include_groups=False
    )
    state_df.reset_index(
        drop=True, inplace=True
    )  # Ensures 'state_name' is not duplicated
    state_df["body"] = "senate"
    state_df.to_csv("state_df.csv", index=False)
    state_df["district"] = state_df["state_name"]
    df = pd.concat([df, state_df])
    return df


if __name__ == "__main__":
    df_maps = import_data()
    # print metadata for each dataframe and the first few rows
    print("data sources:")
    for df_map in df_maps.values():
        print()
        ic(df_map["label"])
        ic(df_map["file_path"])
        ic(df_map["data_source"])

    for model in model_dict.keys():
        print(f"Processing model: {model}")
        df = combined_data(df_maps)
        df = add_model_variables(df=df, model_dict=model_dict, model=model)
        df = add_state_rows(df)
        df["model"] = model
        output_data_path = os.path.join(
            PROCESSED_DATA_DIR, f"house_district_data_{model}.csv"
        )
        df.to_csv(output_data_path, index=False)
        print(f"data written to: {output_data_path}")
