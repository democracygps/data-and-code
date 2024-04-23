"""
run_models.py

Estimates support and uncertainty for gun control measures in house districts.

Note: this script contains legacy code that explicitly estimated the uncertainty
      using a Monte Carlo method. This method was replaced by a more efficient
      method that uses the standard error of the mean.
"""

import statistics
import pandas as pd
from icecream import ic

import os
import sys

from utils import (
    get_fractions,
    estimate_monte_carlo_uncertainties,
    calculate_margin_of_error,
)
from constants import PROCESSED_DATA_DIR, SURVEY_CODED_DATA_PATH

DEFAULT_HOUSE_DISTRICT_DATA = "house_district_data_baseline.csv"

# TESTING FLAG FOR DEVELOPMENT runs a small number of districts
TESTING = True
TESTING = False

# DEBUG FLAG FOR DEVELOPMENT raises exceptions for debugging
DEBUG = True
DEBUG = False

# FULL_MC FLAG FOR DEVELOPMENT uses Monte Carlo method for uncertainty
# FULL_MC has been deprecated and is no longer used
FULL_MC = False


def sample_table(df: pd.DataFrame, sample_dict: dict):
    """
    sample_table based on one column

    sample_dict = {
        "column": "is_gun_owner",
        "sample_size": 10000,
        "weights": {
            True: 0.6,
            False: 0.4,
        },
    }

    """
    column = sample_dict["column"]
    sample_size = sample_dict["sample_size"]
    target_weights = sample_dict["weights"]
    target_counts = {key: value * sample_size for key, value in target_weights.items()}
    df_list = []
    for key, value in target_counts.items():
        df_list.append(df[df[column] == key].sample(n=int(value), replace=True))
    result = pd.concat(df_list)
    result = result.reset_index(drop=True)
    return result


def create_representative_population(
    df_poll,
    predictors,
    row,
    sample_size,
    sample_convergence_criteria,
    sample_convergence_max_count,
):
    """
    Creates a new representative population based on the provided parameters.

    Parameters:
        df_poll (DataFrame): The DataFrame containing polling data.
        predictors (list): List of predictor variables.
        row (Series): The row containing the expected fractions for each predictor value.
        sample_size (int): The size of the sample to be created.
        sample_convergence_criteria (float): The convergence criteria for the sample.
        sample_convergence_max_count (int): The maximum number of iterations for convergence.

    Returns:
        DataFrame: The DataFrame representing the new representative population.
    """

    df_poll_sample = df_poll.copy()

    for _ in range(sample_convergence_max_count):
        converged = True
        for predictor in predictors:
            sample_dict = {
                "column": predictor,
                "sample_size": sample_size,
            }
            values = list(df_poll[predictor].unique())
            weights = {}
            for value in values:
                weights[value] = row[f"{predictor}_{value}"]
            sample_dict["weights"] = weights
            df_poll_sample = sample_table(
                df_poll_sample, sample_dict
            )  # You need to define this function
        for predictor in predictors:
            values = list(df_poll[predictor].unique())
            actual = get_fractions(
                df_poll_sample, [predictor], as_dict=True
            )  # You need to define this function
            for value in values:
                if value in actual:
                    delta = abs(actual[value] - row[f"{predictor}_{value}"])
                    if delta >= sample_convergence_criteria:
                        converged = False
                        break
        if converged:
            break
    if not converged:
        raise ValueError(
            f"Convergence failed for {predictor} {value}. "
            f"Actual convergence ({delta} > threshold ({sample_convergence_criteria})). "
            f"Try increasing sample_size ({sample_size}) or sample_convergence_criteria ({sample_convergence_criteria}). "
            f"Predictor {predictor} {value} has actual fraction {actual[value]}; "
            f"expected fraction {row[f'{predictor}_{value}']}."
        )

    return df_poll_sample


def validate_row(row, validation_threshold):
    """
    Validate a single row of data against a validation threshold.

    Parameters:
        row (Series): The row of data to validate.
        validation_threshold (float): The threshold value for validation.

    Raises:
        ValueError: If validation fails for any condition.
    """
    # Validate stderr
    # @TODO: add value of cell to error message
    for column in row.index:
        if column.endswith("_stderr"):
            if row[column] >= validation_threshold:
                raise ValueError(f"Max {column} >= {validation_threshold}")

            if row[column] <= 0.0:
                raise ValueError(f"Min {column} <= 0.0")

    # Compare and validate means
    mean_columns_1 = [
        column for column in row.index if column.endswith("True_full_sample_mean")
    ]
    mean_columns_2 = [
        column for column in row.index if column.endswith("mean_True_mean")
    ]

    mean_columns_1.sort()
    mean_columns_2.sort()

    for column_1, column_2 in zip(mean_columns_1, mean_columns_2):
        delta = row[column_1] - row[column_2]
        delta = abs(delta)

        if delta >= validation_threshold:
            raise ValueError(f"Max {column_1} - {column_2} >= {validation_threshold}")

        if delta <= 0.0:
            raise ValueError(f"Min {column_1} - {column_2} <= 0.0")


def reorder_columns(df_districts):
    columns = df_districts.columns
    new_columns = []
    for column in columns:
        if (
            (not column.endswith("_mean"))
            and (not column.endswith("_stderr"))
            and (not "_full_sample" in column)
        ):
            new_columns.append(column)
    if FULL_MC:
        for column in columns:
            if column.endswith("_True_mean"):
                new_columns.append(column)
        for column in columns:
            if column.endswith("_True_stderr"):
                new_columns.append(column)
    for column in columns:
        if "_full_sample" in column:
            new_columns.append(column)
    df_districts = df_districts[new_columns]
    return df_districts


def run_samples(
    df_poll: pd.DataFrame,
    df_districts: pd.DataFrame,
    state_predictors: list,
    district_predictors: list,
    outcomes: list,
    target_sample_size: int,
    num_samples: int,
    poll_size: int,
    right_only_poll_size: int,
    inplace: bool = True,
    sample_convergence_max_count: int = 100,
    sample_convergence_criteria: float = 0.005,
    validation_threshold: float = 0.01,
    # num_poll_samples: int = 2000,
    # num_poll_samples: int = 200,
    num_poll_samples: int = 1,
    poll_replacement: bool = True,
    confidence_level: float = 0.95,
    data_output_path: str = None,
):
    """
    Run samples reweighted for predictors

    Some logic for partisan_lean and firearm ownership is hardcoded

    returns df_house_data with new columns

    Args:
    - df_poll: pd.DataFrame with polling data
    - df_districts: pd.DataFrame with district data
    - simultaneous_predictors: list of predictors, all sampled at once
    - sequential_predictors: list of predictors, sampled sequentially
    - outcomes: list of outcomes
    - sample_size: int
    - num_samples: int
    - inplace: bool
    """
    if not data_output_path:
        raise ValueError("data_output_path is required")
    if not inplace:
        df_districts = df_districts.copy()
    all_predictors = set(state_predictors + district_predictors)
    all_predictors = list(all_predictors)
    # we choose to resample states to keep code simple and because cost is low
    first_row = True
    df_districts["errors_short"] = "None"
    error_count = 0
    ic(df_districts["body"].unique())
    df_states = df_districts[df_districts["body"] == "senate"]
    for row_index, row in df_districts.iterrows():
        print(f"row_index: {row_index}; district: {row.iloc[0]}")
        try:
            results = []
            sample_size = target_sample_size
            # differences between samples should be small
            # if row["right_only_flag"]:
            #     sample_size = target_sample_size / row["political_lean_right"]
            df_state_sample = df_poll.copy()
            state_row = df_states[df_states["state_name"] == row["state_name"]].iloc[0]
            df_state_sample = create_representative_population(
                df_state_sample,
                all_predictors,
                state_row,
                sample_size,
                sample_convergence_criteria,
                sample_convergence_max_count,
            )
            for _ in range(num_samples):
                df_poll_sample = df_state_sample.copy()
                df_poll_sample = create_representative_population(
                    df_poll_sample,
                    district_predictors,
                    row,
                    sample_size,
                    sample_convergence_criteria,
                    sample_convergence_max_count,
                )
                if row["right_only_flag"]:
                    df_poll_sample = df_poll_sample[df_poll_sample["leans_right"]]
                if first_row:
                    df_districts["gun_owner_fraction_right_only"] = 0.0
                df_sample_fractions = get_fractions(
                    df_poll_sample,
                    ["is_gun_owner"],
                    as_dict=True,
                )
                df_districts.at[row_index, "gun_owner_fraction_right_only"] = (
                    df_sample_fractions[True]
                )
                # @TODO: not FULL_MC case can be simplified
                # if not FULL_MC:
                result = {}
                for outcome in outcomes:
                    result[outcome] = get_fractions(
                        df_poll_sample, [outcome], as_dict=True
                    )
                if True:
                    result = estimate_monte_carlo_uncertainties(
                        df_poll_sample,
                        outcomes,
                        confidence_level,
                        poll_size,
                        num_poll_samples,
                        poll_replacement,
                        result,
                    )
                results.append(result)

            summary = {}
            for result in results:
                for key, value in result.items():
                    if "True" not in key:
                        continue
                    if key not in summary:
                        summary[key] = {}
                        summary[key]["items"] = []
                    summary[key]["items"].append(value)
            for key, value in summary.items():
                mean = sum(value["items"]) / len(value["items"])
                stderr = statistics.stdev(value["items"])
                # create new columns
                if first_row:
                    df_districts[f"{key}_mean"] = 0.0
                    df_districts[f"{key}_stderr"] = 0.0
                df_districts.at[row_index, f"{key}_mean"] = mean
                df_districts.at[row_index, f"{key}_stderr"] = stderr
            df_sample_fractions = get_fractions(
                df_poll_sample, all_predictors, multiindex=True
            )
            df_sample_fractions = df_sample_fractions.to_frame().rename(
                columns={0: "fraction"}
            )
            ic(df_sample_fractions)
            if first_row:
                df_districts = reorder_columns(df_districts)
            if first_row:
                first_row = False
            if FULL_MC:
                validate_row(df_districts.iloc[row_index], validation_threshold)
        except Exception as exp:
            if DEBUG:
                raise exp
            error_count += 1
            print(exp)
            exception_str_short = str(exp)
            df_districts.at[row_index, "errors_short"] = exception_str_short
        df_districts.to_csv(data_output_path, index=False)
        print(f"data written to {data_output_path}")
        print(f"error_count: {error_count}")
        print(f"row_index: {row_index}; district: {row.iloc[0]}")
    df_districts = reorder_columns(df_districts)
    if not FULL_MC:
        columns = df_districts.columns
        mean_columns = [
            column for column in columns if column.endswith("True_full_sample_mean")
        ]
        stderr_columns = [
            column for column in columns if column.endswith("_True_full_sample_stderr")
        ]
        assert len(mean_columns) == len(stderr_columns)
        mean_columns.sort()
        stderr_columns.sort()
        effective_poll_size = poll_size
        if df_districts.iloc[0]["right_only_flag"]:
            effective_poll_size = right_only_poll_size
        for mean_column, stderr_column in zip(mean_columns, stderr_columns):
            error_results = calculate_margin_of_error(
                df_districts[mean_column],
                effective_poll_size,
            )
            df_districts[stderr_column] = error_results
    if df_districts.iloc[0]["right_only_flag"]:
        # use senate: average lean
        # reset district and body for plotting
        for state_name in df_districts["state_name"].unique():
            state_row = df_districts[df_districts["state_name"] == state_name]
            state_row = state_row[state_row["body"] == "senate"].iloc[0]
            state_slice = df_districts[df_districts["state_name"] == state_name]
            for index, row in state_slice.iterrows():
                updated_row = state_row.copy()
                updated_row["district"] = row["district"]
                updated_row["body"] = row["body"]
                df_districts.loc[index] = updated_row

    return df_districts


if __name__ == "__main__":
    df_poll = pd.read_csv(SURVEY_CODED_DATA_PATH)

    if len(sys.argv) == 1:
        house_data_path = os.path.join(PROCESSED_DATA_DIR, DEFAULT_HOUSE_DISTRICT_DATA)
    else:
        house_data_path = sys.argv[1]
        print(f"Using house_data_path: {house_data_path}")
        # input("Press Enter to continue...")
    house_data_output_path = house_data_path.replace(".csv", "_modelled_fast.csv")
    print(f"house_data_path: {house_data_path}")
    print(f"house_data_output_path: {house_data_output_path}")

    df_house_data = pd.read_csv(house_data_path)

    if TESTING:
        df_house_data = df_house_data.sort_values(by=["body"], ascending=False)
        df_house_data = df_house_data.sort_values(by=["state_name"])
        df_house_data = df_house_data.head(20)

    df_house_data = run_samples(
        df_poll=df_poll,
        df_districts=df_house_data,
        state_predictors=["is_gun_owner"],
        district_predictors=["political_lean"],
        outcomes=[
            "for_assault_weapon_ban",
            "for_large_magazine_ban",
            "for_background_checks",
            "for_safe_storage_laws",
        ],
        target_sample_size=20000,
        num_samples=2,
        poll_size=500,
        right_only_poll_size=168,
        inplace=False,
        data_output_path=house_data_output_path,
    )
    print("df_house_data")
    print(df_house_data.head(3))
    df_house_data.to_csv(house_data_output_path, index=False)
    print(f"data written to {house_data_output_path}")
