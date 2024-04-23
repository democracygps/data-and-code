import math
import numpy as np
import pandas as pd
from icecream import ic

import json
import hashlib
import pprint


def dict_add_hash(dictionary: dict):
    """
    adds a hash and hash_labels to a dictionary
    """
    hash_labels = dictionary.copy()
    dictionary["hash"] = hash_dict(dictionary)
    dictionary["hash_labels"] = hash_labels
    return dictionary


def get_fractions(df: pd.DataFrame, indices, as_dict=False, multiindex=False):
    table: pd.DataFrame = df.groupby(indices).size()
    if not multiindex:
        table = table.reset_index(name="counts")
        table["fraction"] = table["counts"] / table["counts"].sum()
    else:
        table = table / table.sum()
    if as_dict:
        # return table.to_dict(orient="records")
        return table.set_index(indices)["fraction"].to_dict()
    return table



def hash_dict(d):
    """
    Create a consistent hash for a dictionary across Python sessions.
    """
    # Convert the dictionary to a JSON string with sorted keys
    serialized_dict = json.dumps(d, sort_keys=True)

    # Create a hash of the serialized string using sha256
    hash_object = hashlib.sha256(serialized_dict.encode())

    # Get the hexadecimal digest of the hash
    hash_hex = hash_object.hexdigest()

    return hash_hex


def estimate_monte_carlo_uncertainties(
    df,
    outcomes,
    confidence_level,
    poll_size,
    num_poll_samples,
    poll_replacement,
    result,
):
    for outcome in outcomes:
        fractions = get_fractions(df, [outcome], as_dict=True)
        for key, value in fractions.items():
            result[f"{outcome}_{key}_full_sample"] = value
    poll_results = {}
    for _ in range(num_poll_samples):
        poll_df = df.sample(n=poll_size, replace=poll_replacement)
        for outcome in outcomes:
            poll_fractions = get_fractions(poll_df, [outcome], as_dict=True)
            for key, value in poll_fractions.items():
                column_label = f"{outcome}_results_{key}"
                if column_label not in poll_results:
                    poll_results[column_label] = []
                poll_results[column_label].append(value)
    for outcome in outcomes:
        for key, value in poll_fractions.items():
            column_label = f"{outcome}_results_{key}"
            poll_results[column_label].sort()
            result[f"{outcome}_mean_{key}"] = sum(poll_results[column_label]) / len(
                poll_results[column_label]
            )
            min_sample = int((1 - confidence_level) / 2 * num_poll_samples)
            max_sample = int((1 - (1 - confidence_level) / 2) * num_poll_samples)
            std_dev = (
                poll_results[column_label][max_sample]
                - poll_results[column_label][min_sample]
            ) / 2
            result[f"{outcome}_std_dev_{key}"] = std_dev
    return result


def calculate_margin_of_error(p, n, confidence_level=0.95):
    # Z-scores for common confidence levels
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    if confidence_level not in z_scores:
        raise ValueError(
            f"Confidence level must be one of {list(z_scores.keys())}, not {confidence_level}"
        )

    # Select the Z-score for the given confidence level
    Z = z_scores.get(confidence_level, 1.96)  # Default to 95% if not specified
    moe = Z * np.sqrt((p * (1 - p)) / n)
    return moe


def analyze_posterior(posterior_samples):
    # print(len(posterior_samples))
    # print(dir(posterior_samples))

    observed_samples = posterior_samples["observed"]
    observed_samples_array = np.array(observed_samples)

    # Example: print the shape to see [number of samples] x [number of observations]
    print("shape")
    print(observed_samples_array.shape)

    # Calculate the average of each row
    row_averages = np.mean(observed_samples_array, axis=1)

    # Calculate max, min, and average of the row averages
    max_average = np.max(row_averages)
    min_average = np.min(row_averages)
    overall_average = np.mean(row_averages)

    ic(max_average, min_average, overall_average)

    return {
        "max_average": max_average,
        "min_average": min_average,
        "overall_average": overall_average,
    }


if __name__ == "__main__":
    # Example usage
    sample_proportion = 0.5  # e.g., 50% of the sample chose a particular option
    sample_size = 500  # e.g., the sample size
    confidence_level = 0.90  # e.g., 90% confidence level

    margin_of_error = calculate_margin_of_error(
        sample_proportion, sample_size, confidence_level
    )
    print(f"{sample_proportion:1%} Margin of Error: {margin_of_error:.2%}")

    sample_proportion = 0.1  # e.g., 50% of the sample chose a particular option
    sample_size = 500  # e.g., the sample size
    confidence_level = 0.90  # e.g., 90% confidence level

    margin_of_error = calculate_margin_of_error(
        sample_proportion, sample_size, confidence_level
    )
    print(f"{sample_proportion:.0%} Margin of Error: {margin_of_error:.2%}")

    sample_proportion = 0.9  # e.g., 50% of the sample chose a particular option
    sample_size = 500  # e.g., the sample size
    confidence_level = 0.90  # e.g., 90% confidence level

    margin_of_error = calculate_margin_of_error(
        sample_proportion, sample_size, confidence_level
    )
    print(f"{sample_proportion:1%} Margin of Error: {margin_of_error:.2%}")
