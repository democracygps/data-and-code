import pandas as pd
import plotly.graph_objects as go
from icecream import ic

import os

from constants import PROCESSED_DATA_DIR, PLOTS_DIR


file_root = PROCESSED_DATA_DIR
file_dict = {
    "baseline": "house_district_data_baseline_modelled_fast.csv",
    "right_only": "house_district_data_right_only_modelled_fast.csv",
    # "without_center": "house_district_data_without_center_modelled_fast.csv",
}

# Show error bars?
SHOW_ERROR_BARS = True
SHOW_ERROR_BARS = False

# Which body to use?
BODY = "house"
# BODY = "senate"

# show each file as a separate plot?
MERGE_PLOTS = True

# How many points to plot?
NUM_POINTS = 50

# which column to use for thinning and sorting?
POINT_THIN_COLUMN = "partisan_lean (%)"

PLOT_NAME = ""
if SHOW_ERROR_BARS:
    X_AXIS_NAME = "Support for Assault Weapon Laws (%)"
else:
    X_AXIS_NAME = "Minimum Support for Assault Weapon Laws (%)"
if BODY == "house":
    Y_AXIS_NAME = "Percent of US Congressional Districts"
else:
    Y_AXIS_NAME = "Percent of US Senate Districts"

PLOT_FILE_NAME = f"assault-weapon-law-support-{BODY}"

SYMBOL_LIST = ["circle", "cross", "diamond", "square"]
COLOR_LIST = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]  # Blue, Green, Red, Orange


def merge_data():
    dfs = {}
    for name, file in file_dict.items():
        df = pd.read_csv(os.path.join(file_root, file))
        df = df[df["body"] == BODY]
        freq = len(df) // NUM_POINTS
        freq = max(1, freq)
        df = df.iloc[::freq, :]
        df.sort_values(by=POINT_THIN_COLUMN, inplace=True)
        dfs[name] = df
    return dfs


def make_plots(dfs):
    def hover_info(row, data_name, mean_col):
        result = (
            f"Data: {data_name}<br>"
            f"Support: {row[mean_col]:.1f}%<br>"
            f"State: {row['state_name']}<br>"
        )
        if BODY == "house":
            result += f"District: {row['district']}<br>"
        result += (
            f"Lean: {row['partisan_lean (%)']:.1f}%<br>"
            f"Gun ownership: {row['gun_owner_fraction_right_only']:.1f}%<br>"
        )
        return result

    scatter_list = []
    df_index = -1
    for name, df in dfs.items():
        df_index += 1
        mean_columns = [
            col for col in df.columns if col.endswith("_True_full_sample_mean")
        ]
        stderr_columns = [
            col for col in df.columns if col.endswith("_True_full_sample_stderr")
        ]
        max_col = len(mean_columns)
        if not MERGE_PLOTS:
            scatter_list = []
        column_index = -1
        df["gun_owner_fraction_right_only"] = df["gun_owner_fraction_right_only"] * 100
        for mean_col, stderr_col in zip(
            mean_columns[:max_col], stderr_columns[:max_col]
        ):
            column_index += 1
            df.sort_values(by=mean_col, inplace=True)  # Sort inplace
            df["index"] = range(len(df))
            df["index"] = df["index"] / len(df) * 100
            df[mean_col] = df[mean_col] * 100
            df[stderr_col] = df[stderr_col] * 100
            data_name = mean_col.replace("_True_full_sample_mean", "") + f"-{name}"
            df["hover_info"] = df.apply(hover_info, args=(data_name, mean_col), axis=1)
            scatter_dict = {
                "y": df["index"],
                "mode": "markers",
                "name": data_name,
                "marker": dict(
                    color=COLOR_LIST[column_index], symbol=SYMBOL_LIST[df_index]
                ),
                "text": df["hover_info"],
                "hoverinfo": "text",
            }
            if SHOW_ERROR_BARS:
                scatter_dict["x"] = df[mean_col]
                scatter_dict["error_x"] = {
                    "type": "data",
                    "array": df[stderr_col],
                    "visible": True,  # Set to True to make the error bars visible
                }
            else:
                scatter_dict["x"] = df[mean_col] - df[stderr_col]
            scatter_list.append(go.Scatter(**scatter_dict))
        if not MERGE_PLOTS:
            fig = go.Figure(data=scatter_list)
            fig.update_layout(
                title=f"{name} {PLOT_NAME}",
                xaxis_title=X_AXIS_NAME,
                yaxis_title=Y_AXIS_NAME,
            )
            fig.show()

    if MERGE_PLOTS:
        fig = go.Figure(data=scatter_list)
        fig.update_layout(
            title=f"{PLOT_NAME}", xaxis_title=X_AXIS_NAME, yaxis_title=Y_AXIS_NAME
        )
        fig.show()
    return fig


if __name__ == "__main__":
    dfs = merge_data()
    fig = make_plots(dfs)
    # Save the plot as an HTML file
    file_path = os.path.join(PLOTS_DIR, PLOT_FILE_NAME + ".html")
    fig.write_html(file_path)
    print(f"Plot saved to: {file_path}")

## simplified code for ChatGPT interactions and modifications:

# trace3 = go.Scatter(
#     x=df["x_values_for_trace_3"],
#     y=df["y_values_for_trace_3"],
#     mode="markers",
#     name="Trace 3",
#     text=df["tooltip_text_for_trace_3"],  # Replace "tooltip_text_for_trace_3" with the text you want to display in tooltips for trace 3
#     hoverinfo="text"
# )

# # Create the figure and add the traces
# fig = go.Figure(data=[trace1, trace2, trace3])

# # Show the plot
# fig.show()
