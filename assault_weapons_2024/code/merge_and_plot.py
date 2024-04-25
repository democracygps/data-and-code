import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

from icecream import ic

import os

from constants import PROCESSED_DATA_DIR, PLOTS_DIR


file_root = PROCESSED_DATA_DIR
file_dict = {
    "All Voters": "house_district_data_baseline_modelled_fast.csv",
    "Republican and Conservative Voters Only": "house_district_data_right_only_modelled_fast.csv",
    # "without_center": "house_district_data_without_center_modelled_fast.csv",
}

# Show error bars?
SHOW_ERROR_BARS = True
# SHOW_ERROR_BARS = False

# Show annotations?
SHOW_ANNOTATIONS = True
# SHOW_ANNOTATIONS = False

# Which body to use?
BODY = "house"
# BODY = "senate"

# show each file as a separate plot?
MERGE_PLOTS = True
MERGE_PLOTS = False

# How many points to plot?
NUM_POINTS = 50

# which column to use for thinning and sorting?
POINT_THIN_COLUMN = "partisan_lean (%)"

PLOT_NAME = ""
if SHOW_ERROR_BARS:
    Y_AXIS_NAME = "Support for Assault Weapon Laws (%)"
else:
    Y_AXIS_NAME = "Minimum Support for Assault Weapon Laws (%)"

if BODY == "house":
    X_AXIS_NAME = (
        "Percent of US Congressional Districts (%)"
    )
else:
    X_AXIS_NAME = (
        "Percent of US Senate Districts (%)"
    )

PLOT_FILE_NAME = f"assault-weapon-law-support-{BODY}"

SYMBOL_LIST = ["circle", "cross", "diamond", "square"]
SYMBOL_LIST = ["circle", "circle", "circle", "circle"]
# Hexadecimal color list
# COLOR_LIST = ["#1f77b4", "#2ca02c", "#d62728", "#ff7f0e"]  # Blue, Green, Red, Orange

# RGB equivalent
COLOR_LIST = [
    'rgb(31, 119, 180)',  # RGB for "#1f77b4"
    'rgb(44, 160, 44)',  # RGB for "#2ca02c"
    'rgb(214, 39, 40)',  # RGB for "#d62728"
    'rgb(255, 127, 14)',  # RGB for "#ff7f0e"
]
COLOR_LIST_ERRORS = [
    'rgba(31, 119, 180, 0.3)',  # RGB for "#1f77b4"
    'rgba(44, 160, 44, 0.3)',  # RGB for "#2ca02c"
    'rgba(214, 39, 40, 0.3)',  # RGB for "#d62728"
    'rgba(255, 127, 14, 0.3)',  # RGB for "#ff7f0e"
]

SORTED_COLUMNS = {
    'for_background_checks_True_full_sample_mean': "For Background Checks",
    'for_safe_storage_laws_True_full_sample_mean': "For Safe Storage Laws",
    'for_assault_weapon_ban_True_full_sample_mean': "For Assault Weapon Ban",
    'for_large_magazine_ban_True_full_sample_mean': "For Large Magazine Ban",
}

SORTED_ERROR_COLUMNS = [
    'for_background_checks_True_full_sample_stderr',
    'for_safe_storage_laws_True_full_sample_stderr',
    'for_assault_weapon_ban_True_full_sample_stderr',
    'for_large_magazine_ban_True_full_sample_stderr',
]

ANNOTATION_LIST = [
    {
        "index": 0,
        "data_name_index": 0,
        "ax": 80,
        "ay": -50,
    },
    {
        "index": 27,
        "data_name_index": 3,
        "ax": 20,
        "ay": 40,
    },
    {
        "index": -1,
        "data_name_index": 1,
        "xref": "x",
        "yref": "y",
        "text": "Annotation 3",
        "showarrow": True,
        "arrowhead": 7,
        "ax": -80,
        "ay": -90,
    },

]

def merge_data():
    dfs = {}
    for name, file in file_dict.items():
        df = pd.read_csv(os.path.join(file_root, file))
        df = df[df["body"] == BODY]
        freq = len(df) // NUM_POINTS
        freq = max(1, freq)
        df = df.iloc[::freq, :]
        df.sort_values(by=POINT_THIN_COLUMN, inplace=True)
        if SORTED_COLUMNS:
            df = df.rename(columns=SORTED_COLUMNS)
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

    def label_info(row, data_name, mean_col):
        result = (
            f"{data_name} ({row['district']})<br>"
            f"Support: {row[mean_col]:.0f}%; "
            f"Lean: {row['partisan_lean (%)']:.0f}%; "
            f"Gun ownership: {row['gun_owner_fraction_right_only']:.0f}%"
        )
        return result


    scatter_list = []
    df_index = -1
    for name, df in dfs.items():
        df_index += 1
        mean_columns = [
            col for col in df.columns if col.endswith("_True_full_sample_mean")
        ]
        if SORTED_COLUMNS:
            mean_columns = list(SORTED_COLUMNS.values())
        stderr_columns = [
            col for col in df.columns if col.endswith("_True_full_sample_stderr")
        ]
        if SORTED_ERROR_COLUMNS:
            stderr_columns = SORTED_ERROR_COLUMNS
        max_col = len(mean_columns)
        if not MERGE_PLOTS:
            scatter_list = []
        column_index = -1
        df["gun_owner_fraction_right_only"] = df["gun_owner_fraction_right_only"] * 100
        for mean_col, stderr_col in zip(
            mean_columns[:max_col], stderr_columns[:max_col]
        ):
            column_index += 1
            df.sort_values(by=mean_col, inplace=True)
            df.reset_index(inplace=True, drop=True)
            df["index"] = range(len(df))
            df["index"] = df["index"] / len(df) * 100
            df[mean_col] = df[mean_col] * 100
            df[stderr_col] = df[stderr_col] * 100
            if MERGE_PLOTS:
                data_name = mean_col.replace("_True_full_sample_mean", "") + f"-{name}"
            else:
                data_name = mean_col
            df = df.sort_values(by=mean_col)
            df["hover_info"] = df.apply(hover_info, args=(data_name, mean_col), axis=1)
            scatter_dict = {
                "x": df["index"],
                "mode": "markers",
                "name": data_name,
                "marker": dict(
                    color=COLOR_LIST[column_index], symbol=SYMBOL_LIST[df_index]
                ),
                "text": df["hover_info"],
                "hoverinfo": "text",
            }
            if SHOW_ERROR_BARS:
                scatter_dict["mode"] = "markers"
                scatter_dict["y"] = df[mean_col]
                scatter_dict["error_y"] = {
                    "type": "data",
                    "array": df[stderr_col],
                    "color": COLOR_LIST_ERRORS[column_index],
                    "thickness": 5,  # Thin error bars
                    "width": 0,  # Width of the caps
                    "visible": True,  # Set to True to make the error bars visible
                }
            else:
                scatter_dict["y"] = df[mean_col] - df[stderr_col]
            scatter_list.append(go.Scatter(**scatter_dict))
        if not MERGE_PLOTS:
            fig = go.Figure(data=scatter_list)
            for annotation in ANNOTATION_LIST:
                index = annotation["index"]
                data_name = list(SORTED_COLUMNS.values())[annotation["data_name_index"]]
                df.sort_values(by=data_name, inplace=True)
                df.reset_index(inplace=True, drop=True)
                df["index"] = range(len(df))
                df["index"] = df["index"] / len(df) * 100
                ic(index)
                ic(df.shape)
                ic(df.head())
                row = df.iloc[index]
                ic(row)
                text = label_info(row, data_name, data_name)
                ic(df["index"].iloc[index])
                ic(df[data_name].iloc[index])
                fig.add_annotation(
                    x=df["index"].iloc[index],
                    y=df[data_name].iloc[index],
                    text=text,
                    # text="foo",
                    align="center",
                    showarrow=True,
                    ax=annotation["ax"],
                    ay=annotation["ay"],
                    arrowhead=2,  # arrowhead style
                    arrowsize=1,  # arrow size
                    arrowwidth=2,  # arrow width
                    arrowcolor='black',  # arrow color
                    bordercolor='black',  # box border color
                    borderwidth=2,  # box border width
                    borderpad=4,  # box padding
                    bgcolor='lightgrey',  # box background color
                )
                # fig.add_annotation(
#     x=x[2],  # x-coordinate of the point to label
#     y=y[2],  # y-coordinate of the point to label
#     text='C',  # label text
#     showarrow=True,  # show arrow
#     arrowhead=2,  # arrowhead style
#     arrowsize=1,  # arrow size
#     arrowwidth=2,  # arrow width
#     arrowcolor='blue',  # arrow color
#     ax=20,  # arrow x-offset
#     ay=-40,  # arrow y-offset
#     bordercolor='black',  # box border color
#     borderwidth=2,  # box border width
#     borderpad=4,  # box padding
#     bgcolor='lightgrey',  # box background color
# )
            fig.update_layout(
                title=f"{name}",
                xaxis_title=X_AXIS_NAME,
                yaxis_title=Y_AXIS_NAME,
            )
            fig.show()
            file_path = os.path.join(PLOTS_DIR, f"{PLOT_FILE_NAME}-{name}.png")
            pio.write_image(fig, file_path, format='png')
            print(f"Plot saved to: {file_path}")


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
