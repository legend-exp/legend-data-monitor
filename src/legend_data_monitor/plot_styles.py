# -------------------------------------------------------------------------------
# different plot style functions called from the main one depending on parameter
# -------------------------------------------------------------------------------

import warnings

# See mapping user plot structure keywords to corresponding functions in the end of this file
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.dates import DateFormatter, date2num, num2date
from matplotlib.figure import Figure
from pandas import DataFrame, Timedelta, concat

from . import utils

warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------------------------------------------------------
# single parameter plotting functions
# -------------------------------------------------------------------------------


def plot_vs_time(
    data_channel: DataFrame,
    fig: Figure,
    ax: Axes,
    plot_info: dict,
    color=None,
    map_dict=None,
):
    # -------------------------------------------------------------------------
    # plot this data vs time
    # -------------------------------------------------------------------------

    # need to plot this way, and not data_position.plot(...) because the datetime column is of type Timestamp
    # plotting this way, to_pydatetime() converts it to type datetime which is needed for DateFormatter
    # changing the type of the column itself with the table does not work
    data_channel = data_channel.sort_values("datetime")

    # if you inspect event rate, change the 'resampled' option from 'only' (if so) to 'no'
    if plot_info["parameter"] == "event_rate" and plot_info["resampled"] == "only":
        plot_info["resampled"] = "no"

    res_col = color
    all_col = (
        color
        if plot_info["resampled"] == "no" or plot_info["parameter"] == "event_rate"
        else "darkgray"
    )

    if plot_info["resampled"] != "only":
        parameter_array = np.array(data_channel[plot_info["parameter"]])
        ax.plot(
            np.array(data_channel["datetime"].dt.to_pydatetime()),
            parameter_array[:, None],
            zorder=0,
            color=all_col,
            linewidth=1,
        )

    # -------------------------------------------------------------------------
    # plot resampled average
    # -------------------------------------------------------------------------

    if plot_info["resampled"] != "no":
        # unless event rate - already resampled and counted in some time window
        if not plot_info["parameter"] == "event_rate":
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 1 - resampling
            # resample in given time window, as start pick the first timestamp in table
            resampled = (
                data_channel.set_index("datetime")
                .resample(plot_info["time_window"], origin="start")
                .mean(numeric_only=True)
            )
            # will have datetime as index after resampling -> put back
            resampled = resampled.reset_index()
            # the timestamps in the resampled table will start from the first timestamp, and go with sampling intervals
            # I want to shift them by half sampling window, so that the resampled value is plotted in the middle time window in which it was calculated
            resampled["datetime"] = (
                resampled["datetime"] + Timedelta(plot_info["time_window"]) / 2
            )

            parameter_array = np.array(resampled[plot_info["parameter"]])
            ax.plot(
                resampled["datetime"].dt.to_pydatetime(),
                parameter_array[:, None],
                color=res_col,
                zorder=1,
                # marker="o",
                linestyle="-",
            )

            # evaluation of std bands, if enabled
            if plot_info["std"] is True:
                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 2 - std evaluation
                std_data = (
                    data_channel.set_index("datetime")
                    .resample(plot_info["time_window"], origin="start")
                    .std(numeric_only=True)
                )
                std_data = std_data.reset_index()

                # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 3 - appending std to the resampled dataframe
                std_data = std_data.rename(columns={plot_info["parameter"]: "std"})
                new_dataframe = concat(
                    [resampled, std_data[["std"]]], ignore_index=False, axis=1
                )

                ax.fill_between(
                    resampled["datetime"].dt.to_pydatetime(),
                    resampled[plot_info["parameter"]] - new_dataframe["std"],
                    resampled[plot_info["parameter"]] + new_dataframe["std"],
                    alpha=0.25,
                    color=res_col,
                )

    # -------------------------------------------------------------------------
    # beautification
    # -------------------------------------------------------------------------

    # set range if provided
    if plot_info["range"][0] is not None:
        ax.set_ylim(ymin=plot_info["range"][0])
    if plot_info["range"][1] is not None:
        ax.set_ylim(ymax=plot_info["range"][1])

    # plot the position of the two K lines
    if plot_info["event_type"] == "K_events":
        ax.axhline(y=1460.822, color="gray", linestyle="--")
        ax.axhline(y=1524.6, color="gray", linestyle="--")

    # --- time ticks/labels on x-axis
    min_x = date2num(data_channel.iloc[0]["datetime"])
    max_x = date2num(data_channel.iloc[-1]["datetime"])
    time_points = np.linspace(min_x, max_x, 10)
    labels = [
        num2date(time, tz=datetime.now().astimezone().tzinfo).strftime(
            "%Y\n%m/%d\n%H:%M"
        )
        for time in time_points
    ]

    # set ticks
    ax.set_xticks(time_points)
    ax.set_xticklabels(labels)

    # --- set labels
    fig.supxlabel("UTC Time")
    y_label = plot_info["label"]
    if plot_info["unit_label"] == "%":
        y_label += ", %"
    else:
        if (
            "(PULS01ANA)" in y_label
            or "(PULS01)" in y_label
            or "(BSLN01)" in y_label
            or "(MUON01)" in y_label
        ):
            separator = "-" if "-" in y_label else "/"
            parts = y_label.split(separator)

            if len(parts) == 2 and separator == "-":
                y_label += f" [{plot_info['unit']}]"
        else:
            y_label += f" [{plot_info['unit']}]"

    fig.supylabel(y_label)


def par_vs_ch(
    data_channel: DataFrame,
    fig: Figure,
    ax: Axes,
    plot_info: dict,
    color=None,
    map_dict=None,
):
    if len(data_channel[plot_info["parameter"]].unique()) > 1:
        utils.logger.error(
            "\033[91mYou are trying to plot multiple values for a given channel.\nThis is not possible, there should be only one unique value! Try again.\033[0m"
        )
        return

    # -------------------------------------------------------------------------
    # plot data vs channel ID
    # -------------------------------------------------------------------------
    location = data_channel["location"].unique()[0]
    position = data_channel["position"].unique()[0]
    ax.scatter(
        map_dict[str(location)][str(position)],
        data_channel[plot_info["parameter"]].unique()[0],
        color=color,
    )

    # -------------------------------------------------------------------------
    # beautification
    # -------------------------------------------------------------------------
    ax.set_ylabel("")
    ax.set_xlabel("")

    # --- set labels
    fig.supxlabel("Channel ID")
    y_label = (
        f"{plot_info['label']}, {plot_info['unit_label']}"
        if plot_info["unit_label"] == "%"
        else f"{plot_info['label']} [{plot_info['unit_label']}]"
    )
    fig.supylabel(y_label)


def plot_histo(
    data_channel: DataFrame,
    fig: Figure,
    ax: Axes,
    plot_info: dict,
    color=None,
    map_dict=None,
):
    # --- histo range
    # take full range if not specified
    x_min = (
        plot_info["range"][0]
        if plot_info["range"][0] is not None
        else data_channel[plot_info["parameter"]].min()
    )
    x_max = (
        plot_info["range"][1]
        if plot_info["range"][1] is not None
        else data_channel[plot_info["parameter"]].max()
    )

    # --- bin width
    bwidth = {"keV": 2.5}
    bin_width = bwidth[plot_info["unit"]] if plot_info["unit"] in bwidth else 1

    # Compute number of bins
    # sometimes e.g. A/E is always 0.0 => mean = 0 => var = NaN => x_min = NaN => cannot do np.arange
    # why arange tho? why not just number of bins (xmax - xmin) / binwidth?
    if not np.isnan(x_min):
        if bin_width:
            bin_edges = (
                np.arange(x_min, x_max + bin_width, bin_width / 5)
                if plot_info["unit_label"] == "%"
                else np.arange(x_min, x_max + bin_width, bin_width)
            )
        # this never happens unless somebody puts 0 in the bwidth dictionary?
        else:
            bin_edges = 50

        # -------------------------------------------------------------------------
        # Plot histogram
        data_channel[plot_info["parameter"]].plot.hist(
            bins=bin_edges,
            range=[x_min, x_max],
            histtype="step",
            linewidth=1.5,
            ax=ax,
            color=color,
        )

    # -------------------------------------------------------------------------

    # plot the position of the two K lines
    if plot_info["event_type"] == "K_events":
        ax.axvline(x=1460.822, color="gray", linestyle="--")
        ax.axvline(x=1524.6, color="gray", linestyle="--")

    ax.set_yscale("log")
    x_label = (
        f"{plot_info['label']}, {plot_info['unit_label']}"
        if plot_info["unit_label"] == "%"
        else f"{plot_info['label']} [{plot_info['unit_label']}]"
    )
    fig.supxlabel(x_label)


def plot_scatter(
    data_channel: DataFrame,
    fig: Figure,
    ax: Axes,
    plot_info: dict,
    color=None,
    map_dict=None,
):
    # plot data
    ax.scatter(
        np.array(data_channel["datetime"].dt.to_pydatetime()),
        data_channel[plot_info["parameter"]],
        color=color,
        # useful if there are overlapping points (but more difficult to see light colour points...)
        # facecolors='none',
        # edgecolors=color,
    )

    if plot_info["event_type"] == "K_events":
        ax.axhline(y=1460.822, color="gray", linestyle="--")
        ax.axhline(y=1524.6, color="gray", linestyle="--")

    # --- time ticks/labels on x-axis
    ax.xaxis.set_major_formatter(DateFormatter("%Y\n%m/%d\n%H:%M"))

    fig.supxlabel("UTC Time")
    y_label = (
        f"{plot_info['label']}, {plot_info['unit_label']}"
        if plot_info["unit_label"] == "%"
        else f"{plot_info['label']} [{plot_info['unit_label']}]"
    )
    fig.supylabel(y_label)


# -------------------------------------------------------------------------------
# multi parameter plotting functions
# -------------------------------------------------------------------------------


def plot_par_vs_par(
    data_channel: DataFrame,
    fig: Figure,
    ax: Axes,
    plot_info: dict,
    color=None,
    map_dict=None,
):
    par_x = plot_info["parameters"][0]
    par_y = plot_info["parameters"][1]

    ax.scatter(data_channel[par_x], data_channel[par_y], color=color)

    labels = []
    for param in plot_info["parameters"]:
        # construct label
        label = (
            f"{plot_info['label'][param]}, {plot_info['unit_label'][param]}"
            if plot_info["unit_label"][param] == "%"
            else f"{plot_info['label'][param]} [{plot_info['unit_label'][param]}]"
        )
        labels.append(label)

    fig.supxlabel(labels[0])
    fig.supylabel(labels[1])

    # apply range
    # parameter not in range means 1) none was given and defaulted to [None, None], or 2) this parameter was not mentioned in range
    # ? cut data before plotting, not after? could be more efficient to plot smaller data sample?
    if par_x in plot_info["range"]:
        ax.set_xlim(plot_info["range"][par_x])
    if par_y in plot_info["range"]:
        ax.set_ylim(plot_info["range"][par_y])


# !!! WORK IN PROGRESS !!!
# hard to test because A/E vs E is weird with huge ranges of strange large and negative values, kills memory with many bins
# will come back to this later after clarifying what A/E makes sense to plot
# def plot_par_vs_par_hist(data_channel: DataFrame, fig: Figure, ax: Axes, plot_info: dict, color=None):
#     # Compute number of bins
#     # 0 = x, 1 = y
#     nbins = []; ranges = []
#     # NaN check
#     # anynan = False
#     for param in plot_info["parameters"]:
#         # range
#         par_range = [data_channel[param].min(), data_channel[param].max()]

#         # bin width
#         if param == "AoE_Custom":
#             bin_width = 0.1
#             # par_range = [0,2]
#         elif plot_info["unit"][param] == "keV":
#             bin_width = 2.5
#             par_range = [0,3000] # avoid negative values
#         else:
#             bin_width = 1 # default


#         # number of bins
#         nbins.append( int( (par_range[1] - par_range[0])/bin_width ) )
#         ranges.append(par_range)
#         # sometimes e.g. A/E is always 0.0 => mean = 0 => var = NaN => x_min = NaN => cannot plot range [nan, nan]
#         # anynan = anynan or np.isnan(nbins[-1])

#     print(nbins)
#     print(ranges)
#     # if not anynan:
#     h, xedges, yedges, image = ax.hist2d(data_channel[plot_info["parameters"][0]], data_channel[plot_info["parameters"][1]], range=ranges, bins=nbins)

#     labels = []
#     for param in plot_info["parameters"]:
#         label = (
#             f"{plot_info['label'][param]}, {plot_info['unit_label'][param]}"
#             if plot_info["unit_label"][param] == "%"
#             else f"{plot_info['label'][param]} [{plot_info['unit_label'][param]}]"
#         )
#         labels.append(label)

#     fig.supxlabel(labels[0])
#     fig.supylabel(labels[1])

#     del h
#     del xedges
#     del yedges
#     del image


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# UNDER CONSTRUCTION!!!
def plot_heatmap(
    data_channel: DataFrame,
    fig: Figure,
    ax: Axes,
    plot_info: dict,
    color=None,
    map_dict=None,
):
    # some plotting settings
    xbin = int(
        (
            (
                data_channel.iloc[-1]["datetime"] - data_channel.iloc[0]["datetime"]
            ).total_seconds()
            * 1.5
        )
        / 1e3
    )
    if plot_info["parameter"] in ["energies", "energy_in_pe"]:
        col_map = "magma"
        ymin = 0
        ymax = 10
        ybin = 100
    if plot_info["parameter"] == "trigger_pos":
        col_map = "viridis"
        ymin = -200
        ymax = 10000
        ybin = 100

    # to plot spms data, we need a new dataframe with numeric-datetime column and 'unrolled'-list values column
    # (explode the list column instead of concatenating one row at a time)
    new_df = (
        data_channel[["datetime", plot_info["parameter"]]]
        .explode(plot_info["parameter"])
        .dropna(subset=[plot_info["parameter"]])
        .reset_index(drop=True)
    )

    # make sure the datetime column is in UTC (tz-aware input stays aware)
    new_df["datetime"] = pd.to_datetime(new_df["datetime"], utc=True)
    # convert to numeric values for plotting
    x_values = pd.to_numeric(
        new_df["datetime"].dt.tz_convert("UTC").dt.to_pydatetime()
    ).values
    y_values = new_df[plot_info["parameter"]]
    # plot data
    h, xedges, yedges = np.histogram2d(
        x_values,
        y_values,
        bins=[xbin, ybin],
        range=[
            [data_channel.iloc[0]["datetime"], data_channel.iloc[-1]["datetime"]],
            [ymin, ymax],
        ],
    )
    # cmap = copy(fig.get_cmap(col_map))
    # cmap.set_bad(cmap(0))

    ax.pcolor(xedges, yedges, h.T, cmap=col_map)  # norm=mpl.colors.LogNorm(),

    # TO DO: add major locators (pay attention when you have only one point!)

    # set date format
    # --- time ticks/labels on x-axis
    min_x = date2num(data_channel.iloc[0]["datetime"])
    max_x = date2num(data_channel.iloc[-1]["datetime"])
    time_points = np.linspace(min_x, max_x, 10)
    labels = [num2date(time).strftime("%Y\n%m/%d\n%H:%M") for time in time_points]

    # set ticks
    ax.set_xticks(time_points)
    ax.set_xticklabels(labels)

    fig.supxlabel("UTC Time")
    fig.supylabel(f"{plot_info['label']} [{plot_info['unit_label']}]")

    # saving x,y data into output files
    ch_dict = {
        "values": {"all": {}, "resampled": []},
        "mean": "",
        "plot_info": plot_info,
        "timestamp": {
            "all": {},
            "resampled": [],
        },
    }

    return ch_dict


# -------------------------------------------------------------------------------
# mapping user keywords to plot style functions
# -------------------------------------------------------------------------------

PLOT_STYLE = {
    "vs time": plot_vs_time,
    "vs ch": par_vs_ch,
    "histogram": plot_histo,
    "scatter": plot_scatter,
    "heatmap": plot_heatmap,
    "par vs par": plot_par_vs_par,
    # "par vs par histo": plot_par_vs_par_hist
}
