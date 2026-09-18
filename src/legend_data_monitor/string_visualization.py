# -------------------------------------------------------------------------------
# different status map functions called from the main one depending on parameter
# -------------------------------------------------------------------------------

# See mapping user plot structure keywords to corresponding functions in the end of this file


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from pandas import DataFrame, Timedelta, concat

from . import errors, plotting, utils


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# CHANNELS' STATUS FUNCTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def status_plot(subsystem, data_analysis: DataFrame, plot_info: dict, pdf: PdfPages):
    # -------------------------------------------------------------------------
    # plot a map with statuses of channels
    # -------------------------------------------------------------------------

    banner = "\33[95m" + "~" * 50 + "\33[0m"
    utils.logger.info(banner)
    utils.logger.info("\33[95m S T A T U S  M A P : %s\33[0m", plot_info["title"])
    utils.logger.info(banner)

    data_analysis = data_analysis.sort_values(["location", "position"])

    # get threshold values
    low_thr = plot_info["limits"][0]
    high_thr = plot_info["limits"][1]
    utils.logger.debug(
        f"... low threshold for {plot_info['parameter']} set at: {low_thr} {plot_info['unit_label']}"
    )
    utils.logger.debug(
        f"... high threshold for {plot_info['parameter']} set at: {high_thr} {plot_info['unit_label']}"
    )

    # define the title of the status map
    plot_title = f"{plot_info['subsystem']} - "
    if low_thr is not None or high_thr is not None:
        if low_thr is None and high_thr is not None:
            plot_title += (
                f"{plot_info['parameter']} > {high_thr} {plot_info['unit_label']}"
            )
        if low_thr is not None and high_thr is None:
            plot_title += (
                f"{plot_info['parameter']} < {low_thr} {plot_info['unit_label']}"
            )
        if low_thr is not None and high_thr is not None:
            plot_title += f"{plot_info['parameter']} < {low_thr} {plot_info['unit_label']} || {plot_info['parameter']} > {high_thr} {plot_info['unit_label']}"
    if low_thr is None and high_thr is None:
        # there is no point to check values if there are no thresholds
        utils.logger.debug("... there are no thresholds to check for. We skip this!")
        return

    summary_rows = []
    # loop over individual channels (otherwise, the problematic timestamps apply to all detectors, even the OK ones) and create a summary dataframe
    for channel, data_per_ch in data_analysis.groupby("channel", sort=False):
        # let's save some info (they could be lost after resampling, or wrongly averaged - this keeps us safe from similar bugs)
        name = (data_per_ch["name"].unique())[0]
        location = (data_per_ch["location"].unique())[0]
        position = (data_per_ch["position"].unique())[0]
        # if not the event rate, study the status looking at the resample values
        if not plot_info["parameter"] == "event_rate":
            data_per_ch = (
                data_per_ch.set_index("datetime")
                .resample(plot_info["time_window"], origin="start")
                .mean(numeric_only=True)
            )
            data_per_ch = data_per_ch.reset_index()
            data_per_ch["datetime"] = (
                data_per_ch["datetime"] + Timedelta(plot_info["time_window"]) / 2
            )

        status = 0  # -> OK detector (update it if it's out of threshold)
        # get timestamp where the interval is out of threshold
        out_thr_datetimes = np.array([], dtype="datetime64")
        if low_thr is not None or high_thr is not None:
            if low_thr is None and high_thr is not None:
                if (data_per_ch[plot_info["parameter"]] > high_thr).any():
                    status = 1  # -> problematic detector
                    out_thr_datetimes = np.append(
                        out_thr_datetimes,
                        data_per_ch.loc[
                            data_per_ch[plot_info["parameter"]] > high_thr, "datetime"
                        ].values,
                    )

            if low_thr is not None and high_thr is None:
                if (data_per_ch[plot_info["parameter"]] < low_thr).any():
                    status = 1  # -> problematic detector
                    out_thr_datetimes = np.append(
                        out_thr_datetimes,
                        data_per_ch.loc[
                            data_per_ch[plot_info["parameter"]] < low_thr, "datetime"
                        ].values,
                    )

            if low_thr is not None and high_thr is not None:
                if (data_per_ch[plot_info["parameter"]] < low_thr).any() or (
                    data_per_ch[plot_info["parameter"]] > high_thr
                ).any():
                    status = 1  # -> problematic detector
                    out_thr_datetimes = np.append(
                        out_thr_datetimes,
                        data_per_ch.loc[
                            data_per_ch[plot_info["parameter"]] > high_thr, "datetime"
                        ].values,
                    )
                    out_thr_datetimes = np.append(
                        out_thr_datetimes,
                        data_per_ch.loc[
                            data_per_ch[plot_info["parameter"]] < low_thr, "datetime"
                        ].values,
                    )

            # collect essential info (ie: channel, name, location, position, status); one dataframe is built after the loop
            summary_rows.append([channel, name, location, position, status])

        # print message with timestamps where the detector is out of threshold
        if len(out_thr_datetimes) != 0:
            out_thr_datetimes = [
                str(time).replace("T", " ")[:-10] for time in out_thr_datetimes
            ]
            utils.logger.warning(
                "\033[93mChannel %s (str. %s, pos. %s) is out of threshold at:\n%s\033[0m",
                channel,
                location,
                position,
                out_thr_datetimes,
            )

    new_dataframe = DataFrame(
        summary_rows, columns=["channel", "name", "location", "position", "status"]
    )

    # --------------------------------------------------------------------------------------------------------------------------
    # include OFF channels and see what is their status
    off_channels = subsystem.channel_map[subsystem.channel_map["status"] == "off"][
        "channel"
    ].unique()

    off_rows = []
    if len(off_channels) != 0:
        for channel in off_channels:
            # check if the channel is already in the status dataframe; if not, add a new row for it
            if channel not in new_dataframe["channel"].values:
                status_info = subsystem.channel_map[
                    subsystem.channel_map["channel"] == channel
                ]["status"].iloc[0]

                # get status info
                if status_info == "off":
                    status = 3
                if status_info == "AC":
                    status = (
                        2  # not there at the moment, leave it as a future option...
                    )

                # get position within the array + other necessary info
                name, location, position = get_info_from_channel(
                    subsystem.channel_map, channel
                )

                # collect new row for not-ON detectors; concatenated once below
                off_rows.append([channel, name, location, position, status])

    if off_rows:
        new_dataframe = concat(
            [
                new_dataframe,
                DataFrame(
                    off_rows,
                    columns=["channel", "name", "location", "position", "status"],
                ),
            ],
            ignore_index=True,
            axis=0,
        )

    # --------------------------------------------------------------------------------------------------------------------------
    # sort the dataframe according to channel ID number
    new_dataframe = new_dataframe.sort_values("channel")
    new_dataframe = new_dataframe.reset_index()
    new_dataframe = new_dataframe.drop(
        columns=["index"]
    )  # somehow, an index column appears: remove it

    # create a pivot with necessary info
    result = new_dataframe.pivot(index="position", columns="location", values="status")

    # create a fancier copy of the pivot table for the printing
    # (only for a given level of the logger, to save time in making the prettier copy)
    if utils.logger.getEffectiveLevel() is utils.logging.DEBUG:
        output_result = result.fillna("")
        output_result = output_result.replace(3.0, "-")
        output_result = output_result.replace(2.0, "\033[93m-\033[0m")
        output_result = output_result.replace(1.0, "\033[91mX\033[0m")
        output_result = output_result.replace(0.0, "\033[94m\u2713\033[0m")
        # convert to dataframe (to use 'tabulate' library)
        df = DataFrame(output_result.to_records())
        from tabulate import tabulate

        output_result = tabulate(
            df, headers="keys", tablefmt="psql", showindex=False, stralign="center"
        )
        utils.logger.debug(
            "Status map summary for " + plot_info["parameter"] + ":\n%s", output_result
        )

    # --------------------------------------------------------------------------------------------------------------------------
    # create the figure
    fig = plt.figure(num=None, figsize=(8, 12), dpi=80, facecolor="w", edgecolor="k")
    sns.set(font_scale=1)

    # create labels for dets
    labels = new_dataframe.pivot(index="position", columns="location", values="name")

    # labels definition (AFTER having included OFF detectors too)
    # LOCATION:
    x_axis_labels = [f"S{no}" for no in sorted(new_dataframe["location"].unique())]
    # POSITION:
    y_axis_labels = [
        no
        for no in range(
            min(new_dataframe["position"].unique()),
            max(new_dataframe["position"].unique() + 1),
        )
    ]

    # to account for empty strings: ...not a good idea actually...
    # In L60, there are S1,S2,S7,S8: do we really want to display 4 empty strings, i.e. S3-S6? There is no need!
    # x_axis_labels = [f"S{no}" for no in range(min(new_dataframe["location"].unique()), max(new_dataframe["location"].unique()+1))]

    # status map (0=OK, 1=X, 2=AC, 3=OFF)
    custom_cmap = ["#318CE7", "#CC0000", "#F7AB60", "#D0D0D0", "#FFFFFF"]

    # create the heatmap
    status_map = sns.heatmap(
        data=result,
        annot=labels,
        annot_kws={"size": 6},
        vmin=0,
        vmax=len(custom_cmap),
        yticklabels=y_axis_labels,
        xticklabels=x_axis_labels,
        fmt="s",
        cmap=custom_cmap,
        cbar=True,
        cbar_kws={"shrink": 0.5},
        linewidths=1,
        linecolor="white",
        square=True,
        rasterized=True,
    )

    # set user defined heatmap colours
    colorbar = status_map.collections[0].colorbar
    colorbar.set_ticks([1 / 2, 3 / 2, 5 / 2, 7 / 2])
    colorbar.set_ticklabels(["OK", "X", "AC", "OFF"])

    plt.tick_params(
        axis="both",
        which="major",
        # labelsize=20,
        labelbottom=False,
        bottom=False,
        top=False,
        labeltop=True,
    )
    plt.yticks(rotation=0)
    plt.title(plot_title)

    # saving
    plotting.save_pdf(plt, pdf)

    # returning the figure
    return fig


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# EXPOSURE FUNCTION
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def exposure_plot(subsystem, data_analysis: DataFrame, plot_info: dict, pdf: PdfPages):
    if plot_info["subsystem"] == "spms":
        utils.logger.error(
            "\033[91mPlotting the summary is not available for the spms.\nTry again!\033[0m"
        )
        raise errors.MonitoringError("exposure_plot failed (see log for details)")

    # cbar unit (either 'kg d', if exposure is less than 0.1 kg yr, or 'kg yr'); note: exposure, at this point, is evaluated as 'kg yr'
    if data_analysis["exposure"].max() < 0.1:
        cbar_unit = "kg d"
    else:
        cbar_unit = "kg yr"

    # convert exposure into [kg day] if data_analysis["exposure"].max() < 0.1 kg yr
    if cbar_unit == "kg d":
        data_analysis["exposure"] = data_analysis["exposure"] * 365.25
    # data_analysis.loc[data_analysis["exposure"] < 0.1, "exposure"] = data_analysis.loc[data_analysis["exposure"] < 0.1, "exposure"] * 365.25
    # drop duplicate rows, based on channel entry (exposure is constant for a fixed channel)
    data_analysis = data_analysis.drop_duplicates(subset=["channel"])
    # total exposure
    tot_expo = data_analysis["exposure"].sum()
    utils.logger.info(f"Total exposure: {tot_expo:.3f} {cbar_unit}")

    data_analysis = data_analysis.filter(
        ["channel", "name", "location", "position", "exposure", "livetime_in_s"]
    )

    # -------------------------------------------------------------------------------
    # OFF detectors
    # -------------------------------------------------------------------------------

    # include OFF channels and see what is their status
    off_channels = subsystem.channel_map[subsystem.channel_map["status"] == "off"][
        "channel"
    ].unique()

    extra_rows = []
    if len(off_channels) != 0:
        for channel in off_channels:
            # check if the channel is already in the exposure dataframe; if not, add a new row for it
            if channel not in data_analysis["channel"].values:
                status_info = subsystem.channel_map[
                    subsystem.channel_map["channel"] == channel
                ]["status"].iloc[0]

                # get status info
                if status_info != "on":
                    exposure = 0.0
                    livetime_in_s = 0.0

                # get position within the array + other necessary info
                name, location, position = get_info_from_channel(
                    subsystem.channel_map, channel
                )

                # collect new row for not-ON detectors; concatenated once below
                extra_rows.append(
                    [channel, name, location, position, exposure, livetime_in_s]
                )

    # -------------------------------------------------------------------------------
    # ON but NULL exposure detectors
    # -------------------------------------------------------------------------------
    on_channels = subsystem.channel_map[subsystem.channel_map["status"] == "on"][
        "channel"
    ].unique()

    for channel in on_channels:
        if channel in list(data_analysis["channel"].unique()):
            continue

        # if not there, set exposure to zero
        exposure = 0.0
        livetime_in_s = 0.0

        # get position within the array + other necessary info
        name, location, position = get_info_from_channel(subsystem.channel_map, channel)

        # collect new row; concatenated once below
        extra_rows.append([channel, name, location, position, exposure, livetime_in_s])

    if extra_rows:
        data_analysis = concat(
            [
                data_analysis,
                DataFrame(
                    extra_rows,
                    columns=[
                        "channel",
                        "name",
                        "location",
                        "position",
                        "exposure",
                        "livetime_in_s",
                    ],
                ),
            ],
            ignore_index=True,
            axis=0,
        )

    # values to plot
    result = data_analysis.pivot(
        index="position", columns="location", values="exposure"
    )
    result = result.round(3)

    # display it
    if utils.logger.getEffectiveLevel() is utils.logging.DEBUG:
        from tabulate import tabulate

        output_result = tabulate(
            result, headers="keys", tablefmt="psql", showindex=False, stralign="center"
        )
        utils.logger.debug(
            "Status map summary for " + plot_info["parameter"] + ":\n%s", output_result
        )

    # calculate total livetime as sum of content of livetime_in_s column (and then convert it a human readable format)
    tot_livetime = data_analysis["livetime_in_s"].unique()[0]
    tot_livetime, unit = utils.get_livetime(tot_livetime)

    # -------------------------------------------------------------------------------
    # plot
    # -------------------------------------------------------------------------------

    # create the figure
    fig = plt.figure(num=None, figsize=(8, 12), dpi=80, facecolor="w", edgecolor="k")
    sns.set(font_scale=1)

    # create labels for dets, with exposure values
    labels = result.astype(str)

    # labels definition (AFTER having included OFF detectors too) -------------------------------
    # LOCATION:
    x_axis_labels = [f"S{no}" for no in sorted(data_analysis["location"].unique())]
    # POSITION:
    y_axis_labels = [
        no
        for no in range(
            min(data_analysis["position"].unique()),
            max(data_analysis["position"].unique() + 1),
        )
    ]

    # create the heatmap
    status_map = sns.heatmap(
        data=result,
        annot=labels,
        annot_kws={"size": 6},
        yticklabels=y_axis_labels,
        xticklabels=x_axis_labels,
        fmt="s",
        cbar=True,
        cbar_kws={"shrink": 0.5},
        linewidths=1,
        linecolor="white",
        square=True,
        rasterized=True,
    )

    # add title "kg yr" as text on top of the cbar
    plt.text(
        1.08,
        0.89,
        f"({cbar_unit})",
        transform=status_map.transAxes,
        horizontalalignment="center",
        verticalalignment="center",
    )

    plt.tick_params(
        axis="both",
        which="major",
        labelbottom=False,
        bottom=False,
        top=False,
        labeltop=True,
    )
    plt.yticks(rotation=0)
    plt.title(
        f"{plot_info['subsystem']} - {plot_info['title']}\nTotal livetime: {tot_livetime:.2f}{unit}\nTotal exposure: {tot_expo:.3f} {cbar_unit}"
    )

    # saving
    plotting.save_pdf(plt, pdf)

    return fig


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Plotting recurring functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def get_info_from_channel(channel_map: DataFrame, channel: int):
    """Get info (name, location, position) from a channel number, once the channel map is provided as a DataFrame."""
    name = channel_map.loc[channel_map["channel"] == channel]["name"].iloc[0]
    location = channel_map.loc[channel_map["channel"] == channel]["location"].iloc[0]
    position = channel_map.loc[channel_map["channel"] == channel]["position"].iloc[0]

    return name, location, position
