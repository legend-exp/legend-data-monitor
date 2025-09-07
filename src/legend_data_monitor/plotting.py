import io
from typing import Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from pandas import DataFrame
from seaborn import color_palette

from . import (
    analysis_data,
    plot_styles,
    save_data,
    string_visualization,
    subsystem,
    utils,
)

# -------------------------------------------------------------------------

# global variable to be filled later with colors based on number of channels
COLORS = []

# -------------------------------------------------------------------------
# main plotting function(s)
# -------------------------------------------------------------------------

# plotting function that makes subsystem plots
# feel free to write your own one using Dataset, Subsystem and ParamData objects
# for example, this structure won't work to plot one parameter VS the other


def make_subsystem_plots(
    subsystem: subsystem.Subsystem,
    plots: dict,
    dataset_info: dict,
    plt_path: str,
    saving=None,
):

    plt_file = utils.get_output_plot_path(plt_path, "pdf")
    pdf = PdfPages(plt_file)
    is_pdf_saved = False

    for plot_title in plots:
        if "plot_structure" not in plots[plot_title].keys():
            utils.logger.info(f"'{plot_title}' can't be plotted. Skip it.")
            continue

        banner = "\33[95m" + "~" * 50 + "\33[0m"
        utils.logger.info(banner)
        utils.logger.info(f"\33[95m P L O T T I N G : {plot_title}\33[0m")
        utils.logger.info(banner)

        # -------------------------------------------------------------------------
        # settings checks
        # -------------------------------------------------------------------------

        # --- original plot settings provided in json
        plot_settings = plots[plot_title]

        # --- defaults
        # default time window None if not parameter event rate will be accounted for in AnalysisData,
        # here need to account for plot style vs time (None for all others)
        if "time_window" not in plot_settings:
            plot_settings["time_window"] = None
        # same, here need to account for unit label %
        if "variation" not in plot_settings:
            plot_settings["variation"] = False
        # range for parameter
        if "range" not in plot_settings:
            plot_settings["range"] = [None, None]
        # resampling: applies only to vs time plot
        if "resampled" not in plot_settings:
            plot_settings["resampled"] = None
        # status plot requires no plot style option (for now)
        if "plot_style" not in plot_settings:
            plot_settings["plot_style"] = None
        if plot_settings["plot_style"] != "par vs par" and (
            isinstance(plot_settings["parameters"], list)
            and len(plot_settings["parameters"]) > 1
        ):
            utils.logger.warning(
                "\033[93m'%s' is not enabled for multiple parameters. "
                + "We switch to the 'par vs par' option.\033[0m",
                plot_settings["plot_style"],
            )
            plot_settings["plot_style"] = "par vs par"

        # --- additional not in json
        # add saving info + plot where we save things
        plot_settings["saving"] = saving
        plot_settings["plt_path"] = plt_path

        # --- checks
        # resampled not provided for vs time -> set default
        if plot_settings["plot_style"] == "vs time":
            if not plot_settings["resampled"]:
                plot_settings["resampled"] = "also"
                utils.logger.warning(
                    "\033[93mNo 'resampled' option was specified. Both resampled and all entries will be plotted (otherwise you can try again using the option 'no', 'only', 'also').\033[0m"
                )
        # resampled provided for irrelevant plot
        elif plot_settings["resampled"]:
            utils.logger.warning(
                "\033[93mYou're using the option 'resampled' for a plot style that does not need it. For this reason, that option will be ignored.\033[0m"
            )

        # -------------------------------------------------------------------------
        # set up analysis data
        # -------------------------------------------------------------------------

        # --- AnalysisData:
        # - select parameter(s) of interest
        # - subselect type of events (pulser/phy/all/klines)
        # - apply cuts
        # - calculate special parameters if present
        # - get channel mean
        # - calculate variation from mean, if asked
        # note: subsystem.data contains: absolute value of a param, the respective value for aux channel (with ratio and diff already computed)
        data_analysis = analysis_data.AnalysisData(
            subsystem.data, selection=plot_settings | dataset_info
        )
        # check if the dataframe is empty; if so, skip this parameter
        if utils.check_empty_df(data_analysis):
            continue
        utils.logger.debug(data_analysis.data)

        # get list of parameters
        params = plot_settings["parameters"]
        if isinstance(params, str):
            params = [params]

        # this is ok for geds, but for spms? maybe another function will be necessary for this?
        # note: this will not do anything in case the parameter is from hit tier
        aux_analysis, aux_ratio_analysis, aux_diff_analysis = analysis_data.get_aux_df(
            subsystem.data.copy(), params, plot_settings | dataset_info, "pulser01ana"
        )

        # -------------------------------------------------------------------------
        # switch to aux data (if specified in config file)
        # -------------------------------------------------------------------------
        # check if the aux objects are not empty
        # !!! not handled for spms
        if not utils.check_empty_df(aux_ratio_analysis) and not utils.check_empty_df(
            aux_diff_analysis
        ):
            if (
                "AUX_ratio" in plot_settings.keys()
                and plot_settings["AUX_ratio"] is True
            ):
                data_to_plot = aux_ratio_analysis
            if "AUX_diff" in plot_settings.keys() and plot_settings["AUX_diff"] is True:
                data_to_plot = aux_diff_analysis
            if (
                ("AUX_ratio" not in plot_settings and "AUX_diff" not in plot_settings)
                or (plot_settings.get("AUX_ratio") is False)
                or (plot_settings.get("AUX_diff") is False)
            ):
                data_to_plot = data_analysis
        else:
            data_to_plot = data_analysis

        # -------------------------------------------------------------------------
        # set up plot info
        # -------------------------------------------------------------------------

        # -------------------------------------------------------------------------
        # color settings using a pre-defined palette

        # num colors needed = max number of channels per string
        # - find number of unique positions in each string
        # - get maximum occurring
        plot_structure = (
            PLOT_STRUCTURE[plot_settings["plot_structure"]]
            if "plot_structure" in plot_settings
            else None
        )

        if plot_structure == "per cc4":
            if (
                data_to_plot.data.iloc[0]["cc4_id"] is None
                or data_to_plot.data.iloc[0]["cc4_channel"] is None
            ):
                if subsystem.type in ["spms", "pulser", "pulser01ana", "bsln"]:
                    utils.logger.error(
                        "\033[91mPlotting per CC4 is not available for %s. Try again!\033[0m",
                        subsystem.type,
                    )
                    exit()
                else:
                    utils.logger.error(
                        "\033[91mPlotting per CC4 is not available because CC4 ID or/and CC4 channel are 'None'.\nTry again!\033[0m"
                    )
                    exit()
            # ...if cc4 are present, group by them
            max_ch_per_string = (
                data_to_plot.data.groupby("cc4_id")["cc4_channel"].nunique().max()
            )
        else:
            max_ch_per_string = (
                data_to_plot.data.groupby("location")["position"].nunique().max()
            )
        global COLORS
        COLORS = color_palette("hls", max_ch_per_string).as_hex()

        # -------------------------------------------------------------------------
        # basic information needed for plot structure
        plot_info = {
            "title": plot_title,
            "subsystem": subsystem.type,
            "locname": {
                "geds": "string",
                "spms": "fiber",
                "pulser": "puls",
                "pulser01ana": "pulser01ana",
                "FCbsln": "FC bsln",
                "muon": "muon",
            }[subsystem.type],
        }

        # parameters from plot settings to be simply propagated
        plot_info["plot_style"] = plot_settings["plot_style"]
        plot_info["time_window"] = plot_settings["time_window"]
        plot_info["resampled"] = plot_settings["resampled"]
        plot_info["range"] = plot_settings["range"]

        # information for shifting the channels or not (not needed only for the 'per channel' structure option) when plotting the std
        plot_info["std"] = True if plot_structure == "per channel" else False

        # -------------------------------------------------------------------------
        # information needed for plot style depending on parameters

        # first, treat it like multiple parameters, add dictionary to each entry with values for each parameter
        multi_param_info = ["unit", "label", "unit_label", "limits", "event_type"]
        for info in multi_param_info:
            plot_info[info] = {}

        # name(s) of parameter(s) to plot - always list
        plot_info["parameters"] = params
        # preserve original param_mean before potentially adding _var to name
        plot_info["param_mean"] = [x + "_mean" for x in params]
        # add _var if variation asked
        if plot_settings["variation"]:
            plot_info["parameters"] = [x + "_var" for x in params]

        for param in plot_info["parameters"]:
            # plot info should contain final parameter to plot i.e. _var if var is asked
            # unit, label and limits are connected to original parameter name
            # this is messy AF need to rethink
            param_orig = param.rstrip("_var")
            plot_info["unit"][param] = utils.PLOT_INFO[param_orig]["unit"]
            plot_info["label"][param] = utils.PLOT_INFO[param_orig]["label"]

            # modify the labels in case we perform a ratio/diff with aux channel data
            if param_orig in utils.PARAMETER_TIERS.keys():
                if (
                    "AUX_ratio" in plot_settings.keys()
                    and utils.PARAMETER_TIERS[param_orig] != "hit"
                ):
                    if plot_settings["AUX_ratio"] is True:
                        plot_info["label"][param] += (
                            " / " + plot_info["label"][param] + "(PULS01ANA)"
                        )
                if (
                    "AUX_diff" in plot_settings.keys()
                    and utils.PARAMETER_TIERS[param_orig] != "hit"
                ):
                    if plot_settings["AUX_diff"] is True:
                        plot_info["label"][param] += (
                            " - " + plot_info["label"][param] + "(PULS01ANA)"
                        )

            keyword = "variation" if plot_settings["variation"] else "absolute"
            plot_info["limits"][param] = (
                utils.PLOT_INFO[param_orig]["limits"][subsystem.type][keyword]
                if subsystem.type in utils.PLOT_INFO[param_orig]["limits"].keys()
                else [None, None]
            )
            # unit label should be % if variation was asked
            plot_info["unit_label"][param] = (
                "%" if plot_settings["variation"] else plot_info["unit"][param_orig]
            )
            plot_info["event_type"][param] = plot_settings["event_type"]

        if len(params) == 1:
            # change "parameters" to "parameter" - for single-param plotting functions
            plot_info["parameter"] = plot_info["parameters"][0]
            # now, if it was actually a single parameter, convert {param: value} dict structure to just the value
            # this is how one-parameter plotting functions are designed
            for info in multi_param_info:
                plot_info[info] = plot_info[info][plot_info["parameter"]]
            # same for mean
            plot_info["param_mean"] = plot_info["param_mean"][0]

            # threshold values are needed for status map; might be needed for plotting limits on canvas too
            # only needed for single param plots (for now)
            if subsystem.type not in ["pulser", "pulser01ana", "FCbsln", "muon"]:
                keyword = "variation" if plot_settings["variation"] else "absolute"
                plot_info["limits"] = utils.PLOT_INFO[params[0]]["limits"][
                    subsystem.type
                ][keyword]

            # needed for grey lines for K lines, in case we are looking at energy itself (not event rate for example)
            plot_info["event_type"] = plot_settings["event_type"]

        # -------------------------------------------------------------------------
        # call chosen plot structure + plotting
        # -------------------------------------------------------------------------

        if "exposure" in plot_info["parameters"]:
            string_visualization.exposure_plot(
                subsystem, data_to_plot.data, plot_info, pdf
            )
        else:
            utils.logger.debug("Plot structure: %s", plot_settings["plot_structure"])
            plot_structure(data_to_plot.data, plot_info, pdf)

        # For some reason, after some plotting functions the index is set to "channel".
        # We need to set it back otherwise string_visualization.py gets crazy.
        data_to_plot.data = data_to_plot.data.reset_index()

        # -------------------------------------------------------------------------
        # call status plot
        # -------------------------------------------------------------------------

        if "status" in plot_settings and plot_settings["status"]:
            if subsystem.type in ["pulser", "pulser01ana", "FCbsln", "muon"]:
                utils.logger.debug(
                    f"Thresholds are not enabled for {subsystem.type}! Use you own eyes to do checks there"
                )
            else:
                # take care of one parameter and multiple parameters cases
                for param in params:
                    if len(params) == 1:
                        _ = string_visualization.status_plot(
                            subsystem, data_analysis.data, plot_info, pdf
                        )
                    if len(params) > 1:
                        # retrieved the necessary info for the specific parameter under study (just in the multi-parameters case)
                        plot_info_param = save_data.get_param_info(param, plot_info)
                        _ = string_visualization.status_plot(
                            subsystem, data_analysis.data, plot_info_param, pdf
                        )

        # -------------------------------------------------------------------------
        # save results (hdf format)
        # -------------------------------------------------------------------------

        save_data.save_hdf(
            saving,
            plt_path + f"-{subsystem.type}.hdf",
            data_analysis,
            "pulser01ana",
            aux_analysis,
            aux_ratio_analysis,
            aux_diff_analysis,
            plot_info,
        )

        is_pdf_saved = True

    pdf.close()
    if is_pdf_saved:
        utils.logger.info(
            f"All plots saved in: \33[4m{plt_path}-{subsystem.type}.pdf\33[0m"
        )


# -------------------------------------------------------------------------------
# different plot structure functions, defining figures and subplot layouts
# -------------------------------------------------------------------------------

# See mapping user plot structure keywords to corresponding functions in the end of this file


def plot_per_ch(data_analysis: DataFrame, plot_info: dict, pdf: PdfPages):
    # --- choose plot function based on user requested style e.g. vs time or histogram
    plot_style = plot_styles.PLOT_STYLE[plot_info["plot_style"]]
    utils.logger.debug("Plot style: " + plot_info["plot_style"])

    data_analysis = data_analysis.sort_values(["location", "position"])

    # -------------------------------------------------------------------------------

    # separate figure for each string/fiber ("location")
    for location, data_location in data_analysis.groupby("location"):
        utils.logger.debug(f"... {plot_info['locname']} {location}")

        # -------------------------------------------------------------------------------
        # create plot structure: 1 column, N rows with subplot for each channel
        # -------------------------------------------------------------------------------

        # number of channels in this string/fiber
        numch = len(data_location["channel"].unique())
        # create corresponding number of subplots for each channel, set constrained layout to accommodate figure suptitle
        fig, axes = plt.subplots(
            nrows=numch,
            ncols=1,
            figsize=(10, numch * 3),
            sharex=True,
            constrained_layout=True,
        )  # , sharey=True)
        # in case of pulser, axes will be not a list but one axis -> convert to list
        axes = [axes] if numch == 1 else axes

        # -------------------------------------------------------------------------------
        # plot
        # -------------------------------------------------------------------------------

        ax_idx = 0
        # plot one channel on each axis, ordered by position
        for position, data_channel in data_location.groupby("position"):
            utils.logger.debug(f"...... position {position}")
            # define what colors are needed
            # if this function is not called by makes_subsystem_plot() need to define colors locally
            # to be included in a separate function to be called every time (maybe in utils?)
            max_ch_per_string = (
                data_analysis.groupby("location")["position"].nunique().max()
            )
            global COLORS
            COLORS = color_palette("hls", max_ch_per_string).as_hex()

            # plot selected style on this axis
            plot_style(data_channel, fig, axes[ax_idx], plot_info, color=COLORS[ax_idx])

            # --- add summary to axis - only for single channel plots
            # name, position and mean are unique for each channel - take first value
            df_text = data_channel.iloc[0][["channel", "position", "name"]]
            text = df_text["name"] + "\n" + f"channel {df_text['channel']}\n"
            text += (
                f"position {df_text['position']}"
                if plot_info["subsystem"]
                not in ["pulser", "pulser01ana", "FCbsln", "muon"]
                else ""
            )
            if len(plot_info["parameters"]) == 1:
                # in case of 1 parameter, "param mean" entry is a single string param_mean
                # in case of > 1, it's a list of parameters -> ignore for now and plot mean only for 1 param case
                par_mean = data_channel.iloc[0][
                    plot_info["param_mean"]
                ]  # single number
                if plot_info["parameter"] != "event_rate":
                    fwhm_ch = (
                        0  # get_fwhm_for_fixed_ch(data_channel, plot_info["parameter"])
                    )
                    text += f"\nFWHM {fwhm_ch}" if fwhm_ch != 0 else ""

                text += "\n" + (
                    f"mean {round(par_mean, 3)} [{plot_info['unit']}]"
                    if par_mean is not None
                    else ""
                )  # handle with care mean='None' situations
            axes[ax_idx].text(1.01, 0.5, text, transform=axes[ax_idx].transAxes)

            # add grid
            axes[ax_idx].grid("major", linestyle="--")
            axes[ax_idx].set_axisbelow(True)
            # remove automatic y label since there will be a shared one
            axes[ax_idx].set_ylabel("")

            # plot limits
            # check if "limits" present, is not for pulser (otherwise crash when plotting e.g. event rate)
            if "limits" in plot_info:
                plot_limits(axes[ax_idx], plot_info["parameters"], plot_info["limits"])

            ax_idx += 1

        # -------------------------------------------------------------------------------
        if plot_info["subsystem"] in ["pulser", "pulser01ana", "FCbsln", "muon"]:
            y_title = 1.05
            axes[0].set_title("")
        else:
            y_title = 1.01
            axes[0].set_title(f"{plot_info['locname']} {location}")
        fig.suptitle(f"{plot_info['subsystem']} - {plot_info['title']}", y=y_title)

        save_pdf(plt, pdf)

    return fig


def plot_per_cc4(data_analysis: DataFrame, plot_info: dict, pdf: PdfPages):
    if plot_info["subsystem"] in ["pulser", "pulser01ana", "FCbsln", "muon"]:
        utils.logger.error(
            "\033[91mPlotting per CC4 is not available for %s channel.\nTry again with a different plot structure!\033[0m",
            plot_info["subsystem"],
        )
        exit()
    # --- choose plot function based on user requested style e.g. vs time or histogram
    plot_style = plot_styles.PLOT_STYLE[plot_info["plot_style"]]
    utils.logger.debug("Plot style: " + plot_info["plot_style"])

    # --- create plot structure
    # number of cc4s
    no_cc4_id = len(data_analysis["cc4_id"].unique())
    # set constrained layout to accommodate figure suptitle
    fig, axes = plt.subplots(
        no_cc4_id,
        figsize=(10, no_cc4_id * 3),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    # -------------------------------------------------------------------------------
    # create label of format hardcoded for geds sXX-pX-chXXX-name-CC4channel
    # -------------------------------------------------------------------------------
    labels = data_analysis.groupby("channel").first()[
        ["name", "position", "location", "cc4_channel", "cc4_id"]
    ]
    labels["channel"] = labels.index
    labels["label"] = labels[["location", "position", "name", "cc4_channel"]].apply(
        lambda x: f"s{x[0]}-p{x[1]}-{x[2]}-cc4 ch.{x[3]}", axis=1
    )
    # put it in the table
    data_analysis = data_analysis.set_index("channel")
    data_analysis["label"] = labels["label"]

    # -------------------------------------------------------------------------------
    # plot
    # -------------------------------------------------------------------------------

    data_analysis = data_analysis.sort_values(["cc4_id", "cc4_channel", "label"])
    # new subplot for each string
    ax_idx = 0
    for cc4_id, data_cc4_id in data_analysis.groupby("cc4_id"):
        utils.logger.debug(f"... CC4 {cc4_id}")
        # set colors
        max_ch_per_cc4 = data_analysis.groupby("cc4_id")["cc4_channel"].nunique().max()
        global COLORS
        COLORS = color_palette("hls", max_ch_per_cc4).as_hex()

        # new color for each channel
        col_idx = 0
        labels = []
        for label, data_channel in data_cc4_id.groupby("label"):
            cc4_channel = (label.split("-"))[-1]
            utils.logger.debug(f"...... {cc4_channel}")
            plot_style(data_channel, fig, axes[ax_idx], plot_info, COLORS[col_idx])

            labels.append(label)
            if len(plot_info["parameters"]) == 1:
                if plot_info["parameter"] != "event_rate":
                    fwhm_ch = (
                        0  # get_fwhm_for_fixed_ch(data_channel, plot_info["parameter"])
                    )
                    labels[-1] = (
                        label + f" - FWHM: {fwhm_ch}" if fwhm_ch != 0 else label
                    )
                else:
                    labels[-1] = label
            col_idx += 1

        # add grid
        axes[ax_idx].grid("major", linestyle="--")
        axes[ax_idx].set_axisbelow(True)
        # beautification
        axes[ax_idx].set_title(f"CC4 {cc4_id}")
        axes[ax_idx].set_ylabel("")
        axes[ax_idx].legend(labels=labels, loc="center left", bbox_to_anchor=(1, 0.5))

        # plot limits
        # check if "limits" present, is not for pulser (otherwise crash when plotting e.g. event rate)
        if "limits" in plot_info:
            plot_limits(axes[ax_idx], plot_info["parameters"], plot_info["limits"])

        ax_idx += 1

    # -------------------------------------------------------------------------------
    y_title = (
        1.05
        if plot_info["subsystem"] in ["pulser", "pulser01ana", "FCbsln", "muon"]
        else 1.01
    )
    fig.suptitle(f"{plot_info['subsystem']} - {plot_info['title']}", y=y_title)
    save_pdf(plt, pdf)

    return fig


def plot_per_string(data_analysis: DataFrame, plot_info: dict, pdf: PdfPages):
    # --- choose plot function based on user requested style e.g. vs time or histogram
    plot_style = plot_styles.PLOT_STYLE[plot_info["plot_style"]]
    utils.logger.debug("Plot style: " + plot_info["plot_style"])

    # --- create plot structure
    # number of strings/fibers
    no_location = len(data_analysis["location"].unique())
    # set constrained layout to accommodate figure suptitle
    fig, axes = plt.subplots(
        no_location,
        figsize=(10, no_location * 3),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    # in case of pulser, axes will be not a list but one axis -> convert to list
    axes = [axes] if no_location == 1 else axes

    # -------------------------------------------------------------------------------
    # create label of format hardcoded for geds pX-chXXX-name
    # -------------------------------------------------------------------------------

    labels = data_analysis.groupby("channel").first()[["name", "position"]]
    labels["channel"] = labels.index
    labels["label"] = labels[["position", "channel", "name"]].apply(
        lambda x: f"p{x[0]}-ch{str(x[1]).zfill(3)}-{x[2]}", axis=1
    )
    # put it in the table
    data_analysis = data_analysis.set_index("channel")
    data_analysis["label"] = labels["label"]
    data_analysis = data_analysis.sort_values("label")

    # -------------------------------------------------------------------------------
    # plot
    # -------------------------------------------------------------------------------

    data_analysis = data_analysis.sort_values(["location", "label"])
    map_dict = utils.get_map_dict(data_analysis)

    # new subplot for each string
    ax_idx = 0
    for location, data_location in data_analysis.groupby("location"):
        # define what colors are needed
        # if this function is not called by makes_subsystem_plot() need to define colors
        # to be included in a separate function to be called every time (maybe in utils?)
        max_ch_per_string = (
            data_analysis.groupby("location")["position"].nunique().max()
        )
        global COLORS
        COLORS = color_palette("hls", max_ch_per_string).as_hex()

        utils.logger.debug(f"... {plot_info['locname']} {location}")

        # new color for each channel
        col_idx = 0
        labels = []
        for label, data_channel in data_location.groupby("label"):
            plot_style(
                data_channel,
                fig,
                axes[ax_idx],
                plot_info,
                COLORS[col_idx],
                map_dict=map_dict,
            )
            labels.append(label)
            if len(plot_info["parameters"]) == 1:
                if plot_info["parameter"] != "event_rate":
                    fwhm_ch = (
                        0  # get_fwhm_for_fixed_ch(data_channel, plot_info["parameter"])
                    )
                    labels[-1] = (
                        label + f" - FWHM: {fwhm_ch}" if fwhm_ch != 0 else label
                    )
                else:
                    labels[-1] = label
            col_idx += 1

        # add grid
        axes[ax_idx].grid("major", linestyle="--")
        axes[ax_idx].set_axisbelow(True)
        # beautification
        axes[ax_idx].set_title(f"{plot_info['locname']} {location}")
        axes[ax_idx].set_ylabel("")
        axes[ax_idx].legend(labels=labels, loc="center left", bbox_to_anchor=(1, 0.5))

        # plot limits
        # check if "limits" present, is not for pulser (otherwise crash when plotting e.g. event rate)
        if "limits" in plot_info:
            plot_limits(axes[ax_idx], plot_info["parameters"], plot_info["limits"])

        ax_idx += 1

    # -------------------------------------------------------------------------------
    y_title = (
        1.05
        if plot_info["subsystem"] in ["pulser", "pulser01ana", "FCbsln", "muon"]
        else 1.01
    )
    fig.suptitle(f"{plot_info['subsystem']} - {plot_info['title']}", y=y_title)

    save_pdf(plt, pdf)

    return fig


def plot_array(data_analysis: DataFrame, plot_info: dict, pdf: PdfPages):
    if plot_info["subsystem"] == "spms":
        utils.logger.error(
            "\033[91mPlotting per array is not available for the spms.\nTry again!\033[0m"
        )
        exit()

    # --- choose plot function based on user requested style
    plot_style = plot_styles.PLOT_STYLE[plot_info["plot_style"]]
    utils.logger.debug("Plot style: " + plot_info["plot_style"])

    # --- create plot structure
    fig, axes = plt.subplots(
        1,  # no of location
        figsize=(10, 3),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    # -------------------------------------------------------------------------------
    # create label of format hardcoded for geds sX-pX-chXXX-name
    # -------------------------------------------------------------------------------
    labels = data_analysis.groupby("channel").first()[["name", "location", "position"]]
    labels["channel"] = labels.index
    labels["label"] = labels[["location", "position", "channel", "name"]].apply(
        lambda x: f"p{x[1]}-ch{str(x[2])}-{x[3]}", axis=1
    )
    # put it in the table
    data_analysis = data_analysis.set_index("channel")
    data_analysis["label"] = labels["label"]
    data_analysis = data_analysis.sort_values("label")

    data_analysis = data_analysis.sort_values(["location", "label"])
    map_dict = utils.get_map_dict(data_analysis)

    # -------------------------------------------------------------------------------
    # plot
    # -------------------------------------------------------------------------------
    data_analysis = data_analysis.sort_values(["location", "label"])

    # one color for each string
    col_idx = 0
    # some lists to fill with info, string by string
    labels = []
    channels = []
    legend = []

    # group by string
    for location, data_location in data_analysis.groupby("location"):
        utils.logger.debug(f"... {plot_info['locname']} {location}")

        max_ch_per_string = (
            data_analysis.groupby("location")["position"].nunique().max()
        )
        global COLORS
        COLORS = color_palette("hls", max_ch_per_string).as_hex()

        values_per_string = []  # y values - in each string
        channels_per_string = []  # x values - in each string
        # group by channel
        for label, data_channel in data_location.groupby("label"):
            plot_style(data_channel, fig, axes, plot_info, COLORS[col_idx])

            location = data_channel["location"].unique()[0]
            position = data_channel["position"].unique()[0]

            labels.append(label.split("-")[-1])
            channels.append(map_dict[str(location)][str(position)])
            if len(plot_info["parameters"]) == 1:
                values_per_string.append(
                    data_channel[plot_info["parameter"]].unique()[0]
                )
                channels_per_string.append(map_dict[str(location)][str(position)])

        if len(plot_info["parameters"]) == 1:
            # get average of plotted parameter per string (print horizontal line)
            avg_of_string = sum(values_per_string) / len(values_per_string)
            axes.hlines(
                y=avg_of_string,
                xmin=min(channels_per_string),
                xmax=max(channels_per_string),
                color="k",
                linestyle="-",
                linewidth=1,
            )
            utils.logger.debug(f"..... average: {round(avg_of_string, 2)}")

            # get legend entry (print string + colour)
            legend.append(
                mpatches.Patch(
                    color=COLORS[col_idx],
                    label=f"s{location} - avg: {round(avg_of_string, 2)} {plot_info['unit_label']}",
                )
            )

        # LAST thing to update
        col_idx += 1

    # -------------------------------------------------------------------------------
    # add legend
    axes.legend(
        loc=(1.04, 0.0),
        ncol=1,
        frameon=True,
        facecolor="white",
        framealpha=0,
        handles=legend,
    )
    # add grid
    axes.grid("major", linestyle="--")
    # set the grid behind the points
    axes.set_axisbelow(True)
    # beautification
    axes.set_ylabel("")
    axes.set_xlabel("")
    # add x labels
    axes.set_xticks(channels)
    axes.set_xticklabels(labels, fontsize=5)
    # rotate x labels
    plt.xticks(rotation=90, ha="center")
    # title/label
    fig.supxlabel("")
    fig.suptitle(f"{plot_info['subsystem']} - {plot_info['title']}", y=1.05)

    save_pdf(plt, pdf)

    return fig


# -------------------------------------------------------------------------------
# SiPM specific structures
# -------------------------------------------------------------------------------


def plot_per_fiber_and_barrel(data_analysis: DataFrame, plot_info: dict, pdf: PdfPages):
    if plot_info["subsystem"] != "spms":
        utils.logger.error(
            "\033[91mPlotting per fiber-barrel is available ONLY for spms.\nTry again!\033[0m"
        )
        exit()
    # here will be a function plotting SiPMs with:
    # - one figure for top and one for bottom SiPMs
    # - each figure has subplots with N columns and M rows where N is the number of fibers, and M is the number of positions (top/bottom -> 2)
    # this function will only work for SiPMs requiring a columns 'barrel' in the channel map
    # add a check in config settings check to make sure geds are not called with this structure to avoid crash


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# UNDER CONSTRUCTION!!!
def plot_per_barrel_and_position(
    data_analysis: DataFrame, plot_info: dict, pdf: PdfPages
):
    if plot_info["subsystem"] != "spms":
        utils.logger.error(
            "\033[91mPlotting per barrel-position is available ONLY for spms.\nTry again!\033[0m"
        )
        exit()
    # here will be a function plotting SiPMs with:
    # - one figure for each barrel-position combination (IB-top, IB-bottom, OB-top, OB-bottom) = 4 figures in total

    plot_style = plot_styles.PLOT_STYLE[plot_info["plot_style"]]
    utils.logger.debug("Plot style: " + plot_info["plot_style"])

    par_dict = {}

    # re-arrange dataframe to separate location: from location=[IB-015-016] to location=[IB] & fiber=[015-016]
    data_analysis["fiber"] = (
        data_analysis["location"].str.split("-").str[1].str.join("")
        + "-"
        + data_analysis["location"].str.split("-").str[2].str.join("")
    )
    data_analysis["location"] = (
        data_analysis["location"].str.split("-").str[0].str.join("")
    )

    # -------------------------------------------------------------------------------
    # create label of format hardcoded for geds pX-chXXX-name
    # -------------------------------------------------------------------------------

    labels = data_analysis.groupby("channel").first()[
        ["name", "position", "location", "fiber"]
    ]
    labels["channel"] = labels.index
    labels["label"] = labels[
        ["position", "location", "fiber", "channel", "name"]
    ].apply(lambda x: f"{x[0]}-{x[1]}-{x[2]}-ch{str(x[3]).zfill(3)}-{x[4]}", axis=1)
    # put it in the table
    data_analysis = data_analysis.set_index("channel")
    data_analysis["label"] = labels["label"]
    data_analysis = data_analysis.sort_values("label")

    data_analysis = data_analysis.sort_values(["location", "label"])

    # separate figure for each barrel ("location"= IB, OB)...
    for location, data_location in data_analysis.groupby("location"):
        utils.logger.debug(f"... {location} barrel")
        # ...and position ("position"= bottom, top)
        for position, data_position in data_location.groupby("position"):
            utils.logger.debug(f"..... {position}")

            # -------------------------------------------------------------------------------
            # create plot structure: M columns, N rows with subplots for each channel
            # -------------------------------------------------------------------------------

            # number of channels in this barrel
            if location == "IB":
                num_rows = 3
                num_cols = 3
            if location == "OB":
                num_rows = 4
                num_cols = 5
            # create corresponding number of subplots for each channel, set constrained layout to accommodate figure suptitle
            fig, axes = plt.subplots(
                nrows=num_rows,
                ncols=num_cols,
                figsize=(10, num_rows * 3),
                sharex=True,
                constrained_layout=True,
            )  # , sharey=True)

            # -------------------------------------------------------------------------------
            # plot
            # -------------------------------------------------------------------------------

            data_position = data_position.reset_index()
            channel = data_position["channel"].unique()
            det_idx = 0
            col_idx = 0
            labels = []
            for ax_row in axes:
                for (
                    axes
                ) in ax_row:  # this is already the Axes object (no need to add ax_idx)
                    # plot one channel on each axis, ordered by position
                    data_position = data_position[
                        data_position["channel"] == channel[col_idx]
                    ]  # get only rows for a given channel

                    # plotting...
                    if data_position.empty:
                        det_idx += 1
                        continue

                    plot_style(
                        data_position, fig, axes, plot_info, color=COLORS[det_idx]
                    )
                    labels.append(data_position["label"])

                    if channel[det_idx] not in par_dict.keys():
                        par_dict[channel[det_idx]] = {}

                    # set label as title for each axes
                    text = (
                        data_position["label"][0][4:]
                        if position == "top"
                        else data_position["label"][0][7:]
                    )
                    axes.set_title(label=text, loc="center")

                    # add grid
                    axes.grid("major", linestyle="--")
                    axes.set_axisbelow(True)
                    # remove automatic y label since there will be a shared one
                    axes.set_ylabel("")

                    det_idx += 1
                    col_idx += 1

            fig.suptitle(
                f"{plot_info['subsystem']} - {plot_info['title']}\n{position} {location}",
                y=1.15,
            )
            # fig.supylabel(f'{plotdata.param.label} [{plotdata.param.unit_label}]') # --> plot style
            plt.savefig(pdf, format="pdf", bbox_inches="tight")
            # figures are retained until explicitly closed; close to not consume too much memory
            plt.close()

            with io.BytesIO() as buf:
                fig.savefig(buf, bbox_inches="tight")
                buf.seek(0)
                par_dict[f"figure_plot_{location}_{position}"] = buf.getvalue()

    return par_dict


# -------------------------------------------------------------------------------
# plotting functions
# -------------------------------------------------------------------------------


def get_fwhm_for_fixed_ch(data_channel: DataFrame, parameter: str) -> float:
    """Calculate the FWHM of a given parameter for a given channel."""
    entries = data_channel[parameter]
    entries_avg = np.mean(entries)
    fwhm_ch = 2.355 * np.sqrt(np.mean(np.square(entries - entries_avg)))

    if fwhm_ch != 0:
        # Determine the number of decimal places based on the magnitude of the value
        decimal_places = max(0, int(-np.floor(np.log10(abs(fwhm_ch)))) + 2)
        # Format the FWHM value with the appropriate number of decimal places
        formatted_fwhm = "{:.{dp}f}".format(fwhm_ch, dp=decimal_places)
        # Remove trailing zeros from the formatted value
        formatted_fwhm = formatted_fwhm.rstrip("0").rstrip(".")

        return formatted_fwhm
    else:
        return 0


def plot_limits(ax: plt.Axes, params: list, limits: Union[list, dict]):
    """Plot limits (if present) on the plot. The multi-params case is carefully handled."""
    # one parameter case
    if (isinstance(params, list) and len(params) == 1) or isinstance(params, str):
        if not all([x is None for x in limits]):
            if limits[0] is not None:
                ax.axhline(y=limits[0], color="red", linestyle="--")
            if limits[1] is not None:
                ax.axhline(y=limits[1], color="red", linestyle="--")
    # multi-parameters case
    else:
        for idx, param in enumerate(params):
            limits_param = limits[param]
            if not all([x is None for x in limits_param]):
                if limits_param[0] is not None:
                    if idx == 0:
                        ax.axvline(x=limits_param[0], color="red", linestyle="--")
                    if idx == 1:
                        ax.axhline(y=limits_param[0], color="red", linestyle="--")
                if limits_param[1] is not None:
                    if idx == 0:
                        ax.axvline(x=limits_param[1], color="red", linestyle="--")
                    if idx == 1:
                        ax.axhline(y=limits_param[1], color="red", linestyle="--")


def save_pdf(plt, pdf: PdfPages):
    """Save the plot to a PDF file. The plot is closed after save_data."""
    if pdf:
        plt.savefig(pdf, format="pdf", bbox_inches="tight")
        plt.close()


# -------------------------------------------------------------------------------
# mapping user keywords to plot style functions
# -------------------------------------------------------------------------------

PLOT_STRUCTURE = {
    "per channel": plot_per_ch,
    "per cc4": plot_per_cc4,
    "per string": plot_per_string,
    "array": plot_array,
    "per fiber": plot_per_fiber_and_barrel,
    "per barrel": plot_per_barrel_and_position,
}
