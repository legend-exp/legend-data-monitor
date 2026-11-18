import glob
import itertools
import json
import math
import os
import pickle
import re
import shelve
import sys
from functools import partial

import awkward as ak
import h5py
import lh5
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import yaml
from dbetto import TextDB
from lgdo.lh5 import read_as
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker
from matplotlib.patches import Patch

from . import utils

# -------------------------------------------------------------------------

IPython_default = plt.rcParams.copy()
SMALL_SIZE = 8

plt.rc("font", size=SMALL_SIZE)
plt.rc("axes", titlesize=SMALL_SIZE)
plt.rc("axes", labelsize=SMALL_SIZE)
plt.rc("xtick", labelsize=SMALL_SIZE)
plt.rc("ytick", labelsize=SMALL_SIZE)
plt.rc("legend", fontsize=SMALL_SIZE)
plt.rc("figure", titlesize=SMALL_SIZE)
plt.rcParams["font.family"] = "serif"

matplotlib.rcParams["mathtext.fontset"] = "stix"

plt.rc("axes", facecolor="white", edgecolor="black", axisbelow=True, grid=True)

IGNORE_KEYS = utils.IGNORE_KEYS
CALIB_RUNS = utils.CALIB_RUNS


# -------------------------------------------------------------------------
def qc_distributions(
    auto_dir_path: str,
    phy_mtg_data: str,
    output_folder: str,
    start_key: str,
    period: str,
    run: str,
    last_cycle: str,
    det_info: dict,
    save_pdf: bool,
):
    pars_to_inspect = [
        "IsValidBlSlopeClassifier",
        "IsValidTailRmsClassifier",
        "IsValidPzSlopeClassifier",
        "IsValidBlSlopeRmsClassifier",
        "IsValidBlPolyRmsClassifier",
        "IsValidBlSlopeRmsClassifier",
        "IsValidCuspeminClassifier",
        "IsValidCuspemaxClassifier",
    ]

    my_file = os.path.join(
        output_folder, f"{period}/{run}/l200-{period}-{run}-phy-geds.hdf"
    )
    str_chns = det_info["str_chns"]
    utils.logger.debug("...inspecting QC classifiers")
    if not os.path.exists(my_file):
        utils.logger.warning(f"...file not found: {my_file}. Return!")
        return

    end_folder = os.path.join(
        output_folder,
        period,
        run,
        "mtg",
    )
    os.makedirs(end_folder, exist_ok=True)
    shelve_path = os.path.join(
        end_folder,
        f"l200-{period}-{run}-phy-monitoring",
    )

    step = 0.4
    with (
        shelve.open(shelve_path, "c", protocol=pickle.HIGHEST_PROTOCOL) as shelf,
        pd.HDFStore(my_file, "r") as store,
    ):
        df_energy_IsPhysics = store["/IsPhysics_TrapemaxCtcCal"]
        df_energy_IsPhysics = filter_series_by_ignore_keys(
            df_energy_IsPhysics, utils.IGNORE_KEYS, period
        )

        for par in pars_to_inspect:

            mask = df_energy_IsPhysics > 25
            df_All = utils.load_and_filter(store, f"/All_{par}")
            df_IsPulser = utils.load_and_filter(store, f"/IsPulser_{par}")
            df_IsBsln = utils.load_and_filter(store, f"/IsBsln_{par}")
            df_IsPhysics = utils.load_and_filter(store, f"/IsPhysics_{par}", mask=mask)

            if df_All.empty:
                continue

            df_All = filter_series_by_ignore_keys(df_All, utils.IGNORE_KEYS, period)
            if not df_IsPulser.empty:
                df_IsPulser = filter_series_by_ignore_keys(
                    df_IsPulser, utils.IGNORE_KEYS, period
                )
            if not df_IsBsln.empty:
                df_IsBsln = filter_series_by_ignore_keys(
                    df_IsBsln, utils.IGNORE_KEYS, period
                )
            if not df_IsPhysics.empty:
                df_IsPhysics = filter_series_by_ignore_keys(
                    df_IsPhysics, utils.IGNORE_KEYS, period
                )

            for string, det_list in str_chns.items():
                # grid size
                n_dets = len(det_list)
                ncols = math.ceil(math.sqrt(n_dets))
                nrows = math.ceil(n_dets / ncols)

                fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows))
                axes = axes.flatten()

                for i, det in enumerate(det_list):
                    if det not in det_info["detectors"]:
                        continue
                    if not det_info["detectors"][det]["processable"]:
                        continue

                    ax = axes[i]
                    ch = det_info["detectors"][det]["daq_rawid"]
                    # not processed detectors
                    if ch not in df_All.keys():
                        continue

                    vals_all = utils.get_vals(df_All, ch)
                    vals_pulser = utils.get_vals(df_IsPulser, ch)
                    vals_bsln = utils.get_vals(df_IsBsln, ch)
                    vals_phys = utils.get_vals(df_IsPhysics, ch)

                    vals_all = vals_all[~np.isnan(vals_all)]
                    vals_pulser = vals_pulser[~np.isnan(vals_pulser)]
                    vals_bsln = vals_bsln[~np.isnan(vals_bsln)]
                    vals_phys = vals_phys[~np.isnan(vals_phys)]

                    # global bins
                    bins = np.arange(-15, 15 + step, step)

                    # percentages
                    def safe_perc(vals, lo=-5, hi=5):
                        if len(vals) == 0:
                            return np.nan
                        return 100 * np.mean((vals >= lo) & (vals <= hi))

                    perc_all = safe_perc(vals_all)
                    perc_pulser = safe_perc(vals_pulser)
                    perc_bsln = safe_perc(vals_bsln)
                    perc_phys = safe_perc(vals_phys)

                    # plotting
                    ax.hist(
                        vals_all,
                        bins=bins,
                        label=f"All events - {perc_all:.1f}%",
                        histtype="step",
                        facecolor="g",
                    )
                    ax.hist(
                        vals_pulser,
                        bins=bins,
                        label=f"TP - {perc_pulser:.1f}%",
                        histtype="step",
                        facecolor="g",
                    )
                    ax.hist(
                        vals_bsln,
                        bins=bins,
                        label=f"FT - {perc_bsln:.1f}%",
                        histtype="step",
                        facecolor="g",
                    )
                    ax.hist(
                        vals_phys,
                        bins=bins,
                        label=f"~TP, ~FT, E>25 keV - {perc_phys:.1f}%",
                        histtype="step",
                        facecolor="g",
                    )

                    ax.axvline(-5, color="k", linestyle="--")
                    ax.axvline(5, color="k", linestyle="--")
                    ax.axvspan(-15, -5, color="darkgray", alpha=0.2)
                    ax.axvspan(5, 15, color="darkgray", alpha=0.2)
                    ax.set_ylabel("Counts")
                    ax.set_xlabel("Classifiers")
                    ax.legend(
                        title=f"{det} (pos {det_info['detectors'][det]['position']})",
                        loc="upper right",
                    )
                    ax.set_yscale("log")
                    ax.grid(False)
                    ax.set_xlim(-10, 10)

                # hide any unused subplots
                for j in range(i + 1, len(axes)):
                    axes[j].axis("off")

                fig.suptitle(
                    f"{period} {run} - string {string} - {par} - last cycle: {last_cycle}"
                )
                fig.tight_layout()

                if save_pdf:
                    pdf_folder = os.path.join(
                        output_folder, f"{period}/{run}/mtg/pdf", f"st{string}"
                    )
                    os.makedirs(pdf_folder, exist_ok=True)
                    plt.savefig(
                        os.path.join(
                            pdf_folder,
                            f"{period}_{run}_string{string}_{par}.pdf",
                        ),
                        bbox_inches="tight",
                    )

                # serialize+plot in a shelve object
                shelf[f"{period}_{run}_{par}"] = pickle.dumps(fig)
                plt.close()


def qc_ft_failure_rates(
    auto_dir_path: str,
    phy_mtg_data: str,
    output_folder: str,
    start_key: str,
    period: str,
    run: str,
    det_info: dict,
    save_pdf: bool,
):
    my_file = os.path.join(
        output_folder, f"{period}/{run}/l200-{period}-{run}-phy-geds.hdf"
    )
    str_chns = det_info["str_chns"]
    utils.logger.debug("...inspecting FT failure rates")
    if not os.path.exists(my_file):
        utils.logger.warning(f"...file not found: {my_file}. Return!")
        return

    end_folder = os.path.join(
        output_folder,
        period,
        run,
        "mtg",
    )
    os.makedirs(end_folder, exist_ok=True)
    shelve_path = os.path.join(
        end_folder,
        f"l200-{period}-{run}-phy-monitoring",
    )

    with (
        shelve.open(shelve_path, "c", protocol=pickle.HIGHEST_PROTOCOL) as shelf,
        pd.HDFStore(my_file, "r") as store,
    ):
        df = store["/IsBsln_IsBbLike"]
        df_DD = store["/IsBsln_IsDelayedDischarge"]
        df = filter_series_by_ignore_keys(df, utils.IGNORE_KEYS, period)
        df_DD = filter_series_by_ignore_keys(df_DD, utils.IGNORE_KEYS, period)
        df_clean = df[~df_DD]
        color_cycle = itertools.cycle(plt.cm.tab20.colors)

        for string, det_list in str_chns.items():
            fig, ax = plt.subplots(figsize=(12, 6))

            for det in det_list:
                if not det_info["detectors"][det]["processable"]:
                    continue

                ch = det_info["detectors"][det]["daq_rawid"]
                pos = det_info["detectors"][det]["position"]

                if ch not in df_clean.columns:
                    continue

                # Take channel data and resample to hourly counts
                data = df_clean[ch].copy()
                hourly_counts = data.resample("1h").sum()

                # convert to mHz: (counts / 3600 sec) * 1000
                hourly_rate = hourly_counts / 3600 * 1000

                color = next(color_cycle)
                hourly_rate.plot(
                    ax=ax,
                    drawstyle="steps-mid",
                    label=f"{det} - pos {pos}",
                    color=color,
                )

                ax.set_ylabel("FT failure rate [mHz]")
                ax.legend(ncol=2, fontsize="small", loc="upper left")
                ax.grid(False)

            fig.suptitle(f"{period} - {run} - string {string}")
            fig.tight_layout()

            if save_pdf:
                pdf_folder = os.path.join(
                    output_folder, f"{period}/{run}/mtg/pdf", f"st{string}"
                )
                os.makedirs(pdf_folder, exist_ok=True)
                plt.savefig(
                    os.path.join(
                        pdf_folder,
                        f"{period}_{run}_string{string}_FT_failure.pdf",
                    ),
                    bbox_inches="tight",
                )

            # serialize+plot in a shelve object
            shelf[f"{period}_{run}_FT_failure"] = pickle.dumps(fig)
            plt.close()


def mhz_to_percent(mhz, avg_total_forced_mhz):
    return (mhz / avg_total_forced_mhz) * 100


def percent_to_mhz(pct, avg_total_forced_mhz):
    return (pct / 100) * avg_total_forced_mhz


def qc_and_evt_summary_plots(
    auto_dir_path: str,
    phy_mtg_data: str,
    output_folder: str,
    start_key: str,
    period: str,
    run: str,
    last_cycle: str,
    det_info: dict,
    save_pdf: bool,
):
    utils.logger.debug("...inspecting FT failure rates")
    evt_files_phy = sorted(
        glob.glob(f"{auto_dir_path}/generated/tier/evt/phy/{period}/{run}/*.lh5")
    )

    if not evt_files_phy:
        evt_files_phy = sorted(
            glob.glob(f"{auto_dir_path}/generated/tier/pet/phy/{period}/{run}/*.lh5")
        )

    if not evt_files_phy:
        utils.logger.info("...no evt/pet data available, skip!")
        return

    # energies  = read_as("evt/geds", evt_files_phy, 'ak', field_mask=['energy'])
    ged_pul = read_as(
        "evt/coincident", evt_files_phy, "ak", field_mask=["geds", "puls"]
    )
    forced = read_as(
        "evt/trigger", evt_files_phy, "ak", field_mask=["is_forced", "timestamp"]
    )
    is_bb = read_as(
        "evt/geds/quality",
        evt_files_phy,
        "ak",
        field_mask=["is_bb_like", "is_good_channel"],
    )
    is_dis = read_as(
        "evt/geds/quality/is_not_bb_like",
        evt_files_phy,
        "ak",
        field_mask=["is_delayed_discharge"],
    )
    is_fail = read_as(
        "evt/geds/quality/is_not_bb_like",
        evt_files_phy,
        "ak",
        field_mask=["is_empty_bits", "rawid"],
    )

    # build dataframe for FT FAILING events
    mask = forced.is_forced & ~is_bb.is_bb_like & ~is_dis.is_delayed_discharge
    temp = is_fail.rawid[mask]
    y = {ch: np.zeros(len(forced.timestamp[mask])) for ch in set(ak.flatten(temp))}
    for i in range(len(temp)):
        if len(temp[i]) == 0:
            continue
        for ch in temp[i]:
            y[ch][i] += 1
    y["timestamp"] = ak.to_numpy(forced.timestamp[mask])

    df = pd.DataFrame(y)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df.set_index("timestamp", inplace=True)
    daily_cnt = df.resample("h").sum()

    # Folders
    end_folder = os.path.join(output_folder, period, run, "mtg")
    os.makedirs(end_folder, exist_ok=True)
    shelve_path = os.path.join(end_folder, f"l200-{period}-{run}-phy-monitoring")

    str_counts = {}
    color_cycle = itertools.cycle(plt.cm.tab20.colors)

    # --- all forced triggers (denominator across all strings)
    df_all = pd.DataFrame(
        {"timestamp": ak.to_numpy(forced.timestamp[forced.is_forced])}
    )
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], unit="s")
    df_all.set_index("timestamp", inplace=True)
    total_forced = df_all.resample("h").size()  # counts/hour, all strings
    avg_total_forced_mhz = (total_forced.mean() / 3600) * 1000
    on_mass = 0

    # ONE PERIOD, ALL RUNS
    with shelve.open(shelve_path, "c", protocol=pickle.HIGHEST_PROTOCOL) as shelf:
        # --- Per-string plots ---
        for string, det_list in det_info["str_chns"].items():
            fig, ax = plt.subplots(figsize=(12, 6))

            string_counts = None
            string_mass = 0

            for det in det_list:
                if not det_info["detectors"][det]["processable"]:
                    continue
                ch = det_info["detectors"][det]["daq_rawid"]
                if ch not in daily_cnt.columns:
                    continue

                mass = det_info["detectors"][det]["mass_in_kg"]
                if det_info["detectors"][det]["usability"] == "on":
                    on_mass += mass

                hourly_rate = daily_cnt[ch] / 3600 * 1000 / mass
                color = next(color_cycle)
                hourly_rate.plot(ax=ax, drawstyle="steps-mid", label=det, color=color)

                det_counts = daily_cnt[ch]

                string_counts = (
                    det_counts
                    if string_counts is None
                    else string_counts.add(det_counts, fill_value=0)
                )

                string_mass += mass

            if string_counts is not None and string_mass > 0:
                str_counts[string] = string_counts / 3600 * 1000 / string_mass
            else:
                str_counts[string] = None

            m2p = partial(mhz_to_percent, avg_total_forced_mhz=avg_total_forced_mhz)
            p2m = partial(percent_to_mhz, avg_total_forced_mhz=avg_total_forced_mhz)
            secax = ax.secondary_yaxis("right", functions=(m2p, p2m))
            secax.set_ylabel("FT failure fraction (%)")

            ax.set_ylabel("Normalized FT failure rate (mHz/kg)")
            ax.legend(
                ncol=2,
                fontsize="small",
                loc="upper left",
                title=f"Last cycle: {last_cycle}",
            )
            ax.grid(False)
            fig.suptitle(f"{period} - {run} - string {string}")
            fig.tight_layout()

            if save_pdf:
                pdf_folder = os.path.join(
                    output_folder, period, run, "mtg/pdf", f"st{string}"
                )
                os.makedirs(pdf_folder, exist_ok=True)
                plt.savefig(
                    os.path.join(
                        pdf_folder, f"{period}_{run}_string{string}_FT_failure.pdf"
                    ),
                    bbox_inches="tight",
                )

            shelf[f"{period}_{run}_string{string}_FT_failure"] = pickle.dumps(fig)
            plt.close(fig)

        # --- Combined plot of all strings ---
        fig, ax = plt.subplots(figsize=(12, 6))
        color_cycle = itertools.cycle(plt.cm.tab20.colors)
        for string, counts in str_counts.items():
            if counts is not None:
                color = next(color_cycle)
                counts.plot(
                    ax=ax, drawstyle="steps-mid", label=f"String {string}", color=color
                )

        ax.set_ylabel("Normalized FT failure rate (mHz/kg)")
        ax.set_title(f"{period} - {run} - All strings")
        ax.legend(
            ncol=2,
            fontsize="small",
            loc="upper left",
            title=f"Last cycle: {last_cycle}",
        )
        ax.grid(False)
        fig.tight_layout()

        if save_pdf:
            pdf_folder = os.path.join(output_folder, period, run, "mtg/pdf")
            os.makedirs(pdf_folder, exist_ok=True)
            plt.savefig(
                os.path.join(pdf_folder, f"{period}_{run}_all_strings_FT_failure.pdf"),
                bbox_inches="tight",
            )

        shelf[f"{period}_{run}_all_strings_FT_failure"] = pickle.dumps(fig)
        plt.close(fig)

        # --- FT survival fraction ---
        mask_forced = forced.is_forced
        mask_survived = mask_forced & is_bb.is_bb_like & ~is_dis.is_delayed_discharge
        ts_all = pd.to_datetime(forced.timestamp[mask_forced], unit="s")
        ts_survived = pd.to_datetime(forced.timestamp[mask_survived], unit="s")
        df_all = pd.DataFrame({"count": 1}, index=ts_all)
        df_survived = pd.DataFrame({"count": 1}, index=ts_survived)
        total_forced = df_all.resample("h").sum()["count"]
        surviving = df_survived.resample("h").sum()["count"]
        surviving_frac = surviving / total_forced * 100

        fig, ax = plt.subplots(figsize=(12, 6))
        surviving_frac.plot(ax=ax, drawstyle="steps-mid", color="red")
        ax.set_ylabel("FT surviving events (%)")
        ax.set_title(f"{period} - All strings combined")
        ax.grid(False)
        fig.tight_layout()

        if save_pdf:
            pdf_folder = os.path.join(output_folder, period, run, "mtg/pdf")
            os.makedirs(pdf_folder, exist_ok=True)
            plt.savefig(
                os.path.join(pdf_folder, f"{period}_{run}_all_strings_FT_SF.pdf"),
                bbox_inches="tight",
            )

        shelf[f"{period}_{run}_all_strings_FT_SF"] = pickle.dumps(fig)
        plt.close(fig)

        # --- Event rates ---
        fig, ax = plt.subplots(figsize=(10, 3.5))

        # base sample
        base = (
            ged_pul.geds
            & ~ged_pul.puls
            & ~forced.is_forced
            & ~is_dis.is_delayed_discharge
        )

        ser = pd.to_datetime(
            forced.timestamp[ged_pul.geds & ~ged_pul.puls & ~forced.is_forced], unit="s"
        )
        ser_dis = pd.to_datetime(
            forced.timestamp[
                ged_pul.geds
                & ~ged_pul.puls
                & ~forced.is_forced
                & is_dis.is_delayed_discharge
            ],
            unit="s",
        )

        ser_pass = pd.to_datetime(forced.timestamp[base & is_bb.is_bb_like], unit="s")
        ser_fail = pd.to_datetime(forced.timestamp[base & ~is_bb.is_bb_like], unit="s")

        for s, label, color in [
            (ser, "All events", "dimgrey"),
            (ser_dis, "Delayed discharges", "darkorange"),
            (ser_fail, "Failing QC", "crimson"),
            (ser_pass, "Surviving QC", "dodgerblue"),
        ]:
            if s.empty:
                continue
            freq, bin_edges = np.histogram(
                s, bins=pd.date_range(start=s.min(), end=s.max(), freq="h")
            )
            ax.stairs(freq / 3600 * 1000 / on_mass, bin_edges, label=label, color=color)

        ax.set_ylabel("Hourly rate normalized by ON mass (mHz/kg)")
        ax.legend(
            title=f"Last cycle: {last_cycle}\nON mass = {on_mass:.1f} kg",
            loc="upper right",
        )
        ax.grid(False)
        fig.tight_layout()

        if save_pdf:
            pdf_folder = os.path.join(output_folder, period, run, "mtg/pdf")
            os.makedirs(pdf_folder, exist_ok=True)
            plt.savefig(
                os.path.join(pdf_folder, f"{period}_{run}_event_rate_qc.pdf"),
                bbox_inches="tight",
            )
        shelf[f"{period}_{run}_event_rate_qc"] = pickle.dumps(fig)
        plt.close(fig)

        # --- Dead time from discharge windows ---
        mask_puls = ged_pul.puls
        mask_puls_no_dis = ged_pul.puls & ~is_dis.is_delayed_discharge

        length = len(ak.flatten(ak.where(mask_puls)))
        length_no_dis = len(ak.flatten(ak.where(mask_puls_no_dis)))

        # pulser period is assumed to be of 20 s
        livetime_total = length * 20
        livetime_no_dis = length_no_dis * 20

        dead_time_s = livetime_total - livetime_no_dis
        dead_time_pct = (
            (dead_time_s / livetime_total * 100) if livetime_total > 0 else 0.0
        )

        shelf[f"{period}_{run}_dead_time_pct"] = dead_time_pct
        shelf[f"{period}_{run}_dead_time_s"] = dead_time_s

        utils.logger.info(
            f"...dead time from discharges: {dead_time_s:.1f} s ({dead_time_pct:.4f} %)"
        )


def box_summary_plot(
    period: str,
    run: str,
    pars: dict,
    det_info: dict,
    results: dict,
    last_cycle: str,
    info: dict,
    output_dir: str,
    data_type: str,
    save_pdf: bool,
    run_to_apply=None,
    group_label: str = "String",
    group_means: dict = None,
    group_mean_label: str = "Mean rate",
    group_limits: dict = None,
    group_limits_label: str = None,
    faded: set = None,
    flagged: set = None,
    red_points: set = None,
    minmax_flags: dict = None,
    group_alerts: dict = None,
    missing: set = None,
    notes: list = None,
):
    """
    Box plot summary for FEP gain variations for multiple detectors.

    Parameters
    ----------
    period : str
        Period to inspect.
    run : str
        Run to inspect.
    pars : dict
        Calibration results for each detector.
    det_info : dict
        Dictionary with channel names, IDs, and mapping to string and position.
    results : dict
        Dictionary with arrays values (per detector); None if invalid.
    last_cycle : str
        Last cycle of the inspect list; format: YYYYMMDDThhmmssZ.
    info : dict
        Dictionary containing info on a parameter basis (eg label name, file title, colours, limits, ...).
    output_dir : str
        Output folder for saving plots and shelve data.
    data_type : str
        Type of data, either 'cal' or 'phy'.
    save_pdf : bool
        If True, save the summary plot as a PDF.
    run_to_apply :
        Run to apply (eg see ssc data).
    group_label : str
        Prefix of the labels marking each group of channels (the 'string' entry
        of det_info); default: 'String'. Use '' to show the group name only.
    group_means : dict
        Group ('string' entry of det_info) -> mean value, drawn as a line over
        the channels of that group; default: None (no lines).
    group_mean_label : str
        Legend label of the group lines; default: 'Mean rate'.
    group_limits : dict
        Group -> (low, high) thresholds, drawn as dashed lines over each group
        with the range outside shaded grey; default: None.
    group_limits_label : str
        Legend label of the threshold lines.
    faded : set
        Channels drawn transparent, names in grey (eg not in use); replaces
        the red 'Usability: off' names.
    flagged : set
        Channels outside the thresholds: mean marker and name in red.
    red_points : set
        Further channels whose mean marker is red (name colour unchanged).
    minmax_flags : dict
        Channel name -> (low, high) booleans; the min (max) line is drawn red
        if low (high) is True. Default: None (standard error bars).
    group_alerts : dict
        Group -> text written in red next to the group label (eg 'Flasher!').
    missing : set
        Channels without data that should have data, marked with a red cross.
    notes : list
        Extra lines for the legend title (eg bursts).
    """
    utils.logger.debug("...making summary box plots for %s", info["title"])
    detectors = det_info["detectors"]
    plot_data = []
    for ged, item in results.items():
        if ged not in detectors:
            continue

        meta_info = detectors[ged]

        if item is None or len(item) == 0:
            mean = std = min_val = max_val = np.nan
        else:
            mean = np.nanmean(item)
            std = np.nanstd(item)
            min_val = np.nanmin(item)
            max_val = np.nanmax(item)
        try:
            fwhm = pars[ged]["results"]["ecal"]["cuspEmax_ctc_cal"]["eres_linear"][
                "Qbb_fwhm_in_kev"
            ]
        except (KeyError, TypeError):
            fwhm = np.nan

        plot_data.append(
            {
                "ged": ged,
                "string": meta_info["string"],
                "pos": meta_info["position"],
                "mean": mean,
                "std": std,
                "min": min_val,
                "max": max_val,
                "fwhm": fwhm,
                "usability": meta_info.get("usability", None),
                "died_in_run": meta_info.get("died_in_run", False),
            }
        )

    df_plot = pd.DataFrame(plot_data)
    # sort by string, and then position
    df = df_plot.sort_values(["string", "pos"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))

    if info["title"] in ["FEP_gain_stab", "pulser_stab", "baseln_stab"]:
        plt.axhline(0, color="gray", lw=0.5)

    if not df["fwhm"].isna().all():
        fwhm_label = (
            r"$\pm$FWHM/2"
            if info["title"] == "FEP_gain_stab"
            else r"$\pm$FWHM (threshold)"
        )
        ax.bar(
            x,
            df["fwhm"],
            bottom=-df["fwhm"] / 2,
            width=0.4,
            color="orange",
            alpha=0.2,
            label=fwhm_label,
        )

    is_faded = df["ged"].isin(faded or set()).to_numpy()
    for sel, alpha in [(~is_faded, 1.0), (is_faded, 0.35)]:
        if not sel.any():
            continue
        first = alpha == 1.0 or not (~is_faded).any()
        ax.bar(
            x[sel],
            2 * df["std"][sel],  # total height = twice 1 std
            bottom=(df["mean"] - df["std"])[sel],  # center bar on mean
            width=0.6 if minmax_flags is None else 0.3,
            color="skyblue",
            alpha=0.7 * alpha,
            zorder=2,
            label="±1σ" if first else None,
        )
        red = (flagged or set()) | (red_points or set())
        ax.scatter(
            x[sel],
            df["mean"][sel],
            color=np.where(df["ged"].isin(red), "red", "black")[sel],
            alpha=alpha,
            zorder=4,
            label="Mean" if first else None,
        )
        if minmax_flags is None:
            ax.errorbar(
                x[sel],
                df["mean"][sel],
                yerr=[(df["mean"] - df["min"])[sel], (df["max"] - df["mean"])[sel]],
                fmt="none",
                ecolor="#0266c9" if info["title"] != "FEP_gain_stab" else "red",
                alpha=alpha,
                capsize=4,
                zorder=3,
                label="Min/Max" if first else None,
            )
            continue
        # min and max drawn separately, red on the side that is out
        for k, end in enumerate([df["min"][sel], df["max"][sel]]):
            colors = [
                "red" if minmax_flags.get(n, (False, False))[k] else "#0266c9"
                for n in df["ged"][sel]
            ]
            ax.vlines(
                x[sel], df["mean"][sel], end, colors=colors, alpha=alpha, zorder=3
            )
            # caps as wide as the sigma bar
            ax.hlines(
                end, x[sel] - 0.15, x[sel] + 0.15, colors=colors, alpha=alpha, zorder=3
            )

    ax.set_xticks(x)
    xtick_labels = ax.set_xticklabels(df["ged"], rotation=90)
    for i, label in enumerate(xtick_labels):
        if faded is not None and df.iloc[i]["ged"] in faded:
            label.set_color("gray")
            label.set_alpha(0.6)
        elif df.iloc[i]["usability"] in ["off", "false", False]:
            label.set_color("red")
        if flagged and df.iloc[i]["ged"] in flagged:
            label.set_color("red")
        if any((minmax_flags or {}).get(df.iloc[i]["ged"], ())):
            label.set_bbox(
                dict(boxstyle="round,pad=0.2", fc="red", ec="none", alpha=0.3)
            )
        if df.iloc[i]["usability"] in ["ac"]:
            label.set_color("darkorange")
        if df.iloc[i]["died_in_run"]:
            label.set_color("purple")

    ax.axvline(-0.5, color="gray", ls="--", alpha=0.5)
    ymin, ymax = ax.get_ylim()
    label_y = ymin * (ymax / ymin) ** 0.05 if ymin > 0 else -4
    label_y = label_y if info["title"] != "baseln_spike" else 1
    label_va = "baseline"
    is_pmt_rate = info["title"].startswith("pmt_rate_stab")
    if is_pmt_rate:
        # leave room above the data for the group labels
        # scale to the means and thresholds, not to burst maxima
        top = np.nanmax(list(df["mean"]) + [0])
        if group_limits:
            top = max(top, np.nanmax([v[1] for v in group_limits.values()] + [0]))
        # full y range (bursts / flashers stay visible)
        ymin, ymax = 0, 1.05 * max(ymax, top)
        ax.set_ylim(ymin, ymax)
        label_y, label_va = 0.98 * ymax, "top"
    unique_strings = df["string"].unique()
    for s in unique_strings:
        idx = df.index[df["string"] == s]
        left, right = idx.min(), idx.max()
        ax.axvline(right + 0.5, color="gray", ls="--", alpha=0.5)
        ax.text(
            left - 0.45,
            label_y,
            f"{group_label} {s}".strip(),
            rotation=90,
            va=label_va,
            ha="left",
        )
        if group_alerts and s in group_alerts:
            ax.text(
                left + 0.1,
                label_y,
                group_alerts[s],
                color="red",
                fontweight="bold",
                ha="left",
                va=label_va,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="red", alpha=0.9),
                zorder=10,
            )
        if group_means and np.isfinite(group_means.get(s, np.nan)):
            mean = group_means[s]
            ax.hlines(
                mean,
                left - 0.5,
                right + 0.5,
                colors="darkgreen",
                linewidths=1,
                label=group_mean_label if s == unique_strings[0] else None,
            )
            ax.text(
                right + 0.45,
                mean + 0.015 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                f"{mean:.0f} /h",
                rotation=90,
                ha="right",
                va="bottom",
                color="darkgreen",
            )
            if group_limits and np.isfinite(group_limits.get(s, (np.nan,))[0]):
                low, high = group_limits[s]
                ax.hlines(
                    [low, high],
                    left - 0.5,
                    right + 0.5,
                    colors="black",
                    linestyles="--",
                    label=(group_limits_label if s == unique_strings[0] else None),
                )
                span = [left - 0.5, right + 0.5]
                ax.fill_between(span, 0, low, color="gray", alpha=0.25, lw=0)
                ax.fill_between(span, high, ymax, color="gray", alpha=0.25, lw=0)
    for i, name in enumerate(df["ged"]):
        if missing and name in missing:
            ax.plot(
                i,
                0.02 * ax.get_ylim()[1],
                "x",
                color="red",
                ms=10,
                mew=2,
                label="No data" if name == sorted(missing)[0] else None,
            )

    ax.set_ylabel(info["ylabel"])
    ax.set_title(f"{info['plot_title']} - {period} - {run}")

    if info["title"] in ["baseln_stab"]:
        ax.axhline(
            -10,
            ls="--",
            color="black",
            label=r"$\pm$" + f"{info['limits'][1]}% threshold",
        )
        ax.axhline(10, ls="--", color="black")
        ax.axhspan(10, 500, color="gray", alpha=0.25)
        ax.axhspan(-10, -500, color="gray", alpha=0.25)
    if info["title"] in ["baseln_spike"]:
        ax.axhline(
            50, ls="--", color="black", label=f"{info['limits'][1]} ADC upper threshold"
        )
        ax.axhspan(50, 500, color="gray", alpha=0.25)

    if info["title"] == "FEP_gain_stab":
        ax.axhline(-2, ls="--", color="black", label=r"$\pm$2 keV threshold")
        ax.axhline(2, ls="--", color="black")
        ax.axhspan(2, 500, color="gray", alpha=0.25)
        ax.axhspan(-2, -500, color="gray", alpha=0.25)
        plt.ylim(-6, 6)
    if info["title"] == "pulser_stab":
        plt.ylim(-6, 6)
    if info["title"] in ["baseln_stab"]:
        plt.ylim(-20, 20)
    if info["title"] in ["baseln_spike"]:
        plt.ylim(0, 100)

    # Create custom legend entries for usability colors
    legend_patches = []
    handles, labels = ax.get_legend_handles_labels()
    if faded is None:
        legend_patches.append(Patch(color="red", label="Usability: off"))
    # red marks (outside the thresholds) are self-explanatory: no legend entry
    if minmax_flags is not None:
        legend_patches.append(
            Line2D([], [], color="#0266c9", marker="_", ms=8, label="Min/Max")
        )
    if not is_pmt_rate:
        legend_patches.append(Patch(color="darkorange", label="Usability: ac"))
        if df["died_in_run"].any():
            legend_patches.append(Patch(color="purple", label="Dropped out during run"))
    all_handles = handles + legend_patches
    plot_title = f"Last cycle: {last_cycle}" if last_cycle is not None else None
    if notes:
        plot_title = "\n".join(([plot_title] if plot_title else []) + list(notes))
    ax.legend(handles=all_handles, loc="upper right", title=plot_title)
    ax.grid(False)
    plt.tight_layout()

    if save_pdf:
        pdf_folder = os.path.join(output_dir, f"{period}/{run}/mtg/pdf")
        os.makedirs(pdf_folder, exist_ok=True)
        plt.savefig(
            os.path.join(
                pdf_folder,
                f"{period}_{run}_{info['title']}.pdf",
            ),
            bbox_inches="tight",
        )

    # serialize+plot in a shelve object
    serialized_plot = pickle.dumps(fig)
    with shelve.open(
        os.path.join(
            output_dir,
            period,
            run,
            f"mtg/l200-{period}-{run}-{data_type}-monitoring",
        ),
        "c",
        protocol=pickle.HIGHEST_PROTOCOL,
    ) as shelf:
        shelf[f"{period}_{run}_{info['title']}"] = serialized_plot

    plt.close()


def get_pmt_counts(files: list, pmts: dict):
    """
    Count, for each PMT, the muon-DAQ events with a pulse in each cycle file.

    Parameters
    ----------
    files : list
        Sorted list of hit-tier LH5 files (one per DAQ cycle).
    pmts : dict
        PMT infos as returned by utils.build_pmt_info()["detectors"].

    Returns
    -------
    counts : dict
        PMT name -> array (one entry per file) with the number of events with
        containsPulse == 1; NaN where the PMT is missing from a file.
    durations : np.ndarray
        Duration of each file in seconds (first to last muon-DAQ event); NaN if
        it cannot be determined.
    """
    counts = {name: np.full(len(files), np.nan) for name in pmts}
    durations = np.full(len(files), np.nan)

    for idx, file in enumerate(files):
        available = set(lh5.ls(file))
        for name, info in pmts.items():
            ch = info["channel_str"]
            if ch not in available:
                continue
            group = lh5.ls(file, f"{ch}/")[0]
            tbl = lh5.read(group, file, field_mask=["containsPulse", "timestamp"])
            counts[name][idx] = np.count_nonzero(tbl["containsPulse"].nda == 1)
            if np.isnan(durations[idx]) and len(tbl["timestamp"].nda) > 1:
                timestamps = tbl["timestamp"].nda
                durations[idx] = timestamps.max() - timestamps.min()

    return counts, durations


def is_pmt_dead_in_run(
    counts,
    durations,
    min_cycles: int = 3,
    max_fraction: float = 0.1,
    min_expected_counts: float = 10,
) -> bool:
    """
    Check whether a PMT stopped working during the run.

    A PMT is flagged if, from some cycle until the end of the run (at least
    ``min_cycles`` cycles), it sees at most ``max_fraction`` of the pulses
    expected from its earlier rate. The earlier rate is the median of the
    per-cycle rates over at least ``min_cycles`` cycles, so that single bursts
    do not bias it. A dead PMT does not need to go to zero (dark counts).

    Parameters
    ----------
    counts : array_like
        Number of events with a pulse per cycle; NaN entries are ignored.
    durations : array_like
        Duration of each cycle in seconds.
    min_cycles : int
        Minimum number of cycles before and after the drop; default: 3.
    max_fraction : float
        Maximum fraction of the earlier rate after the drop; default: 0.1.
    min_expected_counts : float
        Minimum number of pulses expected after the drop from the earlier rate,
        to avoid flagging low-rate PMTs by chance; default: 10.
    """
    counts = np.asarray(counts, dtype=float)
    durations = np.asarray(durations, dtype=float)
    ok = np.isfinite(counts) & np.isfinite(durations) & (durations > 0)
    counts, durations = counts[ok], durations[ok]
    rates = counts / durations

    for k in range(min_cycles, len(counts) - min_cycles + 1):
        rate_before = np.median(rates[:k])
        expected = rate_before * durations[k:].sum()
        if expected >= min_expected_counts and counts[k:].sum() <= (
            max_fraction * expected
        ):
            return True

    return False


def compute_pmt_rates(counts: dict, durations, usable: list, min_fraction: float = 0.0):
    """
    Compute the per-cycle rate of each PMT and the reference rate of a set of PMTs.

    Parameters
    ----------
    counts : dict
        PMT name -> number of events with a pulse per cycle.
    durations : array_like
        Duration of each cycle in seconds.
    usable : list
        PMTs entering the reference rate.
    min_fraction : float
        Cycles shorter than this fraction of the median cycle duration are
        left out of the per-cycle rates (their rate fluctuates strongly);
        default: 0 (all cycles).

    Returns
    -------
    rates : dict
        PMT name -> array of rates in pulses per hour (valid cycles only), or
        None if the PMT has no data.
    reference_rate : float
        Median over the ``usable`` PMTs of their median cycle rate, in pulses
        per hour (NaN if no usable PMT has data). The median over the cycles is
        robust against bursts (eg flashers), the median over the PMTs against a
        single broken PMT that is not yet marked 'off'.
    """
    durations = np.asarray(durations, dtype=float)
    valid = np.isfinite(durations) & (durations > 0)
    shortest = min_fraction * np.median(durations[valid]) if valid.any() else 0

    rates, medians = {}, []
    for name, c in counts.items():
        c = np.asarray(c, dtype=float)
        ok = np.isfinite(c) & valid
        if not ok.any():
            rates[name] = None
            continue
        use = ok & (durations >= shortest)
        rates[name] = c[use] / durations[use] * 3600
        if name in usable and use.any():
            medians.append(np.median(rates[name]))

    reference_rate = float(np.median(medians)) if medians else np.nan

    return rates, reference_rate


def pmt_run_mean_rates(counts: dict, durations) -> dict:
    """Run-mean rate of each PMT in pulses per hour (NaN without data)."""
    durations = np.asarray(durations, dtype=float)
    out = {}
    for name, c in counts.items():
        c = np.asarray(c, dtype=float)
        ok = np.isfinite(c) & np.isfinite(durations) & (durations > 0)
        out[name] = (
            float(c[ok].sum() / durations[ok].sum() * 3600) if ok.any() else np.nan
        )
    return out


def pmt_rate_limits(references: dict, durations, n_sigma: float) -> dict:
    """
    Thresholds of the PMT rates per location, in pulses per hour.

    The range is the location median ± ``n_sigma`` Poisson standard deviations
    of the rate in one cycle, sqrt(median / T) with T the median cycle
    duration of the run (in hours).

    Returns
    -------
    dict
        Location -> (low, high); (NaN, NaN) if there is no reference.
    """
    durations = np.asarray(durations, dtype=float)
    ok = np.isfinite(durations) & (durations > 0)
    hours = np.median(durations[ok]) / 3600 if ok.any() else np.nan
    limits = {}
    for location, ref in references.items():
        if ref > 0 and hours > 0:
            half = n_sigma * np.sqrt(ref / hours)
            limits[location] = (float(max(ref - half, 0.0)), float(ref + half))
        else:
            limits[location] = (np.nan, np.nan)
    return limits


def classify_pmt_rates(run_means: dict, pmts: dict, limits: dict) -> dict:
    """
    Shifter status of each PMT from its run-mean rate.

    Parameters
    ----------
    run_means : dict
        PMT name -> run-mean rate (pulses per hour, NaN without data).
    pmts : dict
        PMT infos with 'string' (location), 'usability' and 'died_in_run'.
    limits : dict
        Location -> (low, high) thresholds in pulses per hour (see
        :func:`pmt_rate_limits`).

    Returns
    -------
    dict
        PMT name -> 'off' (usability not 'on'), 'no data', 'dropped out'
        (during the run, see :func:`is_pmt_dead_in_run`), 'warning' (outside
        the thresholds) or 'ok'.
    """
    status = {}
    for name, info in pmts.items():
        rate = run_means.get(name, np.nan)
        low, high = limits.get(info["string"], (np.nan, np.nan))
        if info.get("usability") != "on":
            status[name] = "off"
        elif not np.isfinite(rate):
            status[name] = "no data"
        elif info.get("died_in_run"):
            status[name] = "dropped out"
        elif rate < low or rate > high:
            status[name] = "warning"
        else:
            status[name] = "ok"
    return status


def limits_label(n_sigma: float) -> str:
    """Legend label of the PMT rate thresholds, eg '±5σ threshold'."""
    return rf"$\pm${n_sigma:g}$\sigma$ threshold"


def find_threshold_crossings(values, low, high) -> list:
    """
    Return the indices at which a series enters the range outside [low, high].

    ``low`` and ``high`` are arrays (or scalars) of the same length as
    ``values``; each index is the first point of a stretch outside the range
    (also at the start of the series). NaN counts as inside.
    """
    values = np.asarray(values, dtype=float)
    with np.errstate(invalid="ignore"):
        out = (values < low) | (values > high)
    return [j for j in range(len(values)) if out[j] and (j == 0 or not out[j - 1])]


def find_cycle_outliers(
    counts, durations, low: float, high: float, min_fraction: float = 0.5
) -> tuple:
    """
    Check whether the rate of single cycles of a PMT is outside [low, high].

    Only cycles of at least ``min_fraction`` of the median cycle duration are
    checked, since the thresholds are computed for a cycle of median length.

    Returns
    -------
    tuple
        (low, high): True if a cycle is below / above the range.
    """
    counts = np.asarray(counts, dtype=float)
    durations = np.asarray(durations, dtype=float)
    ok = np.isfinite(counts) & np.isfinite(durations) & (durations > 0)
    if not ok.any():
        return False, False
    ok &= durations >= min_fraction * np.median(durations[ok])
    rates = counts[ok] / durations[ok] * 3600
    return bool((rates < low).any()), bool((rates > high).any())


def find_pmt_bursts(counts: dict, durations, pmts: dict, usable: list) -> dict:
    """
    Burst indicator per location: highest cycle / median cycle of the location.

    The cycle rate of a location is the median over its ``usable`` PMTs.
    Returns location -> ratio (NaN if not available).
    """
    durations = np.asarray(durations, dtype=float)
    ok = np.isfinite(durations) & (durations > 0)
    out = {}
    for loc in sorted({info["string"] for info in pmts.values()}):
        names = [n for n in usable if pmts[n]["string"] == loc]
        if not names or not ok.any():
            out[loc] = np.nan
            continue
        cyc = np.array([np.asarray(counts[n], dtype=float)[ok] for n in names])
        cyc = np.nanmedian(cyc, axis=0) / durations[ok] * 3600
        med = np.nanmedian(cyc)
        out[loc] = float(np.nanmax(cyc) / med) if med > 0 else np.nan
    return out


def pmt_rates_file(output_folder: str, period: str, run: str, data_type: str) -> str:
    """Path of the per-run YAML summary of the PMT rates."""
    return os.path.join(
        output_folder,
        period,
        run,
        "mtg",
        f"l200-{period}-{run}-{data_type}-pmt-rates.yaml",
    )


def pmt_window_references(
    output_folder: str,
    period: str,
    run: str,
    data_type: str,
    rates: dict,
    durations,
    pmts: dict,
    usable: list,
) -> tuple:
    """
    Location reference rates over the last week of cycles.

    The window is made of the cycles of the current run (``rates``, without
    short cycles, and ``durations`` of all cycles) and, going back, of the
    cycle rates stored in the summaries of the previous runs (see
    :func:`pmt_rate_summary`), up to 'cycle_history_cycles' cycles. The
    reference of a location is the median over its ``usable`` PMTs of their
    median cycle rate in the window.

    Returns
    -------
    references : dict
        Location -> reference rate in pulses per hour (NaN if not available).
    window_durations : np.ndarray
        Durations of the cycles in the window, in seconds.
    """
    n_cycles = utils.MTG_PLOT_INFO["PmtRate"]["cycle_history_cycles"]
    durations = np.asarray(durations, dtype=float)
    valid = np.isfinite(durations) & (durations > 0)
    if not valid.any():
        return {pmt["string"]: np.nan for pmt in pmts.values()}, durations
    window_durations = list(
        durations[valid & (durations >= 0.5 * np.median(durations[valid]))]
    )
    window = {n: list(rates[n]) for n in usable if rates.get(n) is not None}
    for f in reversed(pmt_rates_files(output_folder, period, run, data_type)):
        need = n_cycles - len(window_durations)
        if need <= 0:
            break
        with open(f) as fh:
            summary = yaml.safe_load(fh)
        if (summary["period"], summary["run"]) == (period, run):
            continue  # an earlier update of the current run
        if "cycles" not in summary:
            break
        m = len(summary["cycles"]["keys"])
        take = min(need, m)
        if take == 0:
            continue
        for n in window:
            prev = summary["pmts"].get(n, {}).get("cycle_rates", [])
            prev = prev if len(prev) == m else [np.nan] * m
            window[n] = list(prev[m - take :]) + window[n]
        window_durations = list(summary["cycles"]["durations"][m - take :]) + (
            window_durations
        )

    references = {}
    for location in sorted({pmt["string"] for pmt in pmts.values()}):
        medians = [
            np.nanmedian(window[n])
            for n in window
            if pmts[n]["string"] == location and np.isfinite(window[n]).any()
        ]
        references[location] = float(np.median(medians)) if medians else np.nan
    return references, np.asarray(window_durations, dtype=float)


def pmt_rate_summary(
    auto_dir_path: str,
    output_folder: str,
    period: str,
    run: str,
    last_cycle: str,
    data_type: str = "phy",
    partition: bool = False,
    save_pdf: bool = False,
):
    """
    Summary box plot of the muon-veto PMT rates for a given run.

    For each DAQ cycle, the rate of each PMT is the number of muon-DAQ events
    with ``containsPulse == 1`` per hour. The plot shows, per PMT, mean, ±1σ and
    min/max of this rate over the cycles, and for each PMT location (floor,
    wall, pillbox) the median run-mean rate of its healthy PMTs (PMTs 'on' in
    the metadata that do not drop out during the run, see
    :func:`is_pmt_dead_in_run`) with the thresholds (see
    :func:`pmt_rate_limits`), the range outside shaded grey. Run means outside
    are red; min / max lines outside are red on that side, with the PMT name
    on red. Only PMTs 'on' are shown (their number in the legend); PMTs
    without data are marked with a red cross. Run means, cycle flags, bursts
    (highest cycle / median cycle per location) and the thresholds are stored
    in a YAML file
    (see :func:`pmt_rates_file`) for :func:`pmt_rate_history`.

    Parameters
    ----------
    auto_dir_path : str
        Path to tmp-auto public data files (eg /data2/public/prodenv/prod-blind/tmp-auto).
    output_folder : str
        Path to output folder.
    period : str
        Period to inspect.
    run : str
        Run to inspect.
    last_cycle : str
        Last cycle of the inspect list; format: YYYYMMDDThhmmssZ.
    data_type : str
        Data type to load; default: 'phy'.
    partition : bool
        False if not partition data; default: False.
    save_pdf : bool
        True if you want to save pdf files too; default: False.

    Returns
    -------
    dict or None
        'rates' (PMT name -> array of rates per cycle, pulses per hour),
        'status' (PMT name -> shifter status), 'references' (location ->
        median rate) and 'bursts' (location -> burst indicator); None if no
        data is found.
    """
    utils.logger.debug("...inspecting PMT rates")
    tier = "pht" if partition is True else "hit"
    files = sorted(
        glob.glob(
            os.path.join(
                auto_dir_path, "generated/tier", tier, data_type, period, run, "*.lh5"
            )
        )
    )
    if not files:
        utils.logger.debug(f"...no {tier} files for {period}-{run}, skip PMT rates")
        return None

    start_key = utils.get_start_key(auto_dir_path, data_type, period, run)
    pmt_info = utils.build_pmt_info(
        os.path.join(auto_dir_path, "inputs"), start_key=start_key
    )
    pmts = pmt_info["detectors"]
    if not pmts:
        utils.logger.debug("...no PMTs in the channel map, skip PMT rates")
        return None

    info = utils.MTG_PLOT_INFO["PmtRate"]
    counts, durations = get_pmt_counts(files, pmts)
    for name, pmt in pmts.items():
        pmt["died_in_run"] = is_pmt_dead_in_run(counts[name], durations)
    usable = [
        name
        for name, pmt in pmts.items()
        if pmt["usability"] == "on" and not pmt["died_in_run"]
    ]

    # per-cycle rates without short cycles, as in the threshold check
    results, _ = compute_pmt_rates(counts, durations, usable, min_fraction=0.5)
    run_means = pmt_run_mean_rates(counts, durations)
    # reference over the last week of cycles (this run and the stored ones
    # before), so that it is stable at the start of a run and the same as in
    # the cycle history
    references, window_durations = pmt_window_references(
        output_folder, period, run, data_type, results, durations, pmts, usable
    )
    for location, ref in references.items():
        utils.logger.debug(f"...PMT median rate ({location}): {ref:.1f} /h")
    n_sigma = info["threshold_sigma"]
    limits = pmt_rate_limits(references, window_durations, n_sigma)
    status = classify_pmt_rates(run_means, pmts, limits)
    bursts = find_pmt_bursts(counts, durations, pmts, usable)
    cycle_flags = {
        n: find_cycle_outliers(counts[n], durations, *limits[pmt["string"]])
        for n, pmt in pmts.items()
        if pmt["usability"] == "on"
    }
    # run means outside, also of PMTs that dropped out
    outside = {
        n
        for n, pmt in pmts.items()
        if pmt["usability"] == "on"
        and not limits[pmt["string"]][0] <= run_means[n] <= limits[pmt["string"]][1]
        and np.isfinite(run_means[n])
    }
    # only PMTs in use are shown
    active = {n for n, pmt in pmts.items() if pmt["usability"] == "on"}
    notes = [f"Active PMTs: {len(active)}"]
    # 'Flasher!' where one cycle exceeds burst_factor x the run median
    alerts = {
        loc: "Flasher!"
        for loc, b in bursts.items()
        if np.isfinite(b) and b > info["burst_factor"]
    }
    plot_options = dict(
        group_label="",
        group_means=references,
        group_mean_label="Location median",
        group_limits=limits,
        group_limits_label=limits_label(n_sigma),
        faded=set(),
        flagged={n for n, s in status.items() if s == "warning"},
        red_points=outside,
        minmax_flags=cycle_flags,
        group_alerts=alerts,
        missing={n for n, s in status.items() if s == "no data"},
        notes=notes,
    )
    shown = {n: r for n, r in results.items() if n in active}
    args = (period, run, None, pmt_info, shown, last_cycle)
    box_summary_plot(*args, info, output_folder, data_type, save_pdf, **plot_options)

    # store the run means for the history plots, and the rates of the cycles
    # (without short ones, as in the plot) for the cycle history
    cycle_rates = {n: r for n, r in results.items() if r is not None}
    valid = np.isfinite(durations) & (durations > 0)
    long_cycles = valid & (durations >= 0.5 * np.median(durations[valid]))
    keys = [os.path.basename(f).split("-")[4] for f in files]
    if len(keys) != len(durations):
        keys = [f"cycle{i:03d}" for i in range(len(durations))]
    summary = {
        "period": period,
        "run": run,
        "last_cycle": last_cycle,
        "references": {k: float(v) for k, v in references.items()},
        "threshold_sigma": n_sigma,
        "limits": {k: [float(v[0]), float(v[1])] for k, v in limits.items()},
        "cycles": {
            "keys": [k for k, use in zip(keys, long_cycles) if use],
            "durations": [float(d) for d in durations[long_cycles]],
        },
        "bursts": {k: float(v) for k, v in bursts.items()},
        "pmts": {
            n: {
                "location": pmts[n]["string"],
                "usability": pmts[n]["usability"],
                "died_in_run": bool(pmts[n]["died_in_run"]),
                "status": status[n],
                "run_mean": float(run_means[n]),
                "min": (
                    float(np.min(cycle_rates[n])) if n in cycle_rates else float("nan")
                ),
                "max": (
                    float(np.max(cycle_rates[n])) if n in cycle_rates else float("nan")
                ),
                "cycle_low": cycle_flags.get(n, (False, False))[0],
                "cycle_high": cycle_flags.get(n, (False, False))[1],
                "cycle_rates": (
                    [
                        float(c / d * 3600)
                        for c, d in zip(
                            np.asarray(counts[n], dtype=float)[long_cycles],
                            durations[long_cycles],
                        )
                    ]
                    if pmts[n]["usability"] == "on"
                    else []
                ),
            }
            for n in sorted(pmts)
        },
    }
    out_file = pmt_rates_file(output_folder, period, run, data_type)
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    return {
        "rates": results,
        "status": status,
        "references": references,
        "bursts": bursts,
    }


def location_box(location: str) -> dict:
    """Title box of a PMT location (muted colour, black text) in the per-PMT plots."""
    colors = {"Floor": "#4a7c59", "Pillbox": "#8c6d46", "Wall": "#6a5a8c"}
    return dict(
        boxstyle="round,pad=0.2",
        fc=colors.get(location, "#5f7f8c"),
        ec="none",
        alpha=0.3,
    )


def pmt_rates_files(output_folder: str, period: str, run: str, data_type: str):
    """Per-run PMT rate summaries up to and including ``period``-``run``, sorted."""

    def run_key(path):
        p, r = os.path.basename(path).split("-")[1:3]
        return int(p.lstrip("p")), int(r.lstrip("r"))

    pattern = os.path.join(
        output_folder, "p*", "r*", "mtg", f"l200-p*-r*-{data_type}-pmt-rates.yaml"
    )
    current = (int(period.lstrip("p")), int(run.lstrip("r")))
    return sorted((f for f in glob.glob(pattern) if run_key(f) <= current), key=run_key)


def load_pmt_rate_history(
    output_folder: str, period: str, run: str, data_type: str, n_runs: int
) -> list:
    """
    Per-run PMT rate summaries (see :func:`pmt_rate_summary`) of the last runs.

    Returns the summaries of the ``n_runs`` runs up to and including
    ``period``-``run``, in chronological order.
    """
    out = []
    for f in pmt_rates_files(output_folder, period, run, data_type)[-n_runs:]:
        with open(f) as fh:
            out.append(yaml.safe_load(fh))
    return out


def pmt_rate_history(
    output_folder: str,
    period: str,
    run: str,
    data_type: str = "phy",
    save_pdf: bool = False,
):
    """
    Expert plots of the PMT run-mean rates over the last runs.

    Two figures from the stored per-run summaries (see :func:`pmt_rate_summary`):
    one panel per location with the run-mean rate of each PMT, the location
    median and the thresholds (range outside shaded grey); and one panel per
    PMT ('on' in the current run, common y axis per location) with run mean,
    min / max over the cycles, the median of its own run means over the shown
    runs and the thresholds around it (see :func:`pmt_rate_limits`). Run
    means and min / max sides outside are marked red, in the first figure
    with the PMT name where it leaves the range. PMTs are left out in runs in
    which they are not 'on'.

    Returns
    -------
    list or None
        Names of the runs shown; None if there is no stored summary.
    """
    info = utils.MTG_PLOT_INFO["PmtRate"]
    history = load_pmt_rate_history(
        output_folder, period, run, data_type, info["history_runs"]
    )
    if not history:
        utils.logger.debug("...no PMT rate summaries, skip PMT rate history")
        return None
    labels = [f"{h['period']}-{h['run']}" for h in history]
    x = np.arange(len(history))
    locations = sorted({p["location"] for h in history for p in h["pmts"].values()})
    names = sorted(
        {n for h in history for n, p in h["pmts"].items() if p["usability"] == "on"}
    )

    def series(name, field):
        return np.array(
            [
                (
                    h["pmts"][name].get(field, np.nan)
                    if name in h["pmts"] and h["pmts"][name]["usability"] == "on"
                    else np.nan
                )
                for h in history
            ],
            dtype=float,
        )

    def ref(location):
        return np.array(
            [h["references"].get(location, np.nan) for h in history], dtype=float
        )

    def label_runs(ax):
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90)
        ax.tick_params(labelbottom=True)

    def limit(location, k):
        # stored per run: location median ± threshold_sigma Poisson sigma
        return np.array(
            [h.get("limits", {}).get(location, [np.nan, np.nan])[k] for h in history],
            dtype=float,
        )

    threshold_label = limits_label(history[-1].get("threshold_sigma", np.nan))

    # red is reserved for runs outside the thresholds
    red_box = dict(boxstyle="round,pad=0.2", fc="red", ec="none", alpha=0.3)
    palette = [c for k, c in enumerate(plt.get_cmap("tab20").colors) if k not in (6, 7)]

    def nanmax(arrays, default=1.0):
        values = np.concatenate([np.ravel(v) for v in arrays] + [[np.nan]])
        return np.nanmax(values) if np.isfinite(values).any() else default

    def shade(ax, low, high, top):
        # extend to the plot edges
        xe = np.r_[x[0] - 0.5, x, x[-1] + 0.5]
        lo, hi = np.r_[low[:1], low, low[-1:]], np.r_[high[:1], high, high[-1:]]
        ax.fill_between(xe, 0, lo, color="gray", alpha=0.25, lw=0)
        ax.fill_between(xe, hi, top, color="gray", alpha=0.25, lw=0)
        ax.plot(xe, lo, "k--", lw=1, label=threshold_label)
        ax.plot(xe, hi, "k--", lw=1)

    def mark_outside(ax, values, low, high, ms):
        with np.errstate(invalid="ignore"):
            out = (values < low) | (values > high)
        ax.plot(x[out], values[out], "o", ms=ms, color="red", zorder=5)
        return out

    figs = {}
    fig, axs = plt.subplots(
        len(locations), 1, figsize=(12, 4.5 * len(locations)), squeeze=False
    )
    for ax, location in zip(axs[:, 0], locations):
        members = [
            n
            for n in names
            if any(h["pmts"].get(n, {}).get("location") == location for h in history)
        ]
        median = ref(location)
        low, high = limit(location, 0), limit(location, 1)
        means = {n: series(n, "run_mean") for n in members}
        top = 1.15 * max(nanmax(means.values()), nanmax([high]))
        shade(ax, low, high, top)
        n_at = {}
        for k, name in enumerate(members):
            ax.plot(
                x,
                means[name],
                "o-",
                ms=3,
                lw=1,
                color=palette[k % len(palette)],
                label=name,
            )
            mark_outside(ax, means[name], low, high, ms=5)
            for j in find_threshold_crossings(means[name], low, high):
                # name next to the point; several in the same run alternate
                # right / left of it
                k_at = n_at.get(j, 0)
                n_at[j] = k_at + 1
                side = 1 if k_at % 2 == 0 else -1
                ax.annotate(
                    name,
                    (x[j], means[name][j]),
                    xytext=(6 * side, 10 * (k_at // 2)),
                    textcoords="offset points",
                    ha="left" if side > 0 else "right",
                    va="center",
                    fontsize=plt.rcParams["legend.fontsize"],
                    bbox=red_box,
                    zorder=5,
                )
        ax.plot(x, median, "k-", lw=1.5, label="Location median")
        ax.set_ylim(0, top)
        ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
        ax.set_ylabel(info["ylabel"])
        ax.set_title(
            f"{location} - PMT rate history - last {len(history)} runs up to {period} - {run}"
        )
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles=handles, ncol=6, loc="lower left")
        label_runs(ax)
    fig.tight_layout()
    figs["pmt_rate_history"] = fig

    # one panel per PMT 'on' in the current run, common y axis per location;
    # reference = median of the PMT's own run means over the shown runs,
    # thresholds ± threshold_sigma Poisson sigma of one cycle around it
    current = history[-1]["pmts"]
    alive = sorted(
        (n for n, p in current.items() if p["usability"] == "on"),
        key=lambda n: (current[n]["location"], n),
    )
    cycle_hours = [
        np.median(h["cycles"]["durations"]) / 3600
        for h in history
        if h.get("cycles", {}).get("durations")
    ]
    hours = float(np.median(cycle_hours)) if cycle_hours else 1.0
    own = {}
    for name in alive:
        m = np.nanmedian(series(name, "run_mean"))
        own[name] = (m,) + pmt_rate_limits(
            {0: m}, [hours * 3600], info["threshold_sigma"]
        )[0]
    tops = {}
    for location in locations:
        members = [n for n in alive if current[n]["location"] == location]
        tops[location] = 1.15 * max(
            nanmax([series(n, "run_mean") for n in members]),
            nanmax([[own[n][2] for n in members]]),
        )
    ncol = 5
    nrow = max(1, int(np.ceil(len(alive) / ncol)))
    fig, axs = plt.subplots(nrow, ncol, figsize=(18, 3.6 * nrow), squeeze=False)
    for ax, name in zip(axs.flat, alive):
        location = current[name]["location"]
        median, low, high = own[name]
        mean = series(name, "run_mean")
        shade(ax, np.full(len(x), low), np.full(len(x), high), tops[location])
        # min / max of the cycles as in the rate plot, red on the side that is
        # outside the thresholds
        ends = [series(name, "min"), series(name, "max")]
        with np.errstate(invalid="ignore"):
            reds = [ends[0] < low, ends[1] > high]
        for end, red in zip(ends, reds):
            colors = np.where(red, "red", "#0266c9")
            ax.vlines(x, mean, end, colors=colors, lw=1, zorder=3)
            ax.scatter(x, end, marker="_", s=30, c=colors, zorder=3)
        clipped = ends[1] > tops[location]
        ax.scatter(
            x[clipped],
            np.full(clipped.sum(), 0.97 * tops[location]),
            marker="^",
            s=16,
            c=np.where(reds[1][clipped], "red", "#0266c9"),
            zorder=5,
        )
        ax.plot(x, mean, "o-", ms=3, color="k", lw=1, label="Run mean", zorder=4)
        ax.axhline(median, color="dimgray", lw=1, label="PMT median")
        out = mark_outside(ax, mean, low, high, ms=4)
        # title marked if the PMT is outside the thresholds in the last run
        last = out[-1] or reds[0][-1] or reds[1][-1]
        ax.set_title(name, loc="left", bbox=red_box if last else None)
        ax.set_title(location, loc="right", bbox=location_box(location))
        ax.set_ylim(0, tops[location])
        ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
        ax.set_ylabel(info["ylabel"])
        handles, _ = ax.get_legend_handles_labels()
        handles.append(
            Line2D([], [], color="#0266c9", marker="_", ms=6, label="Min/Max")
        )
        ax.legend(handles=handles, ncol=2, loc="lower left", fontsize=6)
        label_runs(ax)
    for ax in list(axs.flat)[len(alive) :]:
        ax.axis("off")
    fig.suptitle(
        f"PMT rate history per PMT - last {len(history)} runs up to {period} - {run}"
    )
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / fig.get_figheight()))
    figs["pmt_rate_history_per_pmt"] = fig

    for title, fig in figs.items():
        if save_pdf:
            pdf_folder = os.path.join(output_folder, f"{period}/{run}/mtg/pdf")
            os.makedirs(pdf_folder, exist_ok=True)
            fig.savefig(
                os.path.join(pdf_folder, f"{period}_{run}_{title}.pdf"),
                bbox_inches="tight",
            )
        with shelve.open(
            os.path.join(
                output_folder,
                period,
                run,
                f"mtg/l200-{period}-{run}-{data_type}-monitoring",
            ),
            "c",
            protocol=pickle.HIGHEST_PROTOCOL,
        ) as shelf:
            shelf[f"{period}_{run}_{title}"] = pickle.dumps(fig)
        plt.close(fig)

    return labels


def pmt_cycle_history(
    output_folder: str,
    period: str,
    run: str,
    data_type: str = "phy",
    save_pdf: bool = False,
):
    """
    Expert plot of the PMT rate per cycle over the last week, across runs.

    One panel per PMT 'on' in the current run with the rate of each of the
    last 'cycle_history_cycles' cycles (from the stored per-run summaries, see
    :func:`pmt_rate_summary`; run boundaries as dotted lines), the median of
    the PMT's own cycle rates over these cycles and the thresholds around it
    (see :func:`pmt_rate_limits`). Cycles outside are red; the PMT name is on
    red if the last cycle is outside.

    Returns
    -------
    int or None
        Number of cycles shown; None if there is no stored summary with
        cycles.
    """
    info = utils.MTG_PLOT_INFO["PmtRate"]
    n_cycles = info["cycle_history_cycles"]
    blocks = []
    total = 0
    for f in reversed(pmt_rates_files(output_folder, period, run, data_type)):
        with open(f) as fh:
            summary = yaml.safe_load(fh)
        if "cycles" not in summary:
            break
        blocks.insert(0, summary)
        total += len(summary["cycles"]["keys"])
        if total >= n_cycles:
            break
    if not blocks or not blocks[-1]["cycles"]["keys"]:
        utils.logger.debug("...no PMT cycle rates, skip PMT cycle history")
        return None

    current = blocks[-1]["pmts"]
    alive = sorted(
        (n for n, p in current.items() if p["usability"] == "on"),
        key=lambda n: (current[n]["location"], n),
    )

    def series(name, field):
        out = []
        for b in blocks:
            m = len(b["cycles"]["keys"])
            p = b["pmts"].get(name, {}) if name else {}
            if field == "rate":
                values = p.get("cycle_rates", []) if p.get("usability") == "on" else []
                out.append(values if len(values) == m else [np.nan] * m)
            else:
                out.append(b["cycles"][field])
        return np.concatenate([np.asarray(o, dtype=float) for o in out])[-n_cycles:]

    keys = sum((b["cycles"]["keys"] for b in blocks), [])[-n_cycles:]
    run_of = sum(
        ([f"{b['period']}-{b['run']}"] * len(b["cycles"]["keys"]) for b in blocks), []
    )[-n_cycles:]
    x = np.arange(len(keys))
    starts = [j for j in range(1, len(x)) if run_of[j] != run_of[j - 1]]
    locations = sorted({current[n]["location"] for n in alive})
    red_box = dict(boxstyle="round,pad=0.2", fc="red", ec="none", alpha=0.3)
    ticks = x[::12]

    # reference = median of the PMT's own cycle rates over the shown cycles,
    # thresholds ± threshold_sigma Poisson sigma of one cycle around it
    rates = {n: series(n, "rate") for n in alive}
    durations = series(None, "durations")
    references = {
        n: float(np.nanmedian(r)) if np.isfinite(r).any() else np.nan
        for n, r in rates.items()
    }
    limits = pmt_rate_limits(references, durations, info["threshold_sigma"])
    tops = {}
    for location in locations:
        names = [n for n in alive if current[n]["location"] == location]
        members = np.concatenate([rates[n] for n in names])
        highs = [limits[n][1] for n in names if np.isfinite(limits[n][1])]
        tops[location] = 1.15 * np.nanmax(
            [np.nanpercentile(members, 99) if np.isfinite(members).any() else 0]
            + (highs or [1])
        )

    ncol = 5
    nrow = max(1, int(np.ceil(len(alive) / ncol)))
    fig, axs = plt.subplots(nrow, ncol, figsize=(18, 3.6 * nrow), squeeze=False)
    for ax, name in zip(axs.flat, alive):
        location = current[name]["location"]
        top = tops[location]
        rate, median = rates[name], references[name]
        low, high = limits[name]
        ax.axhspan(0, low, color="gray", alpha=0.25, lw=0)
        ax.axhspan(high, top, color="gray", alpha=0.25, lw=0)
        ax.axhline(
            low, color="k", ls="--", lw=1, label=limits_label(info["threshold_sigma"])
        )
        ax.axhline(high, color="k", ls="--", lw=1)
        for j in starts:
            ax.axvline(j - 0.5, color="dimgray", ls=":", lw=1)
            ax.text(
                j,
                0.97 * top,
                run_of[j].split("-")[1],
                fontsize=6,
                rotation=90,
                ha="left",
                va="top",
                color="dimgray",
            )
        ax.plot(x, rate, "-", lw=0.8, color="k", label="Rate per cycle")
        with np.errstate(invalid="ignore"):
            out = (rate < low) | (rate > high)
        ax.plot(x[out], rate[out], "o", ms=2.5, color="red", zorder=5)
        ax.axhline(median, color="dimgray", lw=1, label="PMT median")
        ax.set_title(name, loc="left", bbox=red_box if out[-1] else None)
        ax.set_title(location, loc="right", bbox=location_box(location))
        ax.set_xlim(x[0] - 0.5, x[-1] + 0.5)
        ax.set_ylim(0, top)
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [
                f"{keys[j][4:6]}-{keys[j][6:8]} {keys[j][9:11]}:{keys[j][11:13]}"
                for j in ticks
            ],
            rotation=90,
        )
        ax.tick_params(labelbottom=True)
        ax.set_ylabel(info["ylabel"])
        ax.legend(ncol=2, loc="lower left", fontsize=6)
    for ax in list(axs.flat)[len(alive) :]:
        ax.axis("off")
    fig.suptitle(f"PMT rate per cycle - last {len(x)} cycles up to {period} - {run}")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.4 / fig.get_figheight()))

    title = "pmt_cycle_history_per_pmt"
    if save_pdf:
        pdf_folder = os.path.join(output_folder, f"{period}/{run}/mtg/pdf")
        os.makedirs(pdf_folder, exist_ok=True)
        fig.savefig(
            os.path.join(pdf_folder, f"{period}_{run}_{title}.pdf"), bbox_inches="tight"
        )
    with shelve.open(
        os.path.join(
            output_folder,
            period,
            run,
            f"mtg/l200-{period}-{run}-{data_type}-monitoring",
        ),
        "c",
        protocol=pickle.HIGHEST_PROTOCOL,
    ) as shelf:
        shelf[f"{period}_{run}_{title}"] = pickle.dumps(fig)
    plt.close(fig)

    return len(x)


def get_muon_evt_data(files: list) -> dict:
    """
    Load the muon-DAQ event table (``evt_muon``) of a list of evt-tier files.

    Parameters
    ----------
    files : list
        List of evt-tier LH5 files.

    Returns
    -------
    dict
        'multiplicity', 'energy_sum' and 'is_muon' arrays, one entry per
        muon-DAQ event (empty arrays if no file has an evt_muon table), and
        'duration', the summed time span of the files in seconds.
    """
    fields = ["multiplicity", "energy_sum", "is_muon"]
    data = {field: [] for field in fields}
    duration = 0.0
    for file in files:
        if "evt_muon" not in lh5.ls(file):
            continue
        tbl = lh5.read("evt_muon", file, field_mask=fields + ["timestamp"])
        for field in fields:
            data[field].append(tbl[field].nda)
        timestamps = tbl["timestamp"].nda
        if len(timestamps) > 1:
            duration += timestamps.max() - timestamps.min()

    data = {
        field: np.concatenate(values) if values else np.array([])
        for field, values in data.items()
    }
    data["duration"] = duration

    return data


def muon_cuts_from_config(config_path: str, timestamp: str) -> dict:
    """
    Multiplicity / light cuts of ``is_muon`` of the production config of a run.

    Reads the muon evt config valid at ``timestamp`` (key format
    YYYYMMDDThhmmssZ) from the dataflow config (eg <prodenv>/inputs/dataprod/
    config) and parses its 'is_muon' expression of the form
    ``((evt.multiplicity >= a) | (evt.energy_sum >= b)) &
    ((evt.multiplicity >= c) | (evt.energy_sum <= d)) & ...``.

    Returns
    -------
    dict or None
        'low_multiplicity' ({'multiplicity': a, 'energy_sum': b}), 'flashing'
        ({'multiplicity': c, 'energy_sum': d}) and 'expression'; None if no
        'is_muon' is defined.
    """
    config = TextDB(config_path, lazy=True).on(timestamp, system="phy")
    inputs = config["snakemake_rules"]["tier_evt"]["inputs"]
    files = []
    if "muon_config" in inputs:
        files.append(inputs["muon_config"]["evt_config"])
    evt = inputs.get("evt_config", [])
    files += evt if isinstance(evt, list) else [evt]
    for f in files:
        with open(str(f).replace("$_", config_path)) as fh:
            operations = (yaml.safe_load(fh) or {}).get("operations", {})
        if "is_muon" not in operations:
            continue
        expression = " ".join(str(operations["is_muon"]["expression"]).split())
        number = r"([0-9.]+)"
        low = re.search(
            rf"evt\.multiplicity\s*>=\s*{number}\s*\)\s*\|\s*\(\s*"
            rf"evt\.energy_sum\s*>=\s*{number}",
            expression,
        )
        flash = re.search(
            rf"evt\.multiplicity\s*>=\s*{number}\s*\)\s*\|\s*\(\s*"
            rf"evt\.energy_sum\s*<=\s*{number}",
            expression,
        )
        if not (low and flash):
            utils.logger.warning(f"...cannot parse is_muon cuts: {expression}")
            return None
        return {
            "low_multiplicity": {
                "multiplicity": float(low.group(1)),
                "energy_sum": float(low.group(2)),
            },
            "flashing": {
                "multiplicity": float(flash.group(1)),
                "energy_sum": float(flash.group(2)),
            },
            "expression": expression,
        }
    return None


def pe_per_unit(hit_file: str, max_channels: int = 10) -> float:
    """
    Return the number of p.e. per unit of ``pulseHeight_in_pe`` (and energy_sum).

    The production multiplicity counts pulses in units of
    ``pulseHeight_trigd`` (single p.e. at 1); where both fields exist, the
    ratio trigd / in_pe of the pulses is the factor (2 for the
    ``pulseHeight / 30`` scale of p18-p22). Without ``pulseHeight_trigd``
    pulseHeight_in_pe is in p.e. (factor 1).
    """
    ratios = []
    n_channels = 0
    for ch in lh5.ls(hit_file):
        group = lh5.ls(hit_file, f"{ch}/")
        if not group:
            continue
        names = [f.split("/")[-1] for f in lh5.ls(hit_file, f"{group[0]}/")]
        if "pulseHeight_in_pe" not in names:
            continue  # not a PMT channel
        if "pulseHeight_trigd" not in names:
            return 1.0
        tbl = lh5.read(
            group[0], hit_file, field_mask=["pulseHeight_in_pe", "pulseHeight_trigd"]
        )
        in_pe = np.asarray(tbl["pulseHeight_in_pe"].nda, dtype=float)
        trigd = np.asarray(tbl["pulseHeight_trigd"].nda, dtype=float)
        use = in_pe > 0.1
        ratios.extend(trigd[use] / in_pe[use])
        n_channels += 1
        if n_channels >= max_channels:
            break
    return float(np.round(np.median(ratios), 1)) if ratios else 1.0


def apply_muon_cuts(multiplicity, energy_sum, cuts: dict):
    """
    Select muon-veto events with the multiplicity / light cuts of ``is_muon``.

    Events are rejected if they are in the low-multiplicity bump (multiplicity
    and light both below the 'low_multiplicity' values) or look like a flashing
    PMT (multiplicity below and light above the 'flashing' values).

    Parameters
    ----------
    multiplicity : array_like
        PMT multiplicity per event.
    energy_sum : array_like
        Integral light per event (p.e.).
    cuts : dict
        'low_multiplicity' and 'flashing', each with 'multiplicity' and
        'energy_sum' thresholds.
    """
    multiplicity = np.asarray(multiplicity)
    energy_sum = np.asarray(energy_sum)
    low, flash = cuts["low_multiplicity"], cuts["flashing"]

    return (
        (multiplicity >= low["multiplicity"]) | (energy_sum >= low["energy_sum"])
    ) & ((multiplicity >= flash["multiplicity"]) | (energy_sum <= flash["energy_sum"]))


# muon-veto plot colours (legendstyles AchatBlue / DeepCove, orange for is_muon)
MV_BLUE, MV_ORANGE, MV_DARK = "#07A9FF", "#EE7733", "#1A2A5B"
# font sizes of the muon-veto analysis plots (Analysis/MuonVeto/plotstyle.py)
MV_FONTS = {
    "font.size": 15,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
}


def mv_style():
    """Matplotlib style of the muon-veto analysis plots (legendstyles if available)."""
    try:
        import legendstyles

        # 'default' first: drop the rcParams the monitoring sets globally
        return ["default", legendstyles.LEGEND, MV_FONTS]
    except ImportError:
        utils.logger.debug("...legendstyles not available, default style")
        return ["default", MV_FONTS]


def mv_watermark(ax):
    """LEGEND-200 watermark (no 'preliminary') if legendstyles is available."""
    try:
        import legendstyles
    except ImportError:
        return
    legendstyles.legend_watermark(ax, logo_suffix="-200", approved=True)


def pmt_light_vs_multiplicity(
    auto_dir_path: str,
    output_folder: str,
    period: str,
    run: str,
    last_cycle: str,
    data_type: str = "phy",
    partition: bool = False,
    save_pdf: bool = False,
):
    """
    2D histogram of the muon-veto integral light vs PMT multiplicity for a run.

    Light in p.e. (energy_sum x :func:`pe_per_unit`) with the cut lines of
    the 'is_muon' expression of the production config valid for the run (see
    :func:`muon_cuts_from_config`). A box at the lower right gives the number
    and rate of all MV trigger events and of the events flagged 'is_muon'; the
    'is_muon' line turns red if the flag differs from the cuts by more than
    the 'is_muon_tolerance' fraction of events (broken flag).

    Parameters
    ----------
    auto_dir_path : str
        Path to tmp-auto public data files (eg /data2/public/prodenv/prod-blind/tmp-auto).
    output_folder : str
        Path to output folder.
    period : str
        Period to inspect.
    run : str
        Run to inspect.
    last_cycle : str
        Last cycle of the inspect list; format: YYYYMMDDThhmmssZ.
    data_type : str
        Data type to load; default: 'phy'.
    partition : bool
        False if not partition data; default: False.
    save_pdf : bool
        True if you want to save pdf files too; default: False.

    Returns
    -------
    dict or None
        Number of events, of events passing the cuts and of events flagged as
        'is_muon', the cuts and the p.e. factor; None if no data is found.
    """
    utils.logger.debug("...inspecting muon-veto light vs multiplicity")
    files = []
    for tier in ["evt", "pet"]:
        files = sorted(
            glob.glob(
                os.path.join(
                    auto_dir_path,
                    "generated/tier",
                    tier,
                    data_type,
                    period,
                    run,
                    "*.lh5",
                )
            )
        )
        if files:
            break
    data = get_muon_evt_data(files)
    multiplicity, energy_sum = data["multiplicity"], data["energy_sum"]
    if len(multiplicity) == 0:
        utils.logger.debug(f"...no evt_muon data for {period}-{run}, skip")
        return None

    info = utils.MTG_PLOT_INFO["PmtLightVsMultiplicity"]
    start_key = utils.get_start_key(auto_dir_path, data_type, period, run)
    cuts = muon_cuts_from_config(
        os.path.join(auto_dir_path, "inputs", "dataprod", "config"), start_key
    )
    tier = "pht" if partition is True else "hit"
    hit_files = sorted(
        glob.glob(
            os.path.join(
                auto_dir_path, "generated/tier", tier, data_type, period, run, "*.lh5"
            )
        )
    )
    scale = pe_per_unit(hit_files[0]) if hit_files else 1.0
    hours = data["duration"] / 3600
    n_events = len(multiplicity)
    n_is_muon = int(np.count_nonzero(data["is_muon"]))
    passed = (
        apply_muon_cuts(multiplicity, energy_sum, cuts) if cuts else np.zeros(0, bool)
    )
    summary = {
        "events": n_events,
        "passed_cuts": int(passed.sum()),
        "is_muon": n_is_muon,
        "cuts": cuts,
        "pe_per_unit": scale,
    }

    with plt.style.context(mv_style()):
        fig, ax = plt.subplots(figsize=(10, 6))
        x_bins = np.arange(-0.5, info["max_multiplicity"] + 1.5)
        y_bins = np.logspace(-0.5, 4.5, 101)
        *_, image = ax.hist2d(
            multiplicity,
            np.clip(energy_sum * scale, y_bins[0], None),
            bins=[x_bins, y_bins],
            cmap="viridis",
            norm=matplotlib.colors.LogNorm(),
            cmin=1,
        )
        fig.colorbar(image, ax=ax, label="Counts")

        # cut lines of the production config; multiplicity cuts lie between
        # integer bins
        if cuts:
            low, flash = cuts["low_multiplicity"], cuts["flashing"]
            x_low, x_flash = low["multiplicity"] - 0.5, flash["multiplicity"] - 0.5
            ax.plot(
                [x_bins[0], x_low, x_low],
                [low["energy_sum"] * scale] * 2 + [y_bins[0]],
                color=MV_ORANGE,
                lw=2,
                label="cut for low-multiplicity bump",
            )
            ax.plot(
                [x_bins[0], x_flash, x_flash],
                [flash["energy_sum"] * scale] * 2 + [y_bins[-1]],
                color=MV_ORANGE,
                lw=2,
                ls=":",
                label="cut for flashing PMTs",
            )

        # numbers and rates at the lower right, aligned in columns; the
        # is_muon line turns red if the flag differs from the cuts
        broken = bool(cuts) and (
            abs(summary["passed_cuts"] - n_is_muon)
            > info["is_muon_tolerance"] * n_events
        )
        rows = [
            ("MV trigger events:", n_events, MV_BLUE),
            (r"$\mathit{is\_muon}$ events:", n_is_muon, "red" if broken else MV_ORANGE),
        ]
        size = plt.rcParams["axes.labelsize"]
        columns = [
            [(label, c) for label, _, c in rows],
            [(f"{n}", c) for _, n, c in rows],
            [(f"({n / hours:.1f} /h)" if hours > 0 else "", c) for _, n, c in rows],
        ]
        box = HPacker(
            children=[
                VPacker(
                    children=[
                        TextArea(t, textprops={"color": c, "fontsize": size})
                        for t, c in column
                    ],
                    align="right",
                    pad=0,
                    sep=4,
                )
                for column in columns
            ],
            align="baseline",
            pad=0,
            sep=6,
        )
        anchored = AnchoredOffsetbox(
            loc="lower right", child=box, pad=0.4, borderpad=0.6, frameon=True
        )
        anchored.patch.set_boxstyle("round")
        anchored.patch.set_alpha(0.8)
        ax.add_artist(anchored)

        ax.set_yscale("log")
        ax.set_xlim(x_bins[0], x_bins[-1])
        ax.set_ylim(y_bins[0], y_bins[-1])
        ax.set_xlabel("PMT multiplicity")
        ax.set_ylabel("Integral light / p.e.")
        ax.legend(
            loc="upper right",
            title=f"{period} - {run}, last cycle {last_cycle}",
            alignment="left",
        )
        mv_watermark(ax)

        if save_pdf:
            pdf_folder = os.path.join(output_folder, f"{period}/{run}/mtg/pdf")
            os.makedirs(pdf_folder, exist_ok=True)
            fig.savefig(
                os.path.join(pdf_folder, f"{period}_{run}_{info['title']}.pdf"),
                bbox_inches="tight",
            )

    # serialize+plot in a shelve object
    with shelve.open(
        os.path.join(
            output_folder,
            period,
            run,
            f"mtg/l200-{period}-{run}-{data_type}-monitoring",
        ),
        "c",
        protocol=pickle.HIGHEST_PROTOCOL,
    ) as shelf:
        shelf[f"{period}_{run}_{info['title']}"] = pickle.dumps(fig)

    plt.close(fig)

    return summary


def qc_average(
    auto_dir_path: str,
    output_folder: str,
    det_info: dict,
    period: str,
    run: str,
    last_cycle: str,
    save_pdf: bool,
    pars_to_inspect: list | None = None,
):
    """
    Evaluate the average rate of passing quality cuts for a given run and period across the whole array for different QC flags.

    Parameters
    ----------
    auto_dir_path : str
        Path to tmp-auto public data files (eg /data2/public/prodenv/prod-blind/tmp-auto).
    output_folder : str
        Path to generated monitoring hdf files.
    det_info : dict
        Dictionary with channel names, IDs, and mapping to string and position.
    period : str
        Period to inspect.
    run : str
        Run under inspection.
    last_cycle : str
        Last cycle of the inspect list; format: YYYYMMDDThhmmssZ.
    save_pdf : bool
        True if you want to save pdf files too; default: False.
    pars_to_inspect : list
        List of parameters (boolean flags) to inspect.
    """
    if pars_to_inspect is None:
        pars_to_inspect = [
            "IsHighlyPositivePolarityCandidate",
            "IsValidBlSlope",
            "IsValidBlSlopeRms",
            "IsValidBlPolyRms",
            "IsValidTailRms",
            "IsNotNoiseBurst",
            "IsValidCuspemin",
            "IsValidCuspemax",
            "IsValidTrapTpmax",
            "IsLowCuspemax",
            "IsDischarge",
            "IsSaturated",
        ]

    my_file = os.path.join(
        output_folder, f"{period}/{run}/l200-{period}-{run}-phy-geds.hdf"
    )
    detectors = det_info["detectors"]
    str_chns = det_info["str_chns"]
    utils.logger.debug("...inspecting QC average values")
    if not os.path.exists(my_file):
        utils.logger.warning(f"...file not found: {my_file}. Return!")
        return

    end_folder = os.path.join(
        output_folder,
        period,
        run,
        "mtg",
    )
    os.makedirs(end_folder, exist_ok=True)
    shelve_path = os.path.join(
        end_folder,
        f"l200-{period}-{run}-phy-monitoring",
    )

    usability_map_file = os.path.join(
        output_folder,
        period,
        run,
        f"l200-{period}-{run}-qcp_summary.yaml",
    )
    output = utils.load_yaml_or_default(usability_map_file, detectors)

    with (
        shelve.open(shelve_path, "c", protocol=pickle.HIGHEST_PROTOCOL) as shelf,
        pd.HDFStore(my_file, "r") as store,
    ):
        for par in pars_to_inspect:
            key = f"/IsPhysics_{par}"
            if key not in store:
                utils.logger.debug("...skipping %s (not found in HDF)", par)
                continue

            geds_df_abs = store[key]
            geds_df_abs = filter_series_by_ignore_keys(
                geds_df_abs, utils.IGNORE_KEYS, period
            )

            # time span
            time_min, time_max = geds_df_abs.index.min(), geds_df_abs.index.max()
            diff = (time_max - time_min).total_seconds()

            # rates in mHz
            rates = geds_df_abs.sum(axis=0) / diff * 1000

            fig, ax = plt.subplots(figsize=(12, 4), sharex=True)
            ax.set_title(f"period: {period} - run: {run} - passing {par}")
            dt_condition = False
            if par == "IsDischarge":
                dt = shelf.get(f"{period}_{run}_dead_time_pct", 0)
                ax.set_title(
                    f"period: {period} - run: {run} - passing {par} - tot dead time {dt:.3f}%"
                )
                dt_condition = bool(
                    dt > utils.MTG_PLOT_INFO["tot_discharge_dead_time"]["limits"][1]
                )

            x_labels, xs, ys = [], [], []
            string_indices = {}
            ct = -1

            for string, det_list in str_chns.items():
                indices = []

                for det_name in det_list:
                    det = detectors[det_name]
                    rawid = det["daq_rawid"]

                    ct += 1
                    x_labels.append(det_name)
                    indices.append(ct)
                    if rawid not in rates:
                        utils.logger.debug(
                            f"{det_name} ({rawid}) missing in dataframe for {par}"
                        )
                        continue

                    ys.append(rates[rawid])
                    xs.append(ct)

                    if par in ["IsDischarge", "IsSaturated"]:
                        condition = bool(
                            (rates[rawid] > utils.MTG_PLOT_INFO[par]["limits"][1]).any()
                        )  # no lower limit for rates
                        utils.update_evaluation_in_memory(
                            output,
                            det_name,
                            "phy",
                            utils.MTG_PLOT_INFO[par]["title"],
                            not condition,
                        )

                        utils.update_evaluation_in_memory(
                            output,
                            det_name,
                            "phy",
                            utils.MTG_PLOT_INFO["tot_discharge_dead_time"]["title"],
                            not dt_condition,
                        )

                string_indices[string] = indices

            ax.scatter(xs, ys, color="dodgerblue", marker="o")

            ax.set_ylabel(f"Average rate {par}=True (mHz)")
            ax.set_yscale("log")
            ax.set_xticks(range(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=90)
            ax.grid(False)

            ymin, ymax = ax.get_ylim()
            label_y = ymin * (ymax / ymin) ** 0.05 if ymin > 0 else 0.1
            for string, indices in string_indices.items():
                left, right = min(indices), max(indices)
                if string == 1:
                    ax.axvline(left - 0.5, ls="--", color="k", alpha=0.5)
                ax.axvline(right + 0.5, ls="--", color="k", alpha=0.5)
                ax.text(
                    left,
                    label_y,
                    f"String {string}",
                    rotation=90,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            if par in ["IsDischarge", "IsSaturated"]:
                upper_limit = (
                    ax.get_ylim()[1]
                    if ax.get_ylim()[1] > 5
                    else utils.MTG_PLOT_INFO[par]["limits"][1] * 1.1
                )
                ax.axhspan(
                    utils.MTG_PLOT_INFO[par]["limits"][1],
                    upper_limit,
                    color="gray",
                    alpha=0.25,
                )
                ax.axhline(
                    utils.MTG_PLOT_INFO[par]["limits"][1],
                    ls="--",
                    color="black",
                    label=f"{utils.MTG_PLOT_INFO[par]['limits'][1]} mHz upper threshold",
                )

            ax.legend(title=f"Last cycle: {last_cycle}")
            plt.tight_layout()

            if par in ["IsDischarge", "IsSaturated"]:
                plot_name = f"{period}_{run}_{utils.MTG_PLOT_INFO[par]['title']}_avg"
            else:
                plot_name = f"{period}_{run}_{par}_avg"

            if save_pdf:
                pdf_dir = os.path.join(end_folder, "pdf")
                os.makedirs(pdf_dir, exist_ok=True)
                pdf_name = os.path.join(pdf_dir, f"{plot_name}.pdf")
                fig.savefig(pdf_name)

            shelf[plot_name] = pickle.dumps(fig)
            plt.close(fig)

    with open(usability_map_file, "w") as f:
        yaml.dump(output, f)


def qc_time_series(
    auto_dir_path: str,
    output_folder: str,
    det_info: dict,
    period: str,
    run: str,
    last_cycle: str,
    save_pdf: bool,
    pars_to_inspect: list | None = None,
):
    """
    Evaluate rate over time of passing quality cuts for a given run and period across the whole array for different QC flags.

    Parameters
    ----------
    auto_dir_path : str
        Path to tmp-auto public data files (eg /data2/public/prodenv/prod-blind/tmp-auto).
    output_folder : str
        Path to generated monitoring hdf files.
    det_info : dict
        Dictionary with channel names, IDs, and mapping to string and position.
    period : str
        Period to inspect.
    run : str
        Run under inspection.
    last_cycle : str
        Last cycle of the inspect list; format: YYYYMMDDThhmmssZ.
    save_pdf : bool
        True if you want to save pdf files too; default: False.
    pars_to_inspect : list
        List of parameters (boolean flags) to inspect.
    """
    if pars_to_inspect is None:
        pars_to_inspect = [
            "IsHighlyPositivePolarityCandidate",
            "IsValidBlSlope",
            "IsValidBlSlopeRms",
            "IsValidBlPolyRms",
            "IsValidTailRms",
            "IsNotNoiseBurst",
            "IsValidCuspemin",
            "IsValidCuspemax",
            "IsValidTrapTpmax",
            "IsLowCuspemax",
            "IsDischarge",
            "IsSaturated",
        ]
    my_file = os.path.join(
        output_folder, f"{period}/{run}/l200-{period}-{run}-phy-geds.hdf"
    )
    detectors = det_info["detectors"]
    str_chns = det_info["str_chns"]
    utils.logger.debug("...inspecting QC time series")
    if not os.path.exists(my_file):
        utils.logger.warning(f"...file not found: {my_file}. Return!")
        return

    end_folder = os.path.join(
        output_folder,
        period,
        run,
        "mtg",
    )
    os.makedirs(end_folder, exist_ok=True)
    shelve_path = os.path.join(
        end_folder,
        f"l200-{period}-{run}-phy-monitoring",
    )

    color_cycle = itertools.cycle(plt.cm.tab20.colors)

    with (
        shelve.open(shelve_path, "c", protocol=pickle.HIGHEST_PROTOCOL) as shelf,
        pd.HDFStore(my_file, "r") as store,
    ):

        for par in pars_to_inspect:
            key = f"/IsPhysics_{par}"
            if key not in store:
                utils.logger.debug("...skipping %s (not found in HDF)", key)
                continue

            geds_df_abs = store[key]
            geds_df_abs = filter_series_by_ignore_keys(
                geds_df_abs, utils.IGNORE_KEYS, period
            )

            for string, channel_list in str_chns.items():
                fig, ax = plt.subplots(figsize=(12, 4))

                for channel_name in channel_list:
                    det = detectors[channel_name]
                    rawid = det["daq_rawid"]
                    pos = det["position"]

                    if rawid not in geds_df_abs.columns:
                        utils.logger.debug(
                            f"{channel_name} ({rawid}) missing in dataframe for {par}"
                        )
                        continue

                    data = geds_df_abs[rawid].copy()
                    true_count = data.sum()
                    time_min, time_max = data.index.min(), data.index.max()
                    diff = (time_max - time_min).total_seconds()

                    true_rate_mHz = round(true_count / diff * 1000, 2)
                    hourly_rate = data.resample("1h").sum() / 3600 * 1000

                    color = next(color_cycle)
                    hourly_rate.plot(
                        ax=ax,
                        drawstyle="steps-mid",
                        label=f"{channel_name} - pos {pos} - {true_rate_mHz} mHz",
                        color=color,
                    )

                ax.grid(False)
                ax.set_ylabel(f"{period} {run} - 1h {par} rate (mHz)")
                fig.suptitle(f"{period} {run} - String: {string}")
                if par in ["IsDischarge", "IsSaturated"]:
                    upper_limit = (
                        ax.get_ylim()[1]
                        if ax.get_ylim()[1] > 5
                        else utils.MTG_PLOT_INFO[par]["limits"][1] * 1.1
                    )
                    ax.axhspan(
                        utils.MTG_PLOT_INFO[par]["limits"][1],
                        upper_limit,
                        color="gray",
                        alpha=0.25,
                    )
                    ax.axhline(
                        utils.MTG_PLOT_INFO[par]["limits"][1],
                        ls="--",
                        color="black",
                        label=f"{utils.MTG_PLOT_INFO[par]['limits'][1]} mHz upper threshold",
                    )
                ax.legend(title=f"Last cycle: {last_cycle}")
                plt.tight_layout()

                if par in ["IsDischarge", "IsSaturated"]:
                    plot_name = f"{period}_{run}_string{string}_{utils.MTG_PLOT_INFO[par]['title']}"  # "_rate" already in the title
                else:
                    plot_name = f"{period}_{run}_string{string}_{par}_rate"

                if save_pdf:
                    pdf_dir = os.path.join(end_folder, "pdf", f"st{string}")
                    os.makedirs(pdf_dir, exist_ok=True)
                    pdf_name = os.path.join(pdf_dir, f"{plot_name}.pdf")
                    fig.savefig(pdf_name)

                # serialize+save plot
                shelf[plot_name] = pickle.dumps(fig)
                plt.close(fig)


def get_energy_key(
    ecal_results: dict,
) -> dict:
    """
    Retrieve the energy calibration results from a given dictionary.

    This function searches for specific keys ('cuspEmax_ctc_runcal' or 'cuspEmax_ctc_cal') in the input `ecal_results` dictionary.
    It returns a sub-dictionary if one of the keys is found, otherwise an empty dictionary is returned.

    Parameters
    ----------
    ecal_results : dict
        Dictionary containing energy calibration results.
    """
    cut_dict = {}
    for key in ["cuspEmax_ctc_runcal", "cuspEmax_ctc_cal"]:
        if key in ecal_results:
            cut_dict = ecal_results[key]
            break
    else:
        utils.logger.debug("No cuspEmax key")
        return cut_dict

    return cut_dict


def get_calibration_file(folder_par: str) -> dict:
    """
    Return the content of the JSON/YAML calibration file in folder_par.

    Parameters
    ----------
    folder_par : str
        Path to the folder containing calibration summary files.
    """
    files = os.listdir(folder_par)
    json_files = [f for f in files if f.endswith(".json")]
    yaml_files = [f for f in files if f.endswith((".yaml", ".yml"))]

    if json_files:
        filepath = os.path.join(folder_par, json_files[0])
        with open(filepath) as f:
            pars_dict = json.load(f)
    elif yaml_files:
        filepath = os.path.join(folder_par, yaml_files[0])
        with open(filepath) as f:
            pars_dict = yaml.load(f, Loader=yaml.CLoader)
    else:
        raise FileNotFoundError(f"No JSON or YAML file found in {folder_par}")

    return pars_dict


def extract_fep_peak(pars_dict: dict, channel: str):
    """
    Return fep_peak_pos, fep_peak_pos_err, fep_gain, fep_gain_err.

    Parameters
    ----------
    pars_dict : dict
        Dictionary containing calibration outputs.
    channel : str
        Channel name or IDs.
    """
    if channel not in pars_dict:
        return np.nan, np.nan, np.nan, np.nan

    # for FEP peak, we want to look at the behaviour over time; take 'ecal' results (not partition ones!)
    ecal_results = pars_dict[channel]["results"]["ecal"]
    pk_fits = get_energy_key(ecal_results).get("pk_fits", {})

    try:
        fep_energy = [p for p in sorted(pk_fits) if 2613 < float(p) < 2616][0]
        try:
            fep_peak_pos = pk_fits[fep_energy]["parameters_in_ADC"]["mu"]
            fep_peak_pos_err = pk_fits[fep_energy]["uncertainties_in_ADC"]["mu"]
        except (KeyError, TypeError):
            fep_peak_pos = pk_fits[fep_energy]["parameters"]["mu"]
            fep_peak_pos_err = pk_fits[fep_energy]["uncertainties"]["mu"]

        fep_gain = fep_peak_pos / 2614.5
        fep_gain_err = fep_peak_pos_err / 2614.5

    except (KeyError, TypeError, IndexError):
        return np.nan, np.nan, np.nan, np.nan

    return fep_peak_pos, fep_peak_pos_err, fep_gain, fep_gain_err


def extract_resolution_at_q_bb(
    pars_dict: dict, channel: str, key_result: str, fit: str = "linear"
):
    """
    Return Qbb_fwhm (linear resolution) and Qbb_fwhm_quad (quadratic resolution).

    Parameters
    ----------
    pars_dict : dict
        Dictionary containing calibration outputs.
    channel : str
        Channel name or IDs (eg ch10000).
    key_result : str
        Key name used to extract the resolution results from the parsed file.
    fit : str
        Fitting method used for energy resolution, either 'linear' or 'quadratic'.
    """
    if channel not in pars_dict:
        return np.nan, np.nan

    result = pars_dict[channel]["results"][key_result].get("cuspEmax_ctc_cal", {})
    eres_linear = result.get("eres_linear") or {}
    Qbb_keys = [k for k in eres_linear if "Qbb_fwhm_in_" in k]
    if not Qbb_keys:
        return np.nan, np.nan

    Qbb_fwhm = result["eres_linear"][Qbb_keys[0]]
    Qbb_fwhm_quad = result["eres_quadratic"][Qbb_keys[0]] if fit != "linear" else np.nan

    return Qbb_fwhm, Qbb_fwhm_quad


def evaluate_fep_cal(
    pars_dict: dict, channel: str, fep_peak_pos: float, fep_peak_pos_err: float
):
    """
    Return calibrated FEP position (fep_cal) and error (fep_cal_err).

    Parameters
    ----------
    pars_dict : dict
        Dictionary containing calibration outputs.
    channel : str
        Channel name or IDs.
    fep_peak_pos : float
        Uncalibrated FEP position.
    fep_peak_pos_err : float
        Uncalibrated FEP position error.
    """
    if channel not in pars_dict:
        return np.nan, np.nan

    ecal_results = get_energy_key(pars_dict[channel]["pars"]["operations"])
    expr = ecal_results["expression"]
    params = ecal_results["parameters"]

    fep_cal = eval(expr, {}, {**params, "cuspEmax_ctc": fep_peak_pos})
    fep_cal_err = eval(expr, {}, {**params, "cuspEmax_ctc": fep_peak_pos_err})

    return fep_cal, fep_cal_err


def get_run_start_end_times(
    sto,
    tiers: list,
    period: str,
    run: str,
    tier: str,
):
    """
    Determine the start and end timestamps for a given run, including the special case for additional final calibration runs.

    Parameters
    ----------
    sto
        Store object to read timestamps from LH5 files.
    tiers : list of str
        Paths to tier data folders based on the inspected processed version.
    period : str
        Period to inspect.
    run : str
        Run to inspect.
    tier : str
        Tier level for the analysis ('hit', 'phy', etc.).
    """
    folder_tier = os.path.join(tiers[0 if tier == "hit" else 1], "cal", period, run)
    dir_path = os.path.join(tiers[-1], "phy", period)
    pattern = re.compile(
        r"^l\d+-p\d+-r\d+-(cal|hit|raw)-\d{8}T\d{6}Z-tier_(dsp|hit|raw)\.lh5$"
    )

    run_files = sorted(f for f in os.listdir(folder_tier) if pattern.match(f))

    # for when we have a calib run but zero phy runs for a given period
    if os.path.isdir(dir_path) and run not in os.listdir(dir_path):
        run_end_time = pd.to_datetime(
            sto.read(
                "ch1027201/dsp/timestamp", os.path.join(folder_tier, run_files[-1])
            )[-1],
            unit="s",
        )
        run_start_time = run_end_time
    else:
        run_start_time = pd.to_datetime(
            sto.read(
                "ch1027201/dsp/timestamp", os.path.join(folder_tier, run_files[0])
            )[0],
            unit="s",
        )
        run_end_time = pd.to_datetime(
            sto.read(
                "ch1027201/dsp/timestamp", os.path.join(folder_tier, run_files[-1])
            )[-1],
            unit="s",
        )

    return run_start_time, run_end_time


def get_calib_data_dict(
    calib_data: dict,
    channel_info: list,
    tiers: list,
    pars: list,
    period: str,
    run: str,
    tier: str,
    key_result: str,
    fit: str,
    data_type: str,
):
    """
    Extract calibration information for a given run and appends it to the provided dictionary.

    This function loads calibration parameters for a specific detector channel and run,
    parses energy calibration results and resolution information, and evaluates
    derived values such as gain and calibration constants. It appends the extracted data
    to the provided `calib_data` dictionary, which is expected to contain keys like
    "fep", "fep_err", "cal_const", "cal_const_err", "run_start", "run_end", "res", and "res_quad".

    Parameters
    ----------
    calib_data : dict
        Dictionary that accumulates calibration results across runs.
    channel_info : list
        List of [channel ID, channel name].
    tiers : list of str
        Paths to tier data folders based on the inspected processed version.
    pars : list of str
        Paths to parameter .yaml/.json files.
    period : str
        Period to inspect.
    run : str
        Run to inspect.
    tier : str
        Tier level for the analysis ('hit', 'phy', etc.).
    key_result : str
        Key name used to extract the resolution results from the parsed file.
    fit : str
        Fitting method used for energy resolution, either 'linear' or 'quadratic'.
    data_type : str
    """
    sto = lh5.LH5Store()
    channel = channel_info[0]
    channel_name = channel_info[1]

    validity_file = os.path.join(pars[2 if tier == "hit" else 3], "validity.yaml")
    with open(validity_file) as f:
        validity_dict = yaml.load(f, Loader=yaml.CLoader)

    # find first key of current run
    run_path = os.path.join(tiers[2 if tier == "hit" else 3], data_type, period, run)
    if not os.path.exists(run_path):
        return calib_data
    start_key = sorted(os.listdir(run_path))[0].split("-")[4]
    # use key to load the right yaml file
    valid_entries = [e for e in validity_dict if e["valid_from"] <= start_key]
    if valid_entries:
        apply = max(valid_entries, key=lambda e: e["valid_from"])["apply"][0]
        run_to_apply = apply.split("/")[-1].split("-")[2]
    else:
        utils.logger.debug(
            f"No valid calibration was found for {period}-{run}. Return."
        )
        return calib_data

    folder_par = os.path.join(
        pars[2 if tier == "hit" else 3], "cal", period, run_to_apply
    )
    pars_dict = get_calibration_file(folder_par)

    if not all(k.startswith("ch") for k in pars_dict.keys()):
        channel = channel_name

    # retrieve calibration parameters
    fep_peak_pos, fep_peak_pos_err, fep_gain, fep_gain_err = extract_fep_peak(
        pars_dict, channel
    )
    Qbb_fwhm, Qbb_fwhm_quad = extract_resolution_at_q_bb(
        pars_dict, channel, key_result, fit
    )
    fep_cal, fep_cal_err = evaluate_fep_cal(
        pars_dict, channel, fep_peak_pos, fep_peak_pos_err
    )

    # get timestamp for additional-final cal run (only for FEP gain display)
    run_start_time, run_end_time = get_run_start_end_times(
        sto, tiers, period, run_to_apply, tier
    )

    calib_data["fep"].append(fep_gain)
    calib_data["fep_err"].append(fep_gain_err)
    calib_data["cal_const"].append(fep_cal)
    calib_data["cal_const_err"].append(fep_cal_err)
    calib_data["run_start"].append(run_start_time)
    calib_data["run_end"].append(run_end_time)
    calib_data["res"].append(Qbb_fwhm)
    calib_data["res_quad"].append(Qbb_fwhm_quad)

    return calib_data


def add_calibration_runs(period: str | list, run_list: list | dict) -> list:
    """
    Add special calibration runs to the run list for a given period.

    Parameters
    ----------
        period : str | list
            Either a string or list of periods
        run_list : list | dict
            Either a list of runs or a dictionary with period keys
    """
    if isinstance(period, list) and isinstance(run_list, dict):
        # multiple periods
        for p in period:
            if p in CALIB_RUNS and p in run_list:
                run_list[p] = run_list[p] + CALIB_RUNS[p]
    else:
        # single period case
        if period in CALIB_RUNS:
            if isinstance(run_list, list):
                run_list.extend(CALIB_RUNS[period])
            else:
                # run_list might be a dict but period is a string
                if period in run_list:
                    run_list[period] = run_list[period] + CALIB_RUNS[period]

    return run_list


def get_tier_keyresult(tiers: list):
    """
    Retrieve proper tier name (pht or hit) and key_result (partition_ecal or ecal) depending if partitioning data exists or not.

    Parameters
    ----------
    tiers : list
        Base directory containing the tier and parameter folders.
    """
    tier = "hit"
    key_result = "ecal"
    if os.path.isdir(tiers[1]):
        if os.listdir(tiers[1]) != []:
            tier = "pht"
            key_result = "partition_ecal"

    return tier, key_result


def compute_diff(
    values: np.ndarray, initial_value: float | int, scale: float | int
) -> np.ndarray:
    """
    Compute relative differences with respect to an initial value. If the initial value is zero, returns an array of nan values.

    Parameters
    ----------
    values : np.ndarray
        Array of values to compute the differences for.
    initial_value : float
        Reference value for computing relative differences.
    scale : float
        Scaling factor.
    """
    if initial_value == 0:
        return np.full_like(values, np.nan, dtype=float)

    return (values - initial_value) / initial_value * scale


def get_calib_pars(
    path: str,
    period: str | list,
    run_list: list,
    channel_info: list,
    partition: bool,
    data_type: str,
    escale: float,
    fit="linear",
) -> dict:
    """
    Retrieve and process calibration parameters across a list of runs for a given channel.

    This function loads calibration data from JSON/YAML files for each specified run, computes gain and calibration constant evolution over time, and returns a dictionary of relevant quantities, including their relative changes with respect to the initial values.
    It optionally appends special calibration runs at the end of a period, if available.

    Parameters
    ----------
    path : str
        Base directory containing the tier and parameter folders.
    period : str or list
        Period to inspect. Can be a list if multiple periods are inspected.
    run_list : list
        List of run to inspect, or a dictionary mapping periods to lists of runs.
    channel_info : list
        List containing [channel ID, channel name].
    partition : bool
        True if you want to retrieve partition calibration results.
    escale : float
        Scaling factor used to compute relative differences in gain and calibration constant.
    fit : str, optional
        Fit method used for energy resolution ("linear" or "quadratic"), by default "linear".
    """
    # add special calib runs at the end of a period
    run_list = add_calibration_runs(period, run_list)
    run_list = [r for r in run_list if "old" not in str(r)]

    calib_data = {
        "fep": [],
        "fep_err": [],
        "cal_const": [],
        "cal_const_err": [],
        "run_start": [],
        "run_end": [],
        "res": [],
        "res_quad": [],
    }

    tiers, pars = utils.get_tiers_pars_folders(path)

    tier, key_result = get_tier_keyresult(tiers)

    for run in run_list:
        calib_data = get_calib_data_dict(
            calib_data,
            channel_info,
            tiers,
            pars,
            period,
            run,
            tier,
            key_result,
            fit,
            data_type,
        )

    for key, item in calib_data.items():
        calib_data[key] = np.array(item)

    init_cal_const, init_fep = 0, 0
    for cal_, fep_ in zip(calib_data["cal_const"], calib_data["fep"]):
        if init_fep == 0 and fep_ != 0:
            init_fep = fep_
        if init_cal_const == 0 and cal_ != 0:
            init_cal_const = cal_

    calib_data["cal_const_diff"] = compute_diff(
        calib_data["cal_const"], init_cal_const, escale
    )
    calib_data["fep_diff"] = compute_diff(calib_data["fep"], init_fep, escale)

    return calib_data


def find_hdf_file(
    directory: str, include: list[str], exclude: list[str] = None
) -> str | None:
    """
    Find the original HDF monitoring file in a given directory, matching inclusion/exclusion filters.

    Parameters
    ----------
    directory : str
        Path to the folder containing the HDF monitoring files.
    include: list[str]
        List of words that the HDF monitoring file to retrieve must contain.
    exclude: list[str] = None
        List of words that the HDF monitoring file to retrieve must NOT contain.
    """
    exclude = exclude or []
    files = os.listdir(directory)
    candidates = [
        f
        for f in files
        if f.endswith(".hdf")
        and all(tag in f for tag in include)
        and not any(tag in f for tag in exclude)
    ]

    return os.path.join(directory, candidates[0]) if candidates else None


def read_if_key_exists(hdf_path: str, key: str) -> pd.DataFrame | None:
    """
    Read an HDF dataset if the key exists, otherwise return None; handle the case where the parameter is saved under either '/key' or 'key'.

    Parameters
    ----------
    hdf_path : str
        Path to the HDF file.
    key : str
        Key to inspect.
    """
    with pd.HDFStore(hdf_path, mode="r") as f:
        try:
            return f[key]
        except KeyError:
            try:
                return f["/" + key]
            except KeyError:
                return None


def get_dfs(phy_mtg_data: str, period: str, run_list: list, parameter: str):
    """
    Load and concatenate monitoring data from HDF files for a given period and list of runs.

    Parameters
    ----------
    phy_mtg_data : str
        Path to the base directory containing monitoring HDF5 files (typically ending in `/mtg/phy`).
    period : str
        Period to inspect.
    run_list : list
        List of available runs.
    parameter : str
        Parameter name used to construct the HDF key for loading specific datasets (e.g., 'TrapemaxCtcCal' looks for 'IsPulser_TrapemaxCtcCal').
    """
    # lists to accumulate dataframes, concatenated at the endo only
    geds_df_cuspEmax_abs = []
    geds_df_cuspEmax_abs_corr = []
    puls_df_cuspEmax_abs = []

    base_dir = os.path.join(phy_mtg_data, period)
    runs = os.listdir(base_dir)
    runs = [r for r in runs if re.fullmatch(r"r\d{3}", r)]

    for r in runs:
        if r not in run_list:
            continue
        run_dir = os.path.join(base_dir, r)

        # geds file
        hdf_geds = find_hdf_file(run_dir, include=["geds"], exclude=["res", "min"])
        if hdf_geds:
            geds_abs = read_if_key_exists(hdf_geds, f"IsPulser_{parameter}")
            if geds_abs is not None:
                geds_df_cuspEmax_abs.append(geds_abs)

            geds_puls_abs = read_if_key_exists(
                hdf_geds, f"IsPulser_{parameter}_pulser01anaDiff"
            )
            if geds_puls_abs is not None:
                geds_df_cuspEmax_abs_corr.append(geds_puls_abs)
        else:
            utils.logger.debug("...hdf_geds missing in %s", r)

        # pulser file
        hdf_puls = find_hdf_file(
            run_dir, include=["pulser01ana"], exclude=["res", "min"]
        )
        if hdf_puls:
            puls_abs = read_if_key_exists(hdf_puls, f"IsPulser_{parameter}")
            if puls_abs is not None:
                puls_df_cuspEmax_abs.append(puls_abs)
        else:
            utils.logger.debug("...hdf_puls missing in %s", r)

    if (
        not geds_df_cuspEmax_abs
        and not geds_df_cuspEmax_abs_corr
        and not puls_df_cuspEmax_abs
    ):
        return None, None, None
    else:
        return (
            (
                pd.concat(geds_df_cuspEmax_abs, ignore_index=False, axis=0)
                if geds_df_cuspEmax_abs
                else pd.DataFrame()
            ),
            (
                pd.concat(geds_df_cuspEmax_abs_corr, ignore_index=False, axis=0)
                if geds_df_cuspEmax_abs_corr
                else pd.DataFrame()
            ),
            (
                pd.concat(puls_df_cuspEmax_abs, ignore_index=False, axis=0)
                if puls_df_cuspEmax_abs
                else pd.DataFrame()
            ),
        )


def get_traptmax_tp0est(phy_mtg_data: str, period: str, run_list: list):
    """
    Load and concatenate trapTmax and tp0est data from HDF files for a given period and list of runs.

    Parameters
    ----------
    phy_mtg_data : str
        Path to the base directory containing monitoring HDF5 files (typically ending in `/mtg/phy`).
    period : str
        Period to inspect.
    run_list : list
        List of available runs.
    """
    geds_df_trapTmax, geds_df_tp0est = [], []
    puls_df_trapTmax, puls_df_tp0est = [], []

    base_dir = os.path.join(phy_mtg_data, period)
    for r in os.listdir(base_dir):
        if r not in run_list:
            continue
        run_dir = os.path.join(base_dir, r)

        # geds
        hdf_geds = find_hdf_file(run_dir, include=["geds"], exclude=["res", "min"])
        if hdf_geds:
            trapTmax = read_if_key_exists(hdf_geds, "IsPulser_TrapTmax")
            if trapTmax is not None:
                geds_df_trapTmax.append(trapTmax)

            tp0est = read_if_key_exists(hdf_geds, "IsPulser_Tp0Est")
            if tp0est is not None:
                geds_df_tp0est.append(tp0est)

        # pulser
        hdf_puls = find_hdf_file(
            run_dir, include=["pulser01ana"], exclude=["res", "min"]
        )
        if hdf_puls:
            trapTmax = read_if_key_exists(hdf_puls, "IsPulser_TrapTmax")
            if trapTmax is not None:
                puls_df_trapTmax.append(trapTmax)

            tp0est = read_if_key_exists(hdf_puls, "IsPulser_Tp0Est")
            if tp0est is not None:
                puls_df_tp0est.append(tp0est)

    return (
        (
            pd.concat(geds_df_trapTmax, ignore_index=False)
            if geds_df_trapTmax
            else pd.DataFrame()
        ),
        (
            pd.concat(geds_df_tp0est, ignore_index=False)
            if geds_df_tp0est
            else pd.DataFrame()
        ),
        (
            pd.concat(puls_df_trapTmax, ignore_index=False)
            if puls_df_trapTmax
            else pd.DataFrame()
        ),
        (
            pd.concat(puls_df_tp0est, ignore_index=False)
            if puls_df_tp0est
            else pd.DataFrame()
        ),
    )


def filter_series_by_ignore_keys(
    series_to_filter: pd.Series, skip_keys: dict, period: str
):
    """
    Remove data from a time-indexed pandas Series that falls within time ranges specified by start and stop timestamps for a given period.

    Parameters
    ----------
    series_to_filter : pd.Series
        The time-indexed pandas Series to be filtered.
    skip_keys : dict
        Dictionary mapping periods to sub-dictionaries containing 'start_keys' and 'stop_keys' lists with timestamp strings in the format '%Y%m%dT%H%M%S%z'.
    period : str
        The period to check for keys to ignore. If not present, the series is returned unmodified.
    """
    if period not in skip_keys:
        return series_to_filter

    start_keys = skip_keys[period]["start_keys"]
    stop_keys = skip_keys[period]["stop_keys"]

    for ki, kf in zip(start_keys, stop_keys):
        isolated_ki = pd.to_datetime(ki.replace("Z", "+0000"), format="%Y%m%dT%H%M%S%z")
        isolated_kf = pd.to_datetime(kf.replace("Z", "+0000"), format="%Y%m%dT%H%M%S%z")
        series_to_filter = series_to_filter[
            (series_to_filter.index < isolated_ki)
            | (series_to_filter.index > isolated_kf)
        ]

    return series_to_filter


def filter_by_period(series: pd.Series, period: str | list) -> pd.Series:
    """
    Return a series filtered by ignore keys for the given period(s).

    Parameters
    ----------
    series : pd.Series
        Input time series (indexed by timestamps) to filter.
    period : str or list
        Period (or list of periods) to inspect.
    """
    if isinstance(period, list):
        for p in period:
            series = filter_series_by_ignore_keys(series, IGNORE_KEYS, p)
    else:
        series = filter_series_by_ignore_keys(series, IGNORE_KEYS, period)

    return series


def compute_diff_and_rescaling(
    series: pd.Series, reference: float, escale: float, variations: bool
):
    """
    Compute relative differences (if 'variations' is True) and rescale values by 'escale'.

    Parameters
    ----------
    series : pd.Series
        Input time series of numerical values.
    reference : float
        Reference value used to compute relative differences.
    escale : float
        Scaling factor, eg 2039 keV.
    variations : bool
        If true, compute relative difference (series - reference)/reference.
    """
    if variations:
        diff = (series - reference) / reference
    else:
        diff = series.copy()

    return diff, diff * escale


def resample_series(series: pd.Series, resampling_time: str, mask: pd.Series):
    """
    Calculate mean/std for resampled time ranges to which a mask is then applied. The function already adds UTC timezones to the series.

    Parameters
    ----------
    series : pd.Series
        Input time series of numerical values.
    resampling_time : str
        Resampling frequency, eg '1h'.
    mask : pd.Series
        Boolean mask aligned to the datetime index; false values mark timestamps that should be excluded, ie set to nan value.
    """
    mean = series.resample(resampling_time).mean()
    std = series.resample(resampling_time).std()

    # add UTC timezone
    if mean.index.tz is None:
        mean = mean.tz_localize("UTC")
        std = std.tz_localize("UTC")
    # different timezone, convert to UTC
    elif mean.index.tz != pytz.UTC:
        mean = mean.tz_convert("UTC")
        std = std.tz_convert("UTC")

    # ensure mask has the same timezone as the resampled series
    if not mask.index.tz:
        mask = mask.tz_localize("UTC")

    # set to nan when the mask is False
    mask = mask.reindex(mean.index, fill_value=False)
    mean[~mask] = np.nan
    std[~mask] = np.nan

    return mean, std


def get_pulser_data(
    resampling_time: str,
    period: str | list,
    dfs: list,
    channel: str,
    escale: float,
    variations=False,
) -> dict:
    """
    Return a dictionary of geds and pulser filtered dataframes for which a time resampling is performed.

    Parameters
    ----------
    resampling_time : str
        Resampling time, eg '1HH' or '10T'.
    period : str | list
        Period or list of periods to inspect.
    dfs : list
        List of dataframes for geds and pulser events.
    channel : str
        Channel to inspect.
    escale : float
        Scaling factor used to compute relative differences in gain and calibration constant.
    variations : bool
        True if you want to retrieve % variations (default: False).
    """
    # geds
    ser_ged_cusp = dfs[0][channel].sort_index()
    ser_ged_cusp = filter_by_period(ser_ged_cusp, period)
    ser_ged_cusp = ser_ged_cusp[
        ~ser_ged_cusp.index.duplicated(keep="first")
    ]  # remove duplicates
    ser_pul_tp0est_new = pd.DataFrame()

    if ser_ged_cusp.empty:
        utils.logger.debug("...geds series is empty after filtering")
        return None

    # check if these dfs are empty or not - if not, then remove spikes
    if isinstance(dfs[6], pd.DataFrame) and not dfs[6].empty:
        ser_pul_tp0est = dfs[6][1027203].sort_index()
        ser_pul_tp0est = filter_by_period(ser_pul_tp0est, period)
        ser_pul_tp0est = ser_pul_tp0est[
            ~ser_pul_tp0est.index.duplicated(keep="first")
        ]  # remove duplicates

        low_lim = 4.8e4
        upp_lim = 5.0e4
        mask = (ser_pul_tp0est > low_lim) & (ser_pul_tp0est < upp_lim)
        ser_pul_tp0est_new = ser_pul_tp0est[mask]

        if not ser_pul_tp0est_new.empty:
            valid_idx = ser_ged_cusp.index.intersection(ser_pul_tp0est_new.index)
            ser_ged_cusp = ser_ged_cusp.reindex(valid_idx)

    # if before, potential mismatches with ser_pul_tp0est
    ser_ged_cusp = ser_ged_cusp.dropna()
    # compute average over the first 10% of elements
    n_elements = max(int(len(ser_ged_cusp) * 0.10), 1)
    ged_cusp_av = np.nanmean(ser_ged_cusp.iloc[:n_elements])
    if np.isnan(ged_cusp_av):
        utils.logger.debug("...the geds average is NaN")
        return None

    ser_ged_cuspdiff, ser_ged_cuspdiff_kev = compute_diff_and_rescaling(
        ser_ged_cusp, ged_cusp_av, escale, variations
    )

    # hour counts masking
    mask = ser_ged_cusp.resample(resampling_time).count() > 0

    # resample geds series
    ged_cusp_hr_av, ged_cusp_hr_std = resample_series(
        ser_ged_cuspdiff_kev, resampling_time, mask
    )
    ged_index = ged_cusp_hr_av.index

    # pulser series
    ser_pul_cusp = ser_pul_cuspdiff = ser_pul_cuspdiff_kev = pul_cusp_hr_av = (
        pul_cusp_hr_std
    ) = None
    ged_cusp_corr = ged_cusp_corr_kev = ged_cusp_cor_hr_av = ged_cusp_cor_hr_std = None
    # ...if pulser is available:
    if not dfs[2].empty:
        ser_pul_cusp = dfs[2][1027203].sort_index()
        ser_pul_cusp = ser_pul_cusp[
            ~ser_pul_cusp.index.duplicated(keep="first")
        ]  # remove duplicates
        ser_pul_cusp = filter_by_period(ser_pul_cusp, period)

        # pulser average and diffs
        if not ser_pul_cusp.empty:
            # check if these dfs are empty or not - if not, then remove spikes
            if isinstance(dfs[6], pd.DataFrame) and not dfs[6].empty:
                if not ser_pul_tp0est_new.empty:
                    valid_idx = ser_pul_cusp.index.intersection(
                        ser_pul_tp0est_new.index
                    )
                    ser_pul_cusp = ser_pul_cusp.reindex(valid_idx)

            # if before, potential mismatches with ser_pul_tp0est
            ser_pul_cusp = ser_pul_cusp.dropna()
            n_elements_pul = max(int(len(ser_pul_cusp) * 0.10), 1)
            pul_cusp_av = np.nanmean(ser_pul_cusp.iloc[:n_elements_pul])
            ser_pul_cuspdiff, ser_pul_cuspdiff_kev = compute_diff_and_rescaling(
                ser_pul_cusp, pul_cusp_av, escale, variations
            )

            pul_cusp_hr_av, pul_cusp_hr_std = resample_series(
                ser_pul_cuspdiff_kev, resampling_time, mask
            )
            pul_cusp_hr_av = pul_cusp_hr_av.reindex(ged_index)
            pul_cusp_hr_std = pul_cusp_hr_std.reindex(ged_index)

            # corrected GED
            common_index = ser_ged_cuspdiff.index.intersection(ser_pul_cuspdiff.index)
            ged_cusp_corr = (
                ser_ged_cuspdiff[common_index] - ser_pul_cuspdiff[common_index]
            )
            ged_cusp_corr_kev = ged_cusp_corr * escale
            ged_cusp_cor_hr_av, ged_cusp_cor_hr_std = resample_series(
                ged_cusp_corr_kev, resampling_time, mask
            )
            ged_cusp_cor_hr_av = ged_cusp_cor_hr_av.reindex(ged_index)
            ged_cusp_cor_hr_std = ged_cusp_cor_hr_std.reindex(ged_index)

    return {
        "ged": {
            "cusp": ser_ged_cusp,
            "cuspdiff": ser_ged_cuspdiff,
            "cuspdiff_kev": ser_ged_cuspdiff_kev,
            "kevdiff_av": ged_cusp_hr_av,
            "kevdiff_std": ged_cusp_hr_std,
        },
        "pul_cusp": {
            "raw": ser_pul_cusp,
            "rawdiff": ser_pul_cuspdiff,
            "kevdiff": ser_pul_cuspdiff_kev,
            "kevdiff_av": pul_cusp_hr_av,
            "kevdiff_std": pul_cusp_hr_std,
        },
        "diff": {
            "raw": None,
            "rawdiff": ged_cusp_corr,
            "kevdiff": ged_cusp_corr_kev,
            "kevdiff_av": ged_cusp_cor_hr_av,
            "kevdiff_std": ged_cusp_cor_hr_std,
        },
    }


def build_new_files(generated_path: str, period: str, run: str, data_type="phy"):
    """
    Generate and store resampled HDF files for a given data run and extract summary info.

    This function:

      - loads the original `.hdf` file for the specified `period` and `run`
      - extracts available keys from the HDF file
      - resamples all applicable time series data into multiple time intervals (10min, 60min)
      - stores each resampled dataset into a separate HDF file
      - extracts metadata from the 'info' key and saves it as a .yaml file

    Parameters
    ----------
    generated_path : str
        Root directory where the data is stored and where new files will be written.
    period : str
        Period (e.g. 'p03') used to construct paths.
    run : str
        Run (e.g. 'r001') used to construct paths.
    data_type : str
        Data type to load; default: 'phy'.
    """
    data_file = os.path.join(
        generated_path,
        "generated/plt/hit",
        data_type,
        period,
        run,
        f"l200-{period}-{run}-{data_type}-geds.hdf",
    )

    if not os.path.exists(data_file):
        utils.logger.debug(f"File not found: {data_file}. Exit here.")
        sys.exit()

    with h5py.File(data_file, "r") as f:
        my_keys = list(f.keys())

    info_dict = {"keys": my_keys}

    resampling_times = ["10min", "60min"]

    for idx, resample_unit in enumerate(resampling_times):
        new_file = os.path.join(
            generated_path,
            "generated/plt/hit",
            data_type,
            period,
            run,
            f"l200-{period}-{run}-{data_type}-geds-res_{resample_unit}.hdf",
        )
        # remove it if already exists so we can start again to append resampled data
        if os.path.exists(new_file):
            os.remove(new_file)

        for k in my_keys:
            if "info" in k:
                # do it once
                if idx == 0:
                    original_df = pd.read_hdf(data_file, key=k)
                    original_df = original_df.astype(str)
                    info_dict.update(
                        {
                            k: {
                                "subsystem": original_df.loc["subsystem", "Value"],
                                "unit": original_df.loc["unit", "Value"],
                                "label": original_df.loc["label", "Value"],
                                "event_type": original_df.loc["event_type", "Value"],
                                "lower_lim_var": original_df.loc[
                                    "lower_lim_var", "Value"
                                ],
                                "upper_lim_var": original_df.loc[
                                    "upper_lim_var", "Value"
                                ],
                                "lower_lim_abs": original_df.loc[
                                    "lower_lim_abs", "Value"
                                ],
                                "upper_lim_abs": original_df.loc[
                                    "upper_lim_abs", "Value"
                                ],
                            }
                        }
                    )
                continue

            original_df = pd.read_hdf(data_file, key=k)

            # mean dataframe is kept
            if "_mean" in k:
                original_df.to_hdf(new_file, key=k, mode="a")
                continue

            original_df.index = pd.to_datetime(original_df.index)
            # resample
            resampled_df = original_df.resample(resample_unit).mean()
            # substitute the original df with the resampled one
            original_df = resampled_df
            # append resampled data to the new file
            resampled_df.to_hdf(new_file, key=k, mode="a")

        if idx == 0:
            json_output = os.path.join(
                generated_path,
                "generated/plt/hit",
                data_type,
                period,
                run,
                f"l200-{period}-{run}-{data_type}-geds-info.yaml",
            )
            with open(json_output, "w") as file:
                json.dump(info_dict, file, indent=4)


def plot_time_series(
    auto_dir_path: str,
    phy_mtg_data: str,
    output_folder: str,
    data_type: str,
    period: str,
    runs: list,
    current_run: str,
    det_info: dict,
    save_pdf: bool,
    escale_val: float,
    last_checked: float | None,
    last_cycle: str,
    partition: bool,
    quadratic: bool,
    zoom: bool,
):
    """
    Generate and save time-series plots of calibration and monitoring data for germanium detectors across multiple runs.

    This function collects physics and calibration data from HDF5 monitoring files and visualizes stability over time.
    Channels with no pulser entries are automatically skipped.
    Corrections are applied to the gain if pulser data is available ('GED corrected'), otherwise uncorrected data is plotted.
    The plots are saved as pickled objects for later retrieval (eg. in the online Dashboard) and optionally as PDFs:

    - plots saved in shelve database files under ``<output_folder>/<period>/mtg/l200-<period>-phy-monitoring``;
    - if `save_pdf=True`, PDF copies saved under ``<output_folder>/<period>/mtg/pdf/st<string>/``.

    Parameters
    ----------
    auto_dir_path : str
        Path to tmp-auto public data files (eg /data2/public/prodenv/prod-blind/tmp-auto).
    phy_mtg_data : str
        Path to generated monitoring hdf files.
    output_folder : str
        Path to output folder.
    period : str
        Period to inspect.
    runs : list
        Available runs to inspect for a given period.
    current_run : str
        Run under inspection.
    det_info : dict
        Dictionary containing detector metadata.
    save_pdf : bool
        True if you want to save pdf files too; default: False.
    escale_val : float
        Energy scale at which evaluating the gain differences; default: 2039 keV (76Ge Qbb).
    last_checked : float | None
        Timestamp of the last check.
    last_cycle : str
        Last cycle of the inspect list; format: YYYYMMDDThhmmssZ.
    partition : bool
        False if not partition data; default: False.
    quadratic : bool
        True if you want to plot the quadratic resolution too; default: False.
    zoom : bool
        True to zoom over y axis; default: False.
    """
    avail_runs = []
    for entry in runs:
        new_entry = entry.replace(",", "").replace("[", "").replace("]", "")
        avail_runs.append(new_entry)
    dataset = {period: avail_runs}
    period_list = list(dataset.keys())
    xlim_idx = 1
    fit_flag = "quadratic" if quadratic is True else "linear"

    detectors = det_info["detectors"]
    str_chns = det_info["str_chns"]
    usability_map_file = os.path.join(
        output_folder,
        period,
        current_run,
        f"l200-{period}-{current_run}-qcp_summary.yaml",
    )
    output = utils.load_yaml_or_default(usability_map_file, detectors)

    # skip detectors with no pulser entries
    no_puls_dets = utils.NO_PULS_DETS
    flag_expr = " or ".join(
        f'(channel == "{channel}" and period in {periods})'
        for channel, periods in no_puls_dets.items()
    )

    # gain over period
    results = {}
    for index_i in range(len(period_list)):
        period = period_list[index_i]
        run_list = dataset[period]

        (
            geds_df_cuspEmax_abs,
            geds_df_cuspEmax_abs_corr,
            puls_df_cuspEmax_abs,
        ) = get_dfs(phy_mtg_data, period, run_list, "Trapemax")
        geds_df_trapTmax, geds_df_tp0est, puls_df_trapTmax, puls_df_tp0est = (
            get_traptmax_tp0est(phy_mtg_data, period, run_list)
        )

        if (
            geds_df_cuspEmax_abs is None
            or geds_df_cuspEmax_abs_corr is None
            # no need to exit if pulser01ana does not exits, handled it properly now
            # or puls_df_cuspEmax_abs is None
        ):
            utils.logger.debug("Dataframes are None for %s!", period)
            continue

        # check if geds df is empty; if pulser is, means we do not apply any correction
        # (and thus geds_corr is also empty - the code will handle the case)
        if (
            geds_df_cuspEmax_abs.empty
            # or geds_df_cuspEmax_abs_corr.empty
            # or puls_df_cuspEmax_abs.empty
        ):
            utils.logger.debug("Dataframes are empty for %s!", period)
            continue

        dfs = [
            geds_df_cuspEmax_abs,
            geds_df_cuspEmax_abs_corr,
            puls_df_cuspEmax_abs,
            geds_df_trapTmax,
            geds_df_tp0est,
            puls_df_trapTmax,
            puls_df_tp0est,
        ]

        end_folder = os.path.join(
            output_folder,
            period,
            "mtg",
        )
        os.makedirs(end_folder, exist_ok=True)
        shelve_path = os.path.join(end_folder, f"l200-{period}-phy-monitoring")
        utils.logger.debug(f"...inspecting gain over {period}")
        with shelve.open(shelve_path, "c", protocol=pickle.HIGHEST_PROTOCOL) as shelf:
            for plot_type in ["corr", "uncorr"]:
                for string, det_list in str_chns.items():
                    for channel_name in det_list:
                        channel = detectors[channel_name]["channel_str"]
                        rawid = detectors[channel_name]["daq_rawid"]
                        pos = detectors[channel_name]["position"]

                        resampling_time = "1h"  # if len(runs)>1 else "10T"

                        rawid = np.int64(rawid)
                        if rawid not in set(dfs[0].columns):
                            utils.logger.debug(
                                f"{channel} is not present in the dataframe!"
                            )
                            continue

                        pulser_data = get_pulser_data(
                            resampling_time,
                            period,
                            dfs,
                            rawid,
                            escale=escale_val,
                            variations=True,
                        )

                        fig, ax = plt.subplots(figsize=(12, 4))
                        pars_data = get_calib_pars(
                            auto_dir_path,
                            period,
                            run_list,
                            [channel, channel_name],
                            partition,
                            data_type,
                            escale=escale_val,
                            fit=fit_flag,
                        )

                        t0 = pars_data["run_start"]
                        if not eval(flag_expr):
                            # PULS01ANA has a signal - we can correct GEDS energies for it!
                            if (
                                pulser_data["pul_cusp"]["kevdiff_av"] is not None
                                and plot_type == "corr"
                            ):
                                pul_cusp_av = pulser_data["pul_cusp"][
                                    "kevdiff_av"
                                ].values.astype(float)
                                diff_av = pulser_data["diff"][
                                    "kevdiff_av"
                                ].values.astype(float)
                                diff_std = pulser_data["diff"][
                                    "kevdiff_std"
                                ].values.astype(float)
                                x = pulser_data["diff"]["kevdiff_av"].index.values

                                plt.fill_between(
                                    x,
                                    diff_av - diff_std,
                                    diff_av + diff_std,
                                    color="k",
                                    alpha=0.2,
                                    label=r"±1$\sigma$",
                                )
                                plt.plot(x, pul_cusp_av, "C2", label="PULS01ANA")
                                plt.plot(x, diff_av, "C4", label="GED corrected")
                            else:
                                ged_av = pulser_data["ged"]["kevdiff_av"].values.astype(
                                    float
                                )
                                ged_std = pulser_data["ged"][
                                    "kevdiff_std"
                                ].values.astype(float)
                                x = pulser_data["ged"]["kevdiff_av"].index.values

                                plt.fill_between(
                                    x,
                                    ged_av - ged_std,
                                    ged_av + ged_std,
                                    color="k",
                                    alpha=0.2,
                                    label=r"±1$\sigma$",
                                )
                                plt.plot(
                                    x,
                                    ged_av,
                                    color="dodgerblue",
                                    label="GED uncorrected",
                                )

                        plt.plot(
                            pars_data["run_start"] - pd.Timedelta(hours=5),
                            pars_data["fep_diff"],
                            "kx",
                            label="FEP gain",
                        )
                        plt.plot(
                            pars_data["run_start"] - pd.Timedelta(hours=5),
                            pars_data["cal_const_diff"],
                            "rx",
                            label="cal. const. diff",
                        )

                        for ti in pars_data["run_start"]:
                            plt.axvline(ti, color="dimgrey", ls="--")

                        for i in range(len(t0)):
                            if i == len(pars_data["run_start"]) - 1:
                                plt.plot(
                                    [t0[i], t0[i] + pd.Timedelta(days=7)],
                                    [pars_data["res"][i] / 2, pars_data["res"][i] / 2],
                                    "b-",
                                )
                                plt.plot(
                                    [t0[i], t0[i] + pd.Timedelta(days=7)],
                                    [
                                        -pars_data["res"][i] / 2,
                                        -pars_data["res"][i] / 2,
                                    ],
                                    "b-",
                                )
                                if quadratic:
                                    plt.plot(
                                        [t0[i], t0[i] + pd.Timedelta(days=7)],
                                        [
                                            pars_data["res_quad"][i] / 2,
                                            pars_data["res_quad"][i] / 2,
                                        ],
                                        color="dodgerblue",
                                        linestyle="-",
                                    )
                                    plt.plot(
                                        [t0[i], t0[i] + pd.Timedelta(days=7)],
                                        [
                                            -pars_data["res_quad"][i] / 2,
                                            -pars_data["res_quad"][i] / 2,
                                        ],
                                        color="dodgerblue",
                                        linestyle="-",
                                    )
                            else:
                                plt.plot(
                                    [t0[i], t0[i + 1]],
                                    [pars_data["res"][i] / 2, pars_data["res"][i] / 2],
                                    "b-",
                                )
                                plt.plot(
                                    [t0[i], t0[i + 1]],
                                    [
                                        -pars_data["res"][i] / 2,
                                        -pars_data["res"][i] / 2,
                                    ],
                                    "b-",
                                )
                                if quadratic:
                                    plt.plot(
                                        [t0[i], t0[i + 1]],
                                        [
                                            pars_data["res_quad"][i] / 2,
                                            pars_data["res_quad"][i] / 2,
                                        ],
                                        color="dodgerblue",
                                        linestyle="-",
                                    )
                                    plt.plot(
                                        [t0[i], t0[i + 1]],
                                        [
                                            -pars_data["res_quad"][i] / 2,
                                            -pars_data["res_quad"][i] / 2,
                                        ],
                                        color="dodgerblue",
                                        linestyle="-",
                                    )

                            if str(pars_data["res"][i] / 2 * 1.1) != "nan" and i < len(
                                pars_data["res"]
                            ) - (xlim_idx - 1):
                                plt.text(
                                    t0[i],
                                    pars_data["res"][i] / 2 * 1.1,
                                    "{:.2f}".format(pars_data["res"][i]),
                                    color="b",
                                )

                            if quadratic:
                                if str(
                                    pars_data["res_quad"][i] / 2 * 1.5
                                ) != "nan" and i < len(pars_data["res"]) - (
                                    xlim_idx - 1
                                ):
                                    plt.text(
                                        t0[i],
                                        pars_data["res_quad"][i] / 2 * 1.5,
                                        "{:.2f}".format(pars_data["res_quad"][i]),
                                        color="dodgerblue",
                                    )

                        fig.suptitle(
                            f"period: {period} - string: {string} - position: {pos} - ged: {channel_name}"
                        )
                        plt.ylabel(r"Energy diff / keV")
                        plt.plot(
                            [0, 1],
                            [0, 1],
                            "b",
                            label=r"Q$_{\beta\beta}$ $\pm$FWHM/2 lin. (threshold)",
                        )
                        if quadratic:
                            plt.plot(
                                [1, 2],
                                [1, 2],
                                "dodgerblue",
                                label=r"Q$_{\beta\beta}$ $\pm$FWHM/2 quad. (threshold)",
                            )

                        if zoom:
                            if flag_expr:
                                plt.ylim(-3, 3)
                            else:
                                bound = np.average(
                                    pulser_data["ged"]["cusp_av"].dropna()
                                )
                                plt.ylim(-2.5 * bound, 2.5 * bound)
                        max_date = pulser_data["ged"]["kevdiff_av"].index.max()
                        time_difference = max_date.tz_localize(None) - t0[
                            -xlim_idx
                        ].tz_localize(None)
                        plt.xlim(
                            t0[0] - pd.Timedelta(hours=8),
                            t0[-xlim_idx] + time_difference * 1.5,
                        )  # pd.Timedelta(days=7))# --> change me to resize the width of the last run
                        plt.legend(loc="lower left", title=f"Last cycle: {last_cycle}")
                        plt.tight_layout()

                        if save_pdf:
                            mgt_folder = os.path.join(end_folder, "pdf", f"st{string}")
                            os.makedirs(mgt_folder, exist_ok=True)

                            pdf_name = os.path.join(
                                mgt_folder,
                                f"{period}_string{string}_pos{pos}_{channel_name}_{plot_type}_gain_shift.pdf",
                            )
                            plt.savefig(pdf_name)

                        # serialize+save the plot
                        serialized_plot = pickle.dumps(plt.gcf())
                        shelf[
                            f"{period}_string{string}_pos{pos}_{channel_name}_{plot_type}_gain_shift"
                        ] = serialized_plot
                        plt.close(fig)

                        # structure of pickle files:
                        #  - p08_string1_pos1_V02160A_param
                        #  - p08_string1_pos2_V02160B_param
                        #  - ...

    # parameters (bsln, gain, ...) variations over run
    utils.logger.debug("...inspecting gain/bsln/etc time series")
    info = utils.MTG_PLOT_INFO
    last_checked = None

    for inspected_parameter in [
        "BlStd",
        "TrapemaxCtcCal",
        "Baseline",
        "Trapemax",
    ]:
        escale_par = escale_val if inspected_parameter == "TrapemaxCtcCal" else 1
        results.update({inspected_parameter: {}})

        for index_i in range(len(period_list)):
            period = period_list[index_i]

            (
                geds_df_cuspEmax_abs,
                geds_df_cuspEmax_abs_corr,
                puls_df_cuspEmax_abs,
            ) = get_dfs(phy_mtg_data, period, [current_run], inspected_parameter)
            geds_df_trapTmax, geds_df_tp0est, puls_df_trapTmax, puls_df_tp0est = (
                get_traptmax_tp0est(phy_mtg_data, period, [current_run])
            )

            if (
                geds_df_cuspEmax_abs is None
                or geds_df_cuspEmax_abs_corr is None
                # no need to exit if pulser01ana does not exits, handled it properly now
                # or puls_df_cuspEmax_abs is None
            ):
                utils.logger.debug(
                    "Dataframes are None for %s-%s!", period, current_run
                )
                continue
            if geds_df_cuspEmax_abs.empty:
                utils.logger.debug(
                    "Dataframes are empty for %s-%s!", period, current_run
                )
                continue
            dfs = [
                geds_df_cuspEmax_abs,
                geds_df_cuspEmax_abs_corr,
                puls_df_cuspEmax_abs,
                geds_df_trapTmax,
                geds_df_tp0est,
                puls_df_trapTmax,
                puls_df_tp0est,
            ]

            end_folder = os.path.join(
                output_folder,
                period,
                current_run,
                "mtg",
            )
            os.makedirs(end_folder, exist_ok=True)
            shelve_path = os.path.join(
                end_folder,
                f"l200-{period}-{current_run}-phy-monitoring",
            )
            utils.logger.debug(
                f"...inspecting {info[inspected_parameter]['title']} over {current_run}"
            )

            with shelve.open(
                shelve_path, "c", protocol=pickle.HIGHEST_PROTOCOL
            ) as shelf:
                for string, det_list in str_chns.items():
                    for channel_name in det_list:
                        channel = detectors[channel_name]["channel_str"]
                        rawid = detectors[channel_name]["daq_rawid"]
                        pos = detectors[channel_name]["position"]

                        resampling_time = "1h"
                        rawid = np.int64(rawid)
                        if rawid not in set(dfs[0].columns):
                            utils.logger.debug(
                                f"{channel} is not present in the dataframe!"
                            )
                            continue

                        pulser_data = get_pulser_data(
                            resampling_time,
                            period,
                            dfs,
                            rawid,
                            escale=escale_par,
                            variations=info[inspected_parameter]["percentage"],
                        )

                        fig, ax = plt.subplots(figsize=(12, 4))
                        pars_data = get_calib_pars(
                            auto_dir_path,
                            period,
                            [current_run],
                            [channel, channel_name],
                            partition,
                            data_type,
                            escale=escale_par,
                            fit=fit_flag,
                        )
                        threshold = (
                            [-pars_data["res"][0] / 2, pars_data["res"][0] / 2]
                            if "Trapemax" in inspected_parameter
                            else info[inspected_parameter]["limits"]
                        )

                        t0 = pars_data["run_start"]
                        if not eval(flag_expr):
                            check_kevdiff = None
                            if (
                                info[inspected_parameter]["percentage"] is True
                                and float(escale_par) == 1.0
                            ):
                                check_kevdiff = pulser_data["ged"]["kevdiff_av"] * 100
                            else:
                                check_kevdiff = pulser_data["ged"]["kevdiff_av"]
                            # check threshold and update YAML summary file
                            # (for energy, do it only for TrapemaxCtcCal and not Trapemax at the moment)
                            if inspected_parameter != "Trapemax":
                                utils.check_threshold(
                                    check_kevdiff,
                                    channel_name,
                                    last_checked,
                                    t0,
                                    threshold,
                                    info[inspected_parameter]["title"],
                                    output,
                                )

                            # PULS01ANA has a signal - we can correct GEDS energies for it!
                            # only in the case of energy parameters
                            if (
                                pulser_data["pul_cusp"]["kevdiff_av"] is not None
                                and inspected_parameter == "TrapemaxCtcCal"
                            ):
                                pul_cusp_av = pulser_data["pul_cusp"][
                                    "kevdiff_av"
                                ].values.astype(float)
                                diff_av = pulser_data["diff"][
                                    "kevdiff_av"
                                ].values.astype(float)
                                diff_std = pulser_data["diff"][
                                    "kevdiff_std"
                                ].values.astype(float)
                                x = pulser_data["diff"]["kevdiff_av"].index.values

                                plt.plot(x, pul_cusp_av, "C2", label="PULS01ANA")
                                plt.plot(x, diff_av, "C4", label="GED corrected")
                                plt.fill_between(
                                    x,
                                    diff_av - diff_std,
                                    diff_av + diff_std,
                                    color="k",
                                    alpha=0.2,
                                    label=r"±1$\sigma$",
                                )

                                results[inspected_parameter].update(
                                    {channel_name: pul_cusp_av.values.astype(float)}
                                )
                            # else, no correction is applied
                            else:
                                if (
                                    info[inspected_parameter]["percentage"] is True
                                    and float(escale_par) == 1.0
                                ):
                                    pulser_data["ged"]["kevdiff_av"] *= 100
                                    pulser_data["ged"]["kevdiff_std"] *= 100

                                vals_av = pulser_data["ged"][
                                    "kevdiff_av"
                                ].values.astype(float)
                                vals_std = pulser_data["ged"][
                                    "kevdiff_std"
                                ].values.astype(float)
                                x = pulser_data["ged"]["kevdiff_av"].index.values

                                plt.plot(
                                    x,
                                    vals_av,
                                    color=info[inspected_parameter]["colors"][0],
                                    label="GED uncorrected",
                                )
                                plt.fill_between(
                                    x,
                                    vals_av - vals_std,
                                    vals_av + vals_std,
                                    color="k",
                                    alpha=0.2,
                                    label=r"±1$\sigma$",
                                )

                                results[inspected_parameter].update(
                                    {
                                        channel_name: pulser_data["ged"][
                                            "kevdiff_av"
                                        ].values.astype(float)
                                    }
                                )

                        # plot resolution only for the energy parameters
                        if inspected_parameter == "TrapemaxCtcCal":
                            plt.plot(
                                [t0[0], t0[0] + pd.Timedelta(days=7)],
                                [pars_data["res"][0] / 2, pars_data["res"][0] / 2],
                                color=info[inspected_parameter]["colors"][1],
                                ls="-",
                            )
                            plt.plot(
                                [t0[0], t0[0] + pd.Timedelta(days=7)],
                                [-pars_data["res"][0] / 2, -pars_data["res"][0] / 2],
                                color=info[inspected_parameter]["colors"][1],
                                ls="-",
                            )

                            if str(pars_data["res"][0] / 2 * 1.1) != "nan" and 0 < len(
                                pars_data["res"]
                            ) - (xlim_idx - 1):
                                plt.text(
                                    t0[0],
                                    pars_data["res"][0] / 2 * 1.1,
                                    "{:.2f}".format(pars_data["res"][0]),
                                    color=info[inspected_parameter]["colors"][1],
                                )
                            plt.plot(
                                [0, 1],
                                [0, 1],
                                color=info[inspected_parameter]["colors"][1],
                                label=r"Q$_{\beta\beta}$ $\pm$FWHM/2 lin. (threshold)",
                            )
                        else:
                            if threshold[1] is not None:
                                plt.plot(
                                    [t0[0], t0[0] + pd.Timedelta(days=7)],
                                    [
                                        threshold[1],
                                        threshold[1],
                                    ],
                                    color=info[inspected_parameter]["colors"][1],
                                    ls="-",
                                    label="Threshold",
                                )
                            if threshold[0] is not None:
                                plt.plot(
                                    [t0[0], t0[0] + pd.Timedelta(days=7)],
                                    [
                                        threshold[0],
                                        threshold[0],
                                    ],
                                    color=info[inspected_parameter]["colors"][1],
                                    ls="-",
                                    label=(
                                        None
                                        if threshold[1] is not None
                                        else "Threshold"
                                    ),
                                )

                        plt.ylabel(info[inspected_parameter]["ylabel"])
                        fig.suptitle(
                            f"period: {period} - run: {current_run} - string: {string} - position: {pos} - ged: {channel_name}"
                        )

                        if zoom is True:
                            bound = np.average(
                                pulser_data["ged"]["kevdiff_std"].dropna()
                            )
                            plt.ylim(-3.5 * bound, 3.5 * bound)

                        max_date = pulser_data["ged"]["kevdiff_av"].index.max()
                        time_difference = max_date.tz_localize(None) - t0[
                            -xlim_idx
                        ].tz_localize(None)
                        plt.xlim(
                            t0[0] - pd.Timedelta(hours=0.5),
                            t0[-xlim_idx] + time_difference * 1.1,
                        )
                        plt.legend(loc="lower left", title=f"Last cycle: {last_cycle}")
                        plt.tight_layout()

                        if save_pdf:
                            mgt_folder = os.path.join(end_folder, "pdf", f"st{string}")
                            os.makedirs(mgt_folder, exist_ok=True)

                            pdf_name = os.path.join(
                                mgt_folder,
                                f"{period}_{current_run}_string{string}_pos{pos}_{channel_name}_{info[inspected_parameter]['title']}.pdf",
                            )
                            plt.savefig(pdf_name)

                        # serialize+save the plot
                        serialized_plot = pickle.dumps(plt.gcf())
                        shelf[
                            f"{period}_{current_run}_string{string}_pos{pos}_{channel_name}_{info[inspected_parameter]['title']}"
                        ] = serialized_plot
                        plt.close(fig)

    with open(usability_map_file, "w") as f:
        yaml.dump(output, f)

    return results
