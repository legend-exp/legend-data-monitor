import glob
import itertools
import json
import math
import os
import pickle
import shelve
import sys
from functools import partial

import awkward as ak
import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import yaml
from lgdo import lh5
from lgdo.lh5 import read_as
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
    det_info: dict,
    save_pdf: bool,
):
    pars_to_inspect = [
        "IsValidBlSlopeClassifier",
        "IsValidTailRmsClassifier",
        "IsValidPzSlopeClassifier",
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

            df_All = filter_series_by_ignore_keys(df_All, utils.IGNORE_KEYS, period)
            df_IsPulser = filter_series_by_ignore_keys(
                df_IsPulser, utils.IGNORE_KEYS, period
            )
            df_IsBsln = filter_series_by_ignore_keys(
                df_IsBsln, utils.IGNORE_KEYS, period
            )
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
                    if not det_info["detectors"][det]["processable"]:
                        continue

                    ax = axes[i]
                    ch = det_info["detectors"][det]["daq_rawid"]
                    vals_all = df_All[ch].values
                    vals_pulser = df_IsPulser[ch].values
                    vals_bsln = df_IsBsln[ch].values
                    vals_phys = df_IsPhysics[ch].values

                    vals_all = vals_all[~np.isnan(vals_all)]
                    vals_pulser = vals_pulser[~np.isnan(vals_pulser)]
                    vals_bsln = vals_bsln[~np.isnan(vals_bsln)]
                    vals_phys = vals_phys[~np.isnan(vals_phys)]

                    # global bins
                    bins = np.arange(-15, 15 + step, step)

                    # percentages
                    perc_all = 100 * np.mean((vals_all >= -5) & (vals_all <= 5))
                    perc_pulser = 100 * np.mean(
                        (vals_pulser >= -5) & (vals_pulser <= 5)
                    )
                    perc_bsln = 100 * np.mean((vals_bsln >= -5) & (vals_bsln <= 5))
                    perc_phys = 100 * np.mean((vals_phys >= -5) & (vals_phys <= 5))

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

                fig.suptitle(f"{period} - {run} - string {string} - {par}")
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
                hourly_counts = data.resample("1H").sum()

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
        field_mask=["is_bb_like", "is_bb_like_old", "is_good_channel"],
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
    daily_cnt = df.resample("H").sum()

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
    total_forced = df_all.resample("H").size()  # counts/hour, all strings
    avg_total_forced_mhz = (total_forced.mean() / 3600) * 1000
    on_mass = 0

    # ONE PERIOD, ALL RUNS
    with shelve.open(shelve_path, "c", protocol=pickle.HIGHEST_PROTOCOL) as shelf:
        # --- Per-string plots ---
        for string, det_list in det_info["str_chns"].items():
            fig, ax = plt.subplots(figsize=(12, 6))
            string_sum = None

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

                string_sum = (
                    hourly_rate if string_sum is None else string_sum + hourly_rate
                )

            str_counts[string] = string_sum

            m2p = partial(mhz_to_percent, avg_total_forced_mhz=avg_total_forced_mhz)
            p2m = partial(percent_to_mhz, avg_total_forced_mhz=avg_total_forced_mhz)
            secax = ax.secondary_yaxis("right", functions=(m2p, p2m))
            secax.set_ylabel("FT failure fraction (%)")

            ax.set_ylabel("Normalized FT failure rate (mHz/kg)")
            ax.legend(ncol=2, fontsize="small", loc="upper left")
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
        ax.legend(ncol=2, fontsize="small", loc="upper left")
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
        total_forced = df_all.resample("H").sum()["count"]
        surviving = df_survived.resample("H").sum()["count"]
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

        mask2 = (
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
        ser_pass = pd.to_datetime(forced.timestamp[mask2 & is_bb.is_bb_like], unit="s")
        ser_fail = pd.to_datetime(forced.timestamp[mask2 & ~is_bb.is_bb_like], unit="s")

        for s, label, color in [
            (ser, "All events", "dimgrey"),
            (ser_dis, "Delayed discharges", "darkorange"),
            (ser_fail, "Failing QC", "crimson"),
            (ser_pass, "Surviving QC", "dodgerblue"),
        ]:
            if s.empty:
                continue
            freq, bin_edges = np.histogram(
                s, bins=pd.date_range(start=s.min(), end=s.max(), freq="H")
            )
            ax.stairs(freq / 3600 * 1000 / on_mass, bin_edges, label=label, color=color)

        ax.set_ylabel("Hourly rate normalized by ON mass (mHz/kg)")
        ax.legend(title=f"ON mass = {on_mass:.1f} kg", loc="upper right")
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


def box_summary_plot(
    period: str,
    run: str,
    pars: dict,
    det_info: dict,
    results: dict,
    info: dict,
    output_dir: str,
    data_type: str,
    save_pdf: bool,
    run_to_apply=None,
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
            }
        )

    df_plot = pd.DataFrame(plot_data)
    # sort by string, and then position
    df = df_plot.sort_values(["string", "pos"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    if not df["fwhm"].isna().all():
        ax.bar(
            x,
            df["fwhm"],
            bottom=-df["fwhm"] / 2,
            width=0.4,
            color="orange",
            alpha=0.2,
            label="FWHM",
        )

    ax.bar(
        x,
        2 * df["std"],  # total height = twice 1 std
        bottom=df["mean"] - df["std"],  # center bar on mean
        width=0.6,
        color="skyblue",
        alpha=0.7,
        label="±1σ",
    )

    ax.scatter(x, df["mean"], color="black", zorder=3, label="Mean")

    ax.errorbar(
        x,
        df["mean"],
        yerr=[df["mean"] - df["min"], df["max"] - df["mean"]],
        fmt="none",
        ecolor="#0266c9" if info["title"] != "FEP_gain" else "red",
        capsize=4,
        label="Min/Max",
    )

    ax.set_xticks(x)
    xtick_labels = ax.set_xticklabels(df["ged"], rotation=90)
    for i, label in enumerate(xtick_labels):
        if df.iloc[i]["usability"] in ["off", "false", False]:
            label.set_color("red")
        if df.iloc[i]["usability"] in ["ac"]:
            label.set_color("darkorange")

    ax.axvline(-0.5, color="gray", ls="--", alpha=0.5)
    ymin, ymax = ax.get_ylim()
    label_y = ymin * (ymax / ymin) ** 0.05 if ymin > 0 else -4
    label_y = label_y if info["title"] != "baseln_spike" else 1
    unique_strings = df["string"].unique()
    for s in unique_strings:
        idx = df.index[df["string"] == s]
        left, right = idx.min(), idx.max()
        ax.axvline(right + 0.5, color="gray", ls="--", alpha=0.5)
        ax.text(left, label_y, f"String {s}", rotation=90)

    ax.set_ylabel(info["ylabel"])
    ax.set_title(f"{period} {run}")

    # Create custom legend entries for usability colors
    legend_patches = []
    # Get the current handles and labels
    handles, labels = ax.get_legend_handles_labels()
    # Add custom patches for usability colors
    legend_patches.append(Patch(color="red", label="Usability: off"))
    legend_patches.append(Patch(color="darkorange", label="Usability: ac"))
    # Combine existing handles with new patches
    all_handles = handles + legend_patches
    # Set the legend with all handles
    ax.legend(handles=all_handles, loc="upper right")
    ax.grid(False)

    if info["title"] in ["baseln_stab"]:
        ax.axhline(-10, ls="--", color="black")
        ax.axhline(10, ls="--", color="black")
        ax.axhspan(10, 500, color="gray", alpha=0.25)
        ax.axhspan(-10, -500, color="gray", alpha=0.25)
    if info["title"] in ["baseln_spike"]:
        ax.axhline(50, ls="--", color="black")
        ax.axhspan(50, 500, color="gray", alpha=0.25)

    if info["title"] in ["FEP_gain", "pulser_stab"]:
        plt.ylim(-6, 6)
    if info["title"] in ["baseln_stab"]:
        plt.ylim(-20, 20)
    if info["title"] in ["baseln_spike"]:
        plt.ylim(0, 100)

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


def compute_dead_time(df, window_ms=10):
    """
    Compute dead time percentage based on discharge windows.

    Parameters
    ----------
    df : pd.DataFrame
        Timestamps and boolean detector columns with is_discharge entries.
    window_ms : float
        Dead time window after each discharge; default: 10 ms.
    """
    times = df.index.view("int64") / 1e9
    dt_total = times[-1] - times[0]

    discharge_times = times[df.any(axis=1).to_numpy()]
    if len(discharge_times) == 0:
        return 0.0

    lost_time = 0.0
    window = window_ms / 1000.0
    next_available = -np.inf

    for t in discharge_times:
        if t >= next_available:
            lost_time += window
            next_available = t + window  # veto until then

    lost_time = len(discharge_times) * (window_ms / 1000.0)
    return lost_time / dt_total * 100


def qc_average(
    auto_dir_path: str,
    output_folder: str,
    det_info: dict,
    period: str,
    run: str,
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

                string_indices[string] = indices

            ax.scatter(xs, ys, color="dodgerblue", marker="o")
            ax.set_title(f"period: {period} - run: {run} - passing {par}")
            # if par == 'IsDischarge':
            #    dt = compute_dead_time(geds_df_abs)
            #    ax.set_title(f"period: {period} - run: {run} - passing {par} - tot dead time {dt:.3f}%")
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

            plt.tight_layout()
            if save_pdf:
                pdf_dir = os.path.join(end_folder, "pdf")
                os.makedirs(pdf_dir, exist_ok=True)
                pdf_name = os.path.join(pdf_dir, f"{period}_{run}_{par}_avg.pdf")
                fig.savefig(pdf_name)

            # serialize+save plot
            shelf[f"{period}_{run}_{par}_avg"] = pickle.dumps(fig)
            plt.close(fig)


def qc_time_series(
    auto_dir_path: str,
    output_folder: str,
    det_info: dict,
    period: str,
    run: str,
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
                    hourly_rate = data.resample("1H").sum() / 3600 * 1000

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
                ax.legend(loc="lower left")
                plt.tight_layout()

                if save_pdf:
                    pdf_dir = os.path.join(end_folder, "pdf", f"st{string}")
                    os.makedirs(pdf_dir, exist_ok=True)
                    pdf_name = os.path.join(
                        pdf_dir, f"{period}_{run}_string{string}_{par}_rate.pdf"
                    )
                    fig.savefig(pdf_name)

                # serialize+save plot
                shelf[f"{period}_{run}_string{string}_{par}_rate"] = pickle.dumps(fig)
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

    # for when we have a calib run but zero phy runs for a given period
    if os.path.isdir(dir_path) and run not in os.listdir(dir_path):
        run_files = sorted(os.listdir(folder_tier))
        run_end_time = pd.to_datetime(
            sto.read(
                "ch1027201/dsp/timestamp", os.path.join(folder_tier, run_files[-1])
            )[-1],
            unit="s",
        )
        run_start_time = run_end_time
    else:
        run_files = sorted(os.listdir(folder_tier))
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
    mean[~mask] = std[~mask] = np.nan

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
    ser_pul_tp0est_new = pd.DataFrame()

    if ser_ged_cusp.empty:
        utils.logger.debug("...geds series is empty after filtering")
        return None

    # check if these dfs are empty or not - if not, then remove spikes
    if isinstance(dfs[6], pd.DataFrame) and not dfs[6].empty:
        ser_pul_tp0est = dfs[6][1027203].sort_index()
        ser_pul_tp0est = filter_by_period(ser_pul_tp0est, period)

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

    # pulser series
    ser_pul_cusp = ser_pul_cuspdiff = ser_pul_cuspdiff_kev = pul_cusp_hr_av = (
        pul_cusp_hr_std
    ) = None
    ged_cusp_corr = ged_cusp_corr_kev = ged_cusp_cor_hr_av = ged_cusp_cor_hr_std = None
    # ...if pulser is available:
    if not dfs[2].empty:
        ser_pul_cusp = dfs[2][1027203].sort_index()
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

            # corrected GED
            common_index = ser_ged_cuspdiff.index.intersection(ser_pul_cuspdiff.index)
            ged_cusp_corr = (
                ser_ged_cuspdiff[common_index] - ser_pul_cuspdiff[common_index]
            )
            ged_cusp_corr_kev = ged_cusp_corr * escale
            ged_cusp_cor_hr_av, ged_cusp_cor_hr_std = resample_series(
                ged_cusp_corr_kev, resampling_time, mask
            )

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
    utils.logger.debug("...inspecting gain/bsln/etc time series")

    # gain over period
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
        utils.logger.debug(f"...inspecting Gain over {period}")
        with shelve.open(shelve_path, "c", protocol=pickle.HIGHEST_PROTOCOL) as shelf:
            for plot_type in ["corr", "uncorr"]:
                for string, det_list in str_chns.items():
                    for channel_name in det_list:
                        channel = detectors[channel_name]["channel_str"]
                        rawid = detectors[channel_name]["daq_rawid"]
                        pos = detectors[channel_name]["position"]

                        resampling_time = "1h"  # if len(runs)>1 else "10T"
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
                            if plot_type == "uncorr":
                                kevdiff = pulser_data["ged"]["kevdiff_av"]
                            else:
                                kevdiff = (
                                    pulser_data["ged"]["kevdiff_av"]
                                    if pulser_data["diff"]["kevdiff_av"] is None
                                    else pulser_data["diff"]["kevdiff_av"]
                                )

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
                        plt.plot([0, 1], [0, 1], "b", label="Qbb FWHM keV lin.")
                        if quadratic:
                            plt.plot(
                                [1, 2],
                                [1, 2],
                                "dodgerblue",
                                label="Qbb FWHM keV quadr.",
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
                        plt.legend(loc="lower left")
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
    info = utils.MTG_PLOT_INFO
    results = {}

    for inspected_parameter in ["Baseline", "Trapemax", "TrapemaxCtcCal", "BlStd"]:
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
                f"...inspecting {inspected_parameter} over {current_run}"
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
                            [pars_data["res"][0], pars_data["res"][0]]
                            if inspected_parameter == "TrapemaxCtcCal"
                            else info[inspected_parameter]["limits"]
                        )

                        t0 = pars_data["run_start"]
                        if not eval(flag_expr):
                            kevdiff = (
                                pulser_data["ged"]["kevdiff_av"]
                                if pulser_data["diff"]["kevdiff_av"] is None
                                else pulser_data["diff"]["kevdiff_av"]
                            )

                            # check threshold and update YAML summary file
                            utils.check_threshold(
                                kevdiff,
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
                                label="Qbb FWHM keV lin.",
                            )
                        else:
                            if info[inspected_parameter]["limits"][1] is not None:
                                plt.plot(
                                    [t0[0], t0[0] + pd.Timedelta(days=7)],
                                    [
                                        info[inspected_parameter]["limits"][1],
                                        info[inspected_parameter]["limits"][1],
                                    ],
                                    color=info[inspected_parameter]["colors"][1],
                                    ls="-",
                                )
                            if info[inspected_parameter]["limits"][0] is not None:
                                plt.plot(
                                    [t0[0], t0[0] + pd.Timedelta(days=7)],
                                    [
                                        info[inspected_parameter]["limits"][0],
                                        info[inspected_parameter]["limits"][0],
                                    ],
                                    color=info[inspected_parameter]["colors"][1],
                                    ls="-",
                                )

                        plt.ylabel(info[inspected_parameter]["ylabel"])
                        fig.suptitle(
                            f"period: {period} - string: {string} - position: {pos} - ged: {channel_name}"
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
                        plt.legend(loc="lower left")
                        plt.tight_layout()

                        if save_pdf:
                            mgt_folder = os.path.join(end_folder, "pdf", f"st{string}")
                            os.makedirs(mgt_folder, exist_ok=True)

                            pdf_name = os.path.join(
                                mgt_folder,
                                f"{period}_{current_run}_string{string}_pos{pos}_{channel_name}_{inspected_parameter}.pdf",
                            )
                            plt.savefig(pdf_name)

                        # serialize+save the plot
                        serialized_plot = pickle.dumps(plt.gcf())
                        shelf[
                            f"{period}_{current_run}_string{string}_pos{pos}_{channel_name}_{inspected_parameter}"
                        ] = serialized_plot
                        plt.close(fig)

    with open(usability_map_file, "w") as f:
        yaml.dump(output, f)

    return results
