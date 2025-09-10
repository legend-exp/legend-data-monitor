import glob
import os
import pickle
import shelve

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from lgdo import lh5

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

matplotlib.rcParams["mathtext.fontset"] = "stix"

plt.rc("axes", facecolor="white", edgecolor="black", axisbelow=True, grid=True)

# -------------------------------------------------------------------------


def load_fit_pars_from_yaml(
    pars_files_list: list, detectors_list: list, detectors_name: list, avail_runs: list
):
    """
    Load detector data from YAML files and return directly as a dict.

    Parameters
    ----------
    pars_files_list : list
        List of file paths to YAML parameter files.
    detectors_list : list
        List of detector raw IDs (eg. 'ch1104000') to extract data for.
    detectors_name : list
        List of detector names (eg. 'V11925A') to extract data for.
    avail_runs : list or None
        Available runs to inspect (e.g. [4, 5, 6]); if None, keep all.

    Returns
    -------
    dict
        {
          "V11925A": {
              "r004": {"mean": ..., "mean_err": ..., "sigma": ..., "sigma_err": ...},
              "r005": {...},
              ...
          },
          "V11925B": {
              "r004": {...},
              ...
          }
        }
    """
    results = {}

    for file_path in pars_files_list:
        run_idx = int(file_path.split("/")[-2].split("r")[-1])
        run_str = f"r{run_idx:03d}"
        if run_str not in avail_runs:
            continue

        run_data = utils.read_json_or_yaml(file_path)
        time = 0 if "par_hit" in file_path else file_path.split("-")[-2]

        for idx, det in enumerate(detectors_list):
            det_key = det if det in run_data else detectors_name[idx]

            pars = utils.deep_get(
                run_data or {}, [det_key, "results", "aoe", "1000-1300keV", time], {}
            )

            results.setdefault(detectors_name[idx], {})[run_str] = {
                "mean": pars.get("mean"),
                "mean_err": pars.get("mean_err"),
                "sigma": pars.get("sigma"),
                "sigma_err": pars.get("sigma_err"),
            }

    return results or None


def evaluate_psd_performance(
    mean_vals: list, sigma_vals: list, run_labels: list, current_run: str, det_name: str
):
    """Evaluate PSD performance metrics: slow shifts and sudden shifts and return a dict with evaluation results."""
    results = {}

    # check prerequisites
    valid_idx = next((i for i, v in enumerate(mean_vals) if not np.isnan(v)), None)

    # handle case where all sigma_vals are NaN
    if all(np.isnan(sigma_vals)):
        sigma_avg = np.nan
    else:
        sigma_avg = np.nanmean(sigma_vals)

    if valid_idx is None or np.isnan(sigma_avg) or sigma_avg == 0:
        results["status"] = None
        results["slow_shift_fail_runs"] = []
        results["sudden_shift_fail_runs"] = []
        results["slow_shifts"] = []
        results["sudden_shifts"] = []
        return results

    # SLOW shifts
    slow_shifts = [(v - mean_vals[valid_idx]) / sigma_avg for v in mean_vals]
    slow_shift_fail_runs = [
        run_labels[i]
        for i, z in enumerate(slow_shifts)
        if abs(z) > 0.5 and run_labels[i] == current_run
    ]
    slow_shift_failed = bool(slow_shift_fail_runs)

    # SUDDEN shifts
    sudden_shifts = []
    for i in range(len(mean_vals) - 1):
        v1, v2, s = mean_vals[i], mean_vals[i + 1], sigma_vals[i]
        if np.isnan(v1) or np.isnan(v2) or np.isnan(s) or s == 0:
            sudden_shifts.append(np.nan)
        else:
            sudden_shifts.append(abs(v2 - v1) / s)

    sudden_shift_fail_runs = [
        f"{run_labels[i]}TO{run_labels[i+1]}"
        for i, z in enumerate(sudden_shifts)
        if not np.isnan(z) and z > 0.25 and run_labels[i + 1] == current_run
    ]
    sudden_shift_failed = bool(sudden_shift_fail_runs)

    status = False
    if not slow_shift_failed and not sudden_shift_failed:
        status = True

    results["status"] = status
    results["slow_shift_fail_runs"] = slow_shift_fail_runs
    results["sudden_shift_fail_runs"] = sudden_shift_fail_runs
    results["slow_shifts"] = slow_shifts
    results["sudden_shifts"] = sudden_shifts

    return results


def update_psd_evaluation_in_memory(
    data: dict, det_name: str, data_type: str, key: str, value: bool | float
):
    """Update the key entry in memory dict, where value can be bool or nan if not available; data_type is either 'cal' or 'phy'."""
    data.setdefault(det_name, {}).setdefault(data_type, {})[key] = value


def evaluate_psd_usability_and_plot(
    period: str,
    current_run: str,
    fit_results_cal: dict,
    det_name: str,
    location,
    output_dir: str,
    psd_data: dict,
    save_pdf: bool,
):
    """Plot PSD stability results across runs, evaluate performance, and save both plot and evaluation summary."""
    run_labels = sorted(fit_results_cal.keys())
    run_positions = list(range(len(run_labels)))

    # extract values
    mean_vals = utils.none_to_nan([fit_results_cal[r]["mean"] for r in run_labels])
    mean_errs = utils.none_to_nan([fit_results_cal[r]["mean_err"] for r in run_labels])
    sigma_vals = utils.none_to_nan([fit_results_cal[r]["sigma"] for r in run_labels])
    sigma_errs = utils.none_to_nan(
        [fit_results_cal[r]["sigma_err"] for r in run_labels]
    )

    # Evaluate performance
    eval_result = evaluate_psd_performance(
        mean_vals, sigma_vals, run_labels, current_run, det_name
    )
    # if all nan entries, comment and exit
    if eval_result["status"] is None:
        return

    fig, axs = plt.subplots(2, 2, figsize=(15, 9), sharex=True)
    (ax1, ax3), (ax2, ax4) = axs

    # mean stability
    mean_avg, mean_std = np.nanmean(mean_vals), np.nanstd(mean_vals)
    ax1.errorbar(
        run_positions,
        mean_vals,
        yerr=mean_errs,
        fmt="s",
        color="blue",
        capsize=4,
        label=r"$\mu_i$",
    )
    ax1.axhline(
        mean_avg,
        linestyle="--",
        color="steelblue",
        label=rf"$\bar{{\mu}} = {mean_avg:.5f}$",
    )
    ax1.fill_between(
        run_positions,
        mean_avg - mean_std,
        mean_avg + mean_std,
        color="steelblue",
        alpha=0.2,
        label="±1 std dev",
    )
    ax1.set_ylabel("Mean stability")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)

    # Sigma stability
    sigma_avg, sigma_std = np.nanmean(sigma_vals), np.nanstd(sigma_vals)
    ax2.errorbar(
        run_positions,
        sigma_vals,
        yerr=sigma_errs,
        fmt="s",
        color="darkorange",
        capsize=4,
        label=r"$\sigma_i$",
    )
    ax2.axhline(
        sigma_avg,
        linestyle="--",
        color="peru",
        label=rf"$\bar{{\sigma}} = {sigma_avg:.5f}$",
    )
    ax2.fill_between(
        run_positions,
        sigma_avg - sigma_std,
        sigma_avg + sigma_std,
        color="peru",
        alpha=0.2,
        label="±1 std dev",
    )
    ax2.set_ylabel("Sigma stability")
    ax2.set_xlabel("Run")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)

    # slow shifts
    ax3.plot(
        run_positions,
        eval_result["slow_shifts"],
        marker="^",
        linestyle="-",
        color="darkorchid",
        label="Slow shifts",
    )
    ax3.axhline(0, color="black", linestyle="--")
    ax3.axhline(0.5, color="crimson", linestyle="--")
    ax3.axhline(-0.5, color="crimson", linestyle="--")
    ax3.set_ylabel(r"$(\mu_i - \mu_0)/\bar{\sigma}$")
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc="upper left", bbox_to_anchor=(0, 0.95), fontsize=12)

    # sudden shifts
    x = np.arange(len(eval_result["sudden_shifts"]))
    y = np.array(eval_result["sudden_shifts"])
    mask = ~np.isnan(y)
    ax4.plot(
        x[mask] + 1,
        y[mask],
        marker="^",
        linestyle="-",
        color="green",
        label="Sudden shifts",
    )
    ax4.axhline(0, color="black", linestyle="--")
    ax4.axhline(0.25, color="crimson", linestyle="--")
    ax4.set_ylabel(r"$|(\mu_{i+1}-\mu_i)/\sigma_i|$")
    ax4.set_xlabel("Run")
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc="upper left", bbox_to_anchor=(0, 0.95), fontsize=12)

    for ax in axs.flatten():
        ax.set_xticks(run_positions)
        ax.set_xticklabels(run_labels, rotation=0)

    fig.suptitle(det_name, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_dir = os.path.join(output_dir, "mtg")
    if save_pdf:
        pdf_folder = os.path.join(output_dir, "pdf", f"st{location[0]}")
        os.makedirs(pdf_folder, exist_ok=True)
        plt.savefig(
            os.path.join(
                pdf_folder,
                f"{period}_string{location[0]}_pos{location[1]}_{det_name}_PSDusability.pdf",
            ),
            bbox_inches="tight",
        )

    # store the serialized plot in a shelve object under key
    serialized_plot = pickle.dumps(plt.gcf())
    with shelve.open(
        os.path.join(
            output_dir,
            f"l200-{period}-cal-monitoring",
        ),
        "c",
        protocol=pickle.HIGHEST_PROTOCOL,
    ) as shelf:
        shelf[
            f"{period}_string{location[0]}_pos{location[1]}_{det_name}_PSDusability"
        ] = serialized_plot

    plt.close()

    # supdate psd status
    update_psd_evaluation_in_memory(
        psd_data, det_name, "cal", "PSD", eval_result["status"]
    )


def check_psd(
    auto_dir_path: str, cal_path:str, pars_files_list: list, output_dir: str, period: str, current_run: str, det_info: dict, save_pdf: bool
):

    # create the folder and parents if missing - for the moment, we store it under the 'phy' folder
    output_dir = os.path.join(output_dir, period)
    output_dir_run = os.path.join(output_dir, current_run, "mtg")
    os.makedirs(output_dir_run, exist_ok=True)

    # Load existing data once (or start empty)
    usability_map_file = os.path.join(
        output_dir_run, f"l200-{period}-{current_run}-qcp_summary.yaml"
    )

    if os.path.exists(usability_map_file):
        with open(usability_map_file) as f:
            psd_data = yaml.safe_load(f) or {}
    else:
        psd_data = {}

    detectors_name = list(det_info["detectors"].keys())
    detectors_list = [det_info["detectors"][d]["channel_str"] for d in detectors_name]
    locations_list = [
        (det_info["detectors"][d]["position"], det_info["detectors"][d]["string"])
        for d in detectors_name
    ]

    cal_runs = sorted(os.listdir(cal_path))
    if len(cal_runs) == 1:
        utils.logger.debug(
            "Only one available calibration run. Save all entries as None and exit."
        )
        for det_name in detectors_name:
            update_psd_evaluation_in_memory(psd_data, det_name, "cal", "PSD", None)

        with open(usability_map_file, "w") as f:
            yaml.dump(psd_data, f, sort_keys=False)

        return

    # retrieve all dets info
    cal_psd_info = load_fit_pars_from_yaml(
        pars_files_list, detectors_list, detectors_name, cal_runs
    )
    if cal_psd_info is None:
        utils.logger.debug("...no data are available at the moment")
        return

    # inspect one single det: plot+saving
    for idx, det_name in enumerate(detectors_name):
        evaluate_psd_usability_and_plot(
            period,
            current_run,
            cal_psd_info[det_name],
            det_name,
            locations_list[idx],
            output_dir,
            psd_data,
            save_pdf,
        )

    with open(usability_map_file, "w") as f:
        yaml.dump(psd_data, f, sort_keys=False)


def fep_gain_variation(
    period: str,
    run: str,
    pars,
    chmap: dict,
    timestamps,
    values,
    output_dir: str,
    save_pdf: bool,
):

    ged = chmap["name"]
    string = chmap["string"]
    position = chmap["position"]

    bin_size = 600
    bins = np.arange(0, timestamps.max() + bin_size, bin_size)

    bin_idx = np.digitize(timestamps, bins) - 1  # shift to 0-based

    df = pd.DataFrame({"time": timestamps, "value": values, "bin": bin_idx})

    stats = df.groupby("bin")["value"].agg(["mean", "std", "count"]).reset_index()
    stats["time"] = bins[stats["bin"]] + bin_size / 2

    min_counts = 20
    stats.loc[stats["count"] < min_counts, ["mean", "std"]] = np.nan

    # Choose baseline: first mean if valid, otherwise last valid mean
    if pd.notna(stats["mean"].iloc[0]):
        baseline = stats["mean"].iloc[0]
    else:
        baseline = stats["mean"].dropna().iloc[-1]

    norm_values = (values - baseline) / baseline * 2039

    x_bins = bins
    y_bins = np.linspace(-10, 10, 40)
    means = (stats["mean"] - baseline) / baseline * 2039

    plt.figure(figsize=(10, 5))
    plt.hist2d(timestamps, norm_values, bins=(x_bins, y_bins), cmap="Blues")
    plt.colorbar(label="Counts")

    plt.plot(stats["time"], means, "x-", color="red", label="10min mean")

    plt.fill_between(
        stats["time"],
        -stats["std"] / baseline * 2039,
        stats["std"] / baseline * 2039,
        color="red",
        alpha=0.2,
        label="±1 std",
    )

    fwhm = pars["results"]["ecal"]["cuspEmax_ctc_cal"]["eres_linear"]["Qbb_fwhm_in_kev"]

    if fwhm < 5:
        plt.ylim(-5, 5)

    plt.axhline(0, ls="--", color="black")
    plt.axhline(-fwhm / 2, ls="-", color="blue")
    plt.axhline(fwhm / 2, ls="-", color="blue", label="±FWHM/2")
    plt.text(0, fwhm / 2 + 0.5, f"+FWHM/2 = {fwhm/2:.2f} keV", color="k")

    plt.legend(loc="lower left")
    plt.xlabel("time [s]")
    plt.ylabel("FEP gain variation [keV]")
    plt.title(f"{period} {run} string {string} position {position} {ged}")
    plt.tight_layout()

    if save_pdf:
        pdf_folder = os.path.join(output_dir, f"{period}/{run}/mtg/pdf", f"st{string}")
        os.makedirs(pdf_folder, exist_ok=True)
        plt.savefig(
            os.path.join(
                pdf_folder,
                f"{period}_{run}_str{string}_pos{position}_{ged}_FEP_gain_variation.pdf",
            ),
            bbox_inches="tight",
        )

    # store the serialized plot in a shelve object under key
    serialized_plot = pickle.dumps(plt.gcf())
    with shelve.open(
        os.path.join(
            output_dir,
            f"{period}/{run}/mtg/l200-{period}-{run}-cal-monitoring",
        ),
        "c",
        protocol=pickle.HIGHEST_PROTOCOL,
    ) as shelf:
        shelf[f"{period}_{run}_str{string}_pos{position}_{ged}_FEP_gain_variation"] = (
            serialized_plot
        )

    plt.close()

    return means


def fep_gain_variation_summary(
    period: str,
    run: str,
    pars,
    detectors: dict,
    results,
    output_dir: str,
    save_pdf: bool,
):

    plot_data = []
    for ged, item in results.items():

        fwhm = pars[ged]["results"]["ecal"]["cuspEmax_ctc_cal"]["eres_linear"][
            "Qbb_fwhm_in_kev"
        ]
        meta_info = detectors[ged]
        plot_data.append(
            {
                "ged": ged,
                "string": meta_info["string"],
                "pos": meta_info["position"],
                "mean": item.mean(),
                "std": item.std(),
                "min": item.min(),
                "max": item.max(),
                "fwhm": fwhm,
            }
        )

    df_plot = pd.DataFrame(plot_data)

    # Sort by string then position
    df = df_plot.sort_values(["string", "pos"]).reset_index(drop=True)

    # Assuming df is already built and sorted
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(df))

    # Plot bars for std around the mean
    ax.bar(
        x,
        2 * df["std"],  # total height = 2σ
        bottom=df["mean"] - df["std"],  # center bar on mean
        width=0.6,
        color="skyblue",
        alpha=0.7,
        label="±1σ",
    )

    ax.bar(
        x,
        df["fwhm"],
        bottom=-df["fwhm"] / 2,
        width=0.4,
        color="orange",
        alpha=0.2,
        label="FWHM",
    )

    # Overlay mean as a point
    ax.scatter(x, df["mean"], color="black", zorder=3, label="Mean")

    # Overlay min/max as error bars
    ax.errorbar(
        x,
        df["mean"],
        yerr=[df["mean"] - df["min"], df["max"] - df["mean"]],
        fmt="none",
        ecolor="red",
        capsize=4,
        label="Min/Max",
    )

    # X-axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels(df["ged"], rotation=90)
    ax.axvline(-0.5, color="gray", ls="--", alpha=0.5)

    unique_strings = df["string"].unique()
    for s in unique_strings:
        idx = df.index[df["string"] == s]
        left, right = idx.min(), idx.max()
        ax.axvline(right + 0.5, color="gray", ls="--", alpha=0.5)
        ax.text(left, -3.5, f"String {s}", rotation=90)

    ax.set_ylabel("FEP gain variation [keV]")
    ax.set_title(f"{period} {run}")
    ax.legend(loc="upper right")

    ax.axhline(0, ls="--", color="black")

    plt.ylim(-5, 5)
    plt.tight_layout()

    if save_pdf:
        pdf_folder = os.path.join(output_dir, f"{period}/{run}/mtg/pdf")
        plt.savefig(
            os.path.join(
                pdf_folder,
                f"{period}_{run}_FEP_gain_variation_summary.pdf",
            ),
            bbox_inches="tight",
        )

    # store the serialized plot in a shelve object under key
    serialized_plot = pickle.dumps(plt.gcf())
    with shelve.open(
        os.path.join(
            output_dir,
            f"{period}/{run}/mtg/l200-{period}-{run}-cal-monitoring",
        ),
        "c",
        protocol=pickle.HIGHEST_PROTOCOL,
    ) as shelf:
        shelf[f"{period}_{run}_FEP_gain_variation_summary"] = serialized_plot

    plt.close()


def load_calib_results(period, run, prod_ref_dir):
    directory = os.path.join(prod_ref_dir, "generated/par/hit/cal", period, run)
    file = glob.glob(os.path.join(directory, "*par_hit.yaml"))[0]
    return yaml.safe_load(open(file))


def check_calibration(
    tmp_auto_dir, output_folder, period, run, det_info, save_pdf=False
):

    hit_files = sorted(
        glob.glob(
            os.path.join(tmp_auto_dir, "generated/tier/hit/cal", period, run, "*")
        )
    )

    pars = load_calib_results(period, run, tmp_auto_dir)

    detectors = det_info["detectors"]
    fep_mean_results = {}

    for ged, item in detectors.items():
        if not item["processable"]:
            continue

        hit_files_data = lh5.read_as(
            item["channel_str"] + "/hit/",
            hit_files,
            library="ak",
            field_mask=["cuspEmax_ctc_cal", "timestamp", "is_valid_cal"],
        )

        mask = (
            hit_files_data.is_valid_cal
            & (hit_files_data.cuspEmax_ctc_cal > 2600)
            & (hit_files_data.cuspEmax_ctc_cal < 2630)
        )
        timestamps = hit_files_data[mask].timestamp.to_numpy()
        timestamps -= timestamps[0]
        energies = hit_files_data[mask].cuspEmax_ctc_cal.to_numpy()

        fep_mean_results[ged] = fep_gain_variation(
            period,
            run,
            pars=pars[ged],
            chmap=item,
            timestamps=timestamps,
            values=energies,
            output_dir=output_folder,
            save_pdf=save_pdf,
        )
    fep_gain_variation_summary(
        period, run, pars, detectors, fep_mean_results, output_folder, save_pdf
    )

    output = {
        ged: {
            "cal": {
                "npeak": None,
                "fwhm_ok": None,
                "FEP_gain_stab": None,
                "const_stab": None,
                "PSD": None,
                "first_run": True,
            },
            "phy": {"pulser_stab": None, "baseln_stab": None, "baseln_spike": None},
        }
        for ged in detectors
    }

    for ged, item in detectors.items():
        if item["system"] != "geds":
            continue
        if not item["analysis"]["processable"]:
            continue

        if pars[ged]["results"]["ecal"]["cuspEmax_ctc_cal"]["pk_fits"][2614.511][
            "validity"
        ]:
            output[ged]["cal"]["npeak"] = True
        else:
            output[ged]["cal"]["npeak"] = False

        fwhm = pars[ged]["results"]["ecal"]["cuspEmax_ctc_cal"]["eres_linear"][
            "Qbb_fwhm_in_kev"
        ]
        if np.isnan(fwhm):
            output[ged]["cal"]["fwhm_ok"] = False
            output[ged]["cal"]["FEP_gain_stab"] = False
        else:
            output[ged]["cal"]["fwhm_ok"] = True

        if output[ged]["cal"]["fwhm_ok"]:
            fep_means = fep_mean_results[ged]
            max_dev = fep_means.max()
            min_dev = fep_means.min()

            # if max_dev > fwhm / 2 or min_dev < -fwhm / 2: potential issue here
            if max_dev > 2 or min_dev < -2:  # ±2 keV
                output[ged]["cal"]["FEP_gain_stab"] = False
            else:
                output[ged]["cal"]["FEP_gain_stab"] = True
        else:
            output[ged]["cal"]["FEP_gain_stab"] = False

        if run == "r000":
            output[ged]["cal"]["first_run"] = True
        else:
            output[ged]["cal"]["first_run"] = False

        if not output[ged]["cal"]["first_run"]:
            if output[ged]["cal"]["fwhm_ok"]:
                gain = pars[ged]["results"]["ecal"]["cuspEmax_ctc_cal"]["eres_linear"][
                    "parameters"
                ]["a"]
                prev_run = f"r{int(run[1:])-1:03d}"
                prev_pars = load_calib_results(period, prev_run, tmp_auto_dir)
                prev_gain = prev_pars[ged]["results"]["ecal"]["cuspEmax_ctc_cal"][
                    "eres_linear"
                ]["parameters"]["a"]

                gain_change_keV = abs(gain - prev_gain) / prev_gain * 2039

                # if abs(gain_change_keV) > fwhm / 2: same as above
                if abs(gain_change_keV) > 2:  # 2 keV
                    output[ged]["cal"]["baseln_stab"] = False
                else:
                    output[ged]["cal"]["baseln_stab"] = True

            else:
                output[ged]["const_stab"] = False

    yaml.dump(
        output,
        open(
            os.path.join(
                output_folder, f"{period}/{run}/l200-{period}-{run}-qcp_summary.yaml"
            ),
            "w",
        ),
    )
