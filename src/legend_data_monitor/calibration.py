import glob
import os
import pickle
import shelve

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import yaml
from legendmeta import LegendMetadata

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


def update_psd_evaluation_in_memory(data: dict, det_name: str, value: bool | float):
    """Update the PSD entry in memory dict, where value can be bool or nan if not available."""
    data.setdefault(det_name, {}).setdefault("cal", {})["PSD"] = value


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
            f"l200-{period}-phy-monitoring",
        ),
        "c",
        protocol=pickle.HIGHEST_PROTOCOL,
    ) as shelf:
        shelf[
            f"{period}_string{location[0]}_pos{location[1]}_{det_name}_PSDusability"
        ] = serialized_plot

    plt.close()

    # supdate psd status
    update_psd_evaluation_in_memory(psd_data, det_name, eval_result["status"])


def check_psd(
    auto_dir_path: str, output_dir: str, period: str, current_run: str, save_pdf: bool
):

    found = False
    for tier in ["hit", "pht"]:
        cal_path = os.path.join(auto_dir_path, "generated/par", tier, "cal", period)
        if os.path.isdir(cal_path):
            found = True
            break
    if found is False:
        utils.logger.debug(f"No valid folder {cal_path} found. Exiting.")
        exit()

    pars_files_list = sorted(glob.glob(f"{cal_path}/*/*.yaml"))
    if not pars_files_list:
        pars_files_list = sorted(glob.glob(f"{cal_path}/*/*.json"))

    metadata_path = os.path.join(auto_dir_path, "inputs")
    lmeta = LegendMetadata(metadata_path)
    chmap = lmeta.channelmap()
    detectors_name = [det for det, info in chmap.items() if info["system"] == "geds"]
    detectors_list = [
        f"ch{chmap[det]['daq']['rawid']}"
        for det in detectors_name
        if chmap[det]["name"] == det
    ]
    locations_list = [
        (chmap[det]["location"]["position"], chmap[det]["location"]["string"])
        for det in detectors_name
        if chmap[det]["name"] == det
    ]

    # retrieve all dets info
    cal_runs = sorted(os.listdir(cal_path))
    cal_psd_info = load_fit_pars_from_yaml(
        pars_files_list, detectors_list, detectors_name, cal_runs
    )
    if cal_psd_info is None:
        utils.logger.debug("...no data are available at the moment")
        return

    # create the folder and parents if missing - for the moment, we store it under the 'phy' folder
    output_dir = os.path.join(output_dir, period, current_run)
    os.makedirs(output_dir, exist_ok=True)

    # Load existing data once (or start empty)
    usability_map_file = os.path.join(
        output_dir, f"l200-{period}-{current_run}-summary.yaml"
    )

    if os.path.exists(usability_map_file):
        with open(usability_map_file) as f:
            psd_data = yaml.safe_load(f) or {}
    else:
        psd_data = {}

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
