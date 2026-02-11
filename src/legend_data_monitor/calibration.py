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

from . import monitoring, utils

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
        if "old" in file_path.split("/")[-2]:
            continue

        run_idx = int(file_path.split("/")[-2].split("r")[-1])
        run_str = f"r{run_idx:03d}"
        if run_str not in avail_runs:
            continue

        run_data = utils.read_json_or_yaml(file_path)
        time = file_path.split("-")[-2]

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
    if not (len(mean_vals) == len(sigma_vals) == len(run_labels)):
        results["status"] = None
        return results
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
    slow_shifts = [float((v - mean_vals[valid_idx]) / sigma_avg) for v in mean_vals]

    slow_shift_fail_runs = []
    for i, z in enumerate(slow_shifts):
        if run_labels[i] != current_run:
            continue

        # If fit pars from yaml are missinng -> fail
        if np.isnan(mean_vals[i]) or np.isnan(sigma_vals[i]) or sigma_vals[i] == 0:
            slow_shift_fail_runs.append(run_labels[i])
            continue

        # Slow shift threshold
        if abs(z) > 0.5:
            slow_shift_fail_runs.append(run_labels[i])
    slow_shift_failed = bool(slow_shift_fail_runs)

    # SUDDEN shifts
    # Fix first entry to 0 (if present), else NaN
    if np.isnan(mean_vals[0]) or np.isnan(sigma_vals[0]) or sigma_vals[0] == 0:
        sudden_shifts = [float("nan")]
    else:
        sudden_shifts = [0.0]
    # Backward logic
    for i in range(1, len(mean_vals)):
        mu_curr = mean_vals[i]
        mu_prev = mean_vals[i - 1]
        sigma_curr = sigma_vals[i]

        if (
            np.isnan(mu_curr)
            or np.isnan(mu_prev)
            or np.isnan(sigma_curr)
            or sigma_curr == 0
        ):
            sudden_shifts.append(float("nan"))
        else:
            val = abs(mu_curr - mu_prev) / sigma_curr
            sudden_shifts.append(float(val))

    sudden_shift_fail_runs = []
    for i, z in enumerate(sudden_shifts):
        if run_labels[i] != current_run:
            continue

        # If fit pars from yaml are missinng -> fail
        if np.isnan(mean_vals[i]) or np.isnan(sigma_vals[i]) or sigma_vals[i] == 0:
            sudden_shift_fail_runs.append(run_labels[i])
            continue

        # Slow shift threshold (if z is NaN here, PREVIOUS run was missing - let that PASS)
        if not np.isnan(z) and z > 0.25:
            sudden_shift_fail_runs.append(run_labels[i])

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

    # Mean stability
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

    # Slow shifts
    ax3.plot(
        run_positions,
        eval_result["slow_shifts"],
        marker="^",
        markersize=10,
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

    # Sudden shifts
    y = np.array(eval_result["sudden_shifts"])
    x = np.array(run_positions)
    ax4.plot(
        x,
        y,
        marker="^",
        markersize=10,
        linestyle="-",
        color="green",
        label="Sudden shifts",
    )
    ax4.axhline(0, color="black", linestyle="--")
    ax4.axhline(0.25, color="crimson", linestyle="--")
    ax4.set_ylabel(r"$|(\mu_{i}-\mu_{i-1})/\sigma_i|$")
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
    utils.update_evaluation_in_memory(
        psd_data, det_name, "cal", "PSD", eval_result["status"]
    )


def check_psd(
    auto_dir_path: str,
    cal_path: str,
    pars_files_list: list,
    output_dir: str,
    period: str,
    current_run: str,
    det_info: dict,
    save_pdf: bool,
):
    """
    Evaluate the PSD usability for a set of detectors based on calibration results; save results in a YAML summary file; plot per-detector PSD stability data and store them as shelve file (and pdf if wanted).

    Parameters
    ----------
    auto_dir_path : str
        Path to tmp-auto public data files (eg /data2/public/prodenv/prod-blind/tmp-auto).
    cal_path : str
        Path to the directory containing calibration runs (eg /data2/public/prodenv/prod-blind/tmp-auto/generated/par/<tier>/cal/<period>).
    pars_files_list : list
        List of YAML/JSON files containing results for each calibration run.
    output_dir : str
        Path to output folder where the output summary YAML and plots will be stored.
    period : str
        Period to inspect.
    current_run : str
        Run to inspect.
    det_info : dict
        Dictionary containing detector metadata.
    save_pdf : bool
        True if you want to save pdf files too; default: False.
    """
    if not any(current_run in file for file in pars_files_list):
        utils.logger.debug(
            f"...no calibration files found for run {current_run}. Exiting."
        )
        return

    # create the folder and parents if missing - for the moment, we store it under the 'phy' folder
    output_dir_run = os.path.join(output_dir, period, current_run)
    os.makedirs(os.path.join(output_dir_run, "mtg"), exist_ok=True)

    # Load existing data once (or start empty)
    usability_map_file = os.path.join(
        output_dir_run, f"l200-{period}-{current_run}-qcp_summary.yaml"
    )

    detectors_name = list(det_info["detectors"].keys())
    detectors_list = [det_info["detectors"][d]["channel_str"] for d in detectors_name]
    locations_list = [
        (det_info["detectors"][d]["string"], det_info["detectors"][d]["position"])
        for d in detectors_name
    ]

    psd_data = utils.load_yaml_or_default(usability_map_file, det_info["detectors"])

    cal_runs = sorted(os.listdir(cal_path))
    if len(cal_runs) == 1:
        utils.logger.debug(
            "Only one available calibration run. Save all entries as None and exit."
        )
        for det_name in detectors_name:
            utils.update_evaluation_in_memory(psd_data, det_name, "cal", "PSD", None)

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
    utils.logger.debug("...inspecting PSD stability in cal runs")
    for idx, det_name in enumerate(detectors_name):
        evaluate_psd_usability_and_plot(
            period,
            current_run,
            cal_psd_info[det_name],
            det_name,
            locations_list[idx],
            os.path.join(output_dir, period),
            psd_data,
            save_pdf,
        )

    with open(usability_map_file, "w") as f:
        yaml.dump(psd_data, f, sort_keys=False)


def fep_gain_variation(
    period: str,
    run: str,
    pars: dict,
    chmap: dict,
    timestamps: np.ndarray,
    values: np.ndarray,
    output_dir: str,
    save_pdf: bool,
    shelf: shelve.Shelf,
):
    """
    Compute and plot FEP gain variation for a single detector; optional pdf saving; store a serialized plot in a shelve object.

    Parameters
    ----------
    period : str
        Period to inspect.
    run : str
        Run to inspect.
    pars : dict
        Calibration results dictionary for a given detector.
    chmap : dict
        Dictionary with detector info, must include 'name', 'string', 'position'.
    timestamps : np.ndarray
        Array of timestamps for a given detector.
    values : np.ndarray
        Array of energies for a given detector.
    output_dir : str
        Path to output folder where plots will be stored.
    save_pdf : bool
        If True, save a PDF of the plot.
    shelf : shelve.Shelf
        Open shelve object where serialized plots will be stored.
    """
    ged = chmap["name"]
    string = chmap["string"]
    position = chmap["position"]

    bin_size = 600
    bins = np.arange(0, timestamps.max() + bin_size, bin_size)

    bin_idx = np.digitize(timestamps, bins) - 1  # shift to 0-based

    df = pd.DataFrame({"time": timestamps, "value": values, "bin": bin_idx})

    stats = df.groupby("bin")["value"].agg(["mean", "std", "count"]).reset_index()
    stats["time"] = bins[stats["bin"]] + bin_size / 2

    min_counts = 5
    stats.loc[stats["count"] < min_counts, ["mean", "std"]] = np.nan

    fig, ax = plt.subplots(figsize=(10, 5))

    # Choose baseline: first mean if valid, otherwise last valid mean
    valid_means = stats["mean"].dropna()
    if not valid_means.empty:
        if pd.notna(stats["mean"].iloc[0]):
            baseline = stats["mean"].iloc[0]
        else:
            baseline = stats["mean"].dropna().iloc[-1]

        norm_values = (values - baseline) / baseline * 2039

        x_bins = bins
        y_bins = np.linspace(-10, 10, 40)
        means = (stats["mean"] - baseline) / baseline * 2039

        ax.hist2d(timestamps, norm_values, bins=(x_bins, y_bins), cmap="Blues")
        fig.colorbar(ax.collections[0], label="Counts")

        ax.plot(stats["time"], means, "x-", color="red", label="10 min mean")

        ax.fill_between(
            stats["time"],
            (stats["mean"] - stats["std"] - baseline) / baseline * 2039,
            (stats["mean"] + stats["std"] - baseline) / baseline * 2039,
            color="red",
            alpha=0.2,
            label="±1 std",
        )

    fwhm = (
        (pars or {}).get("results")
        or {}.get("ecal")
        or {}.get("cuspEmax_ctc_cal")
        or {}.get("eres_linear")
        or {}.get("Qbb_fwhm_in_kev")
    )
    if isinstance(fwhm, (int, float)) and not np.isnan(fwhm):
        if fwhm < 5:
            plt.ylim(-5, 5)

        plt.axhline(0, ls="--", color="black")
        plt.axhline(-fwhm / 2, ls="-", color="blue")
        plt.axhline(
            fwhm / 2, ls="-", color="blue", label=f"±FWHM/2 = ±{fwhm/2:.2f} keV"
        )

    plt.legend(loc="lower left", title=f"Min. counts = {min_counts}")
    plt.xlabel("time [s]")
    plt.ylabel("FEP gain variation [keV]")
    plt.title(f"{period} {run} string {string} position {position} {ged}")
    plt.tight_layout()

    if save_pdf:
        pdf_folder = os.path.join(output_dir, period, run, "mtg/pdf", f"st{string}")
        os.makedirs(pdf_folder, exist_ok=True)
        plt.savefig(
            os.path.join(
                pdf_folder,
                f"{period}_{run}_string{string}_pos{position}_{ged}_FEP_gain_variation.pdf",
            ),
            bbox_inches="tight",
        )

    # store the serialized plot in a shelve object under key
    serialized_plot = pickle.dumps(plt.gcf())
    shelf[f"{period}_{run}_str{string}_pos{position}_{ged}_FEP_gain_variation"] = (
        serialized_plot
    )
    plt.close()

    if valid_means.empty:
        return None

    return means


def check_calibration(
    tmp_auto_dir: str,
    output_folder: str,
    period: str,
    run: str,
    first_run: bool,
    det_info: dict,
    save_pdf=False,
):
    """
    Check calibration stability for a given run and update monitoring summary YAML file.

    Parameters
    ----------
    tmp_auto_dir : str
        Path to tmp-auto public data files (eg /data2/public/prodenv/prod-blind/tmp-auto).
    output_folder : str
        Path to output folder where the output summary YAML and plots will be stored.
    period : str
        Period to inspect.
    run : str
        Run to inspect.
    first_run : bool
        Flag indicating whether this is the first run of the period.
    det_info : dict
        Dictionary containing detector metadata.
    save_pdf : bool
        True if you want to save pdf files too; default: False.
    """
    detectors = det_info["detectors"]
    usability_map_file = os.path.join(
        output_folder, period, run, f"l200-{period}-{run}-qcp_summary.yaml"
    )
    output = utils.load_yaml_or_default(usability_map_file, detectors)
    fep_mean_results = {}

    directory = os.path.join(tmp_auto_dir, "generated/par/hit/cal", period, run)
    files = sorted(glob.glob(os.path.join(directory, "*par_hit.yaml")))
    if not files:
        utils.logger.debug(f"...no calibration files found for run {run}. Exiting.")
        return
    pars = utils.read_json_or_yaml(files[0])

    # avoid case where multiple cal runs were processed but we are still requiring to inspect the first run
    if run in files:
        first_run = True

    # find nearest previous run
    prev_pars = None
    if not first_run:
        run_number = int(run[1:])
        for offset in range(1, run_number + 1):  # check run-1, run-2, ...
            prev_run = f"r{run_number - offset:03d}"
            directory = os.path.join(
                tmp_auto_dir, "generated/par/hit/cal", period, prev_run
            )
            files = sorted(glob.glob(os.path.join(directory, "*par_hit.yaml")))
            if files:
                utils.logger.debug(f"...using previous calibration from {prev_run}")
                prev_pars = utils.read_json_or_yaml(files[0])
                break

        if prev_pars is None:
            utils.logger.debug(
                f"No previous calibration files found for {run}, treat as first run"
            )
            first_run = True

    shelve_path = os.path.join(
        output_folder,
        period,
        run,
        f"mtg/l200-{period}-{run}-cal-monitoring",
    )
    os.makedirs(os.path.dirname(shelve_path), exist_ok=True)
    utils.logger.debug("...inspecting FEP, calib peaks, stability in calibrations")

    hit_files = sorted(
        glob.glob(
            os.path.join(tmp_auto_dir, "generated/tier/hit/cal", period, run, "*")
        )
    )

    with shelve.open(shelve_path, "c", protocol=pickle.HIGHEST_PROTOCOL) as shelf:
        for ged, item in detectors.items():
            if not item["processable"]:
                continue

            # avoid cases where the detector is not present in the output files
            if item["channel_str"] not in lh5.ls(hit_files[0], ""):
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
            if timestamps.size == 0:
                continue
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
                shelf=shelf,
            )

            # build summary in memory
            ecal_results = pars[ged]["results"]["ecal"]
            ecal = monitoring.get_energy_key(
                ecal_results
            )  # check for cuspEmax_ctc_runcal or cuspEmax_ctc_cal
            pk_fits = monitoring.get_energy_key(ecal_results).get("pk_fits", {})

            operations = pars[ged]["operations"]
            operations_ecal = monitoring.get_energy_key(
                operations
            )  # check for cuspEmax_ctc_runcal or cuspEmax_ctc_cal

            # find FEP and low-E peaks (keys digits changed in the past, so let's be generic)
            fep_peaks = [p for p in pk_fits if 2613 < p < 2616]
            low_peaks = [p for p in pk_fits if 580 < p < 586]

            fep_valid = False
            low_valid = False
            if fep_peaks:
                fep_energy = fep_peaks[0]
                fep_valid = ecal["pk_fits"][fep_energy].get("validity", False)
            if low_peaks:
                low_energy = low_peaks[0]
                low_valid = ecal["pk_fits"][low_energy].get("validity", False)

            # true only if both peaks are valid
            overall_valid = fep_valid and low_valid
            utils.update_evaluation_in_memory(
                output, ged, "cal", "npeak", overall_valid
            )

            fwhm = (ecal.get("eres_linear") or {}).get("Qbb_fwhm_in_kev")
            fwhm_ok = isinstance(
                fwhm, (int, float, np.integer, np.floating)
            ) and not np.isnan(fwhm)
            utils.update_evaluation_in_memory(output, ged, "cal", "fwhm_ok", fwhm_ok)

            # FEP gain stability - independent from fwhm; if we use that value, than put it back in the if statement
            if fep_mean_results[ged] is not None:
                # remove nan (gaps) or it will return False
                arr = np.array(fep_mean_results[ged], dtype=float)
                stable = bool(np.all(np.abs(arr[~np.isnan(arr)]) <= 2))
            else:
                stable = False
            utils.update_evaluation_in_memory(
                output, ged, "cal", "FEP_gain_stab", stable
            )

            if fwhm_ok:
                # bsln stability (only if not first run)
                if not first_run:
                    # channel might not be present in the previous run, leave it None if so
                    if ged in prev_pars:
                        gain = operations_ecal["parameters"]["a"]
                        prev_gain = monitoring.get_energy_key(
                            prev_pars[ged]["operations"]
                        )["parameters"]["a"]
                        gain_dev = abs(gain - prev_gain) / prev_gain * 2039
                        utils.update_evaluation_in_memory(
                            output, ged, "cal", "const_stab", gain_dev <= 2
                        )

            else:
                if not first_run:
                    utils.update_evaluation_in_memory(
                        output, ged, "cal", "const_stab", False
                    )

    # plot
    monitoring.box_summary_plot(
        period,
        run,
        pars,
        det_info,
        fep_mean_results,
        utils.MTG_PLOT_INFO["FEP_variation"],
        output_folder,
        "cal",
        save_pdf,
    )

    with open(usability_map_file, "w") as f:
        yaml.dump(output, f)

    with open(usability_map_file, "w") as f:
        yaml.dump(output, f)


def check_calibration_lac_ssc(
    tmp_auto_dir: str,
    output_folder: str,
    period: str,
    run: str,
    run_to_apply: str,
    first_run: bool,
    det_info: dict,
    data_type="cal",
    save_pdf=False,
):
    """
    Check calibration stability for a given run and update monitoring summary YAML file in special LAC or SSC data.

    Parameters
    ----------
    tmp_auto_dir : str
        Path to tmp-auto public data files (eg /data2/public/prodenv/prod-blind/tmp-auto).
    output_folder : str
        Path to output folder where the output summary YAML and plots will be stored.
    period : str
        Period to inspect.
    run : str
        Run to inspect.
    run_to_apply : str
        Calibration run to apply to these data.
    first_run : bool
        Flag indicating whether this is the first run of the period.
    det_info : dict
        Dictionary containing detector metadata.
    save_pdf : bool
        True if you want to save pdf files too; default: False.
    """
    detectors = det_info["detectors"]
    usability_map_file = os.path.join(
        output_folder, period, run, f"l200-{period}-{run}-qcp_summary.yaml"
    )
    output = utils.load_yaml_or_default(usability_map_file, detectors)
    fep_mean_results = {}

    directory = os.path.join(
        tmp_auto_dir, "generated/par/hit/cal", period, run_to_apply
    )
    files = sorted(glob.glob(os.path.join(directory, "*par_hit.yaml")))
    if not files:
        utils.logger.debug(
            f"...no calibration files found for run {run_to_apply}. Exiting."
        )
        return
    pars = utils.read_json_or_yaml(files[0])

    # find nearest previous run
    shelve_path = os.path.join(
        output_folder,
        period,
        run,
        f"mtg/l200-{period}-{run}-{data_type}-monitoring",
    )
    os.makedirs(os.path.dirname(shelve_path), exist_ok=True)
    utils.logger.debug("...inspecting FEP, calib peaks, stability in calibrations")

    # load ssc/lac data
    hit_files = sorted(
        glob.glob(
            os.path.join(
                tmp_auto_dir, "generated/tier/hit", data_type, period, run, "*"
            )
        )
    )
    output = utils.load_yaml_or_default(usability_map_file, detectors)
    fep_mean_results = {}

    with shelve.open(shelve_path, "c", protocol=pickle.HIGHEST_PROTOCOL) as shelf:
        for ged, item in detectors.items():
            if not item["processable"]:
                continue

            # avoid cases where the detector is not present in the output files
            if item["channel_str"] not in lh5.ls(hit_files[0], ""):
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
            if timestamps.size == 0:
                continue
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
                shelf=shelf,
            )

            # build summary in memory
            ecal_results = pars[ged]["results"]["ecal"]
            ecal = monitoring.get_energy_key(
                ecal_results
            )  # check for cuspEmax_ctc_runcal or cuspEmax_ctc_cal
            pk_fits = monitoring.get_energy_key(ecal_results).get("pk_fits", {})

            # find FEP and low-E peaks (keys digits changed in the past, so let's be generic)
            fep_peaks = [p for p in pk_fits if 2613 < p < 2616]
            low_peaks = [p for p in pk_fits if 580 < p < 586]

            fep_valid = False
            low_valid = False
            if fep_peaks:
                fep_energy = fep_peaks[0]
                fep_valid = ecal["pk_fits"][fep_energy].get("validity", False)
            if low_peaks:
                low_energy = low_peaks[0]
                low_valid = ecal["pk_fits"][low_energy].get("validity", False)

            # true only if both peaks are valid
            overall_valid = fep_valid and low_valid
            utils.update_evaluation_in_memory(
                output, ged, data_type, "npeak", overall_valid
            )

            fwhm = (ecal.get("eres_linear") or {}).get("Qbb_fwhm_in_kev")
            fwhm_ok = isinstance(
                fwhm, (int, float, np.integer, np.floating)
            ) and not np.isnan(fwhm)
            utils.update_evaluation_in_memory(
                output, ged, data_type, "fwhm_ok", fwhm_ok
            )

            # FEP gain stability - independent from fwhm; if we use that value, than put it back in the if statement
            if fep_mean_results[ged] is not None:
                # remove nan (gaps) or it will return False
                arr = np.array(fep_mean_results[ged], dtype=float)
                stable = bool(np.all(np.abs(arr[~np.isnan(arr)]) <= 2))
            else:
                stable = False
            utils.update_evaluation_in_memory(
                output, ged, data_type, "FEP_gain_stab", stable
            )

    # plot
    monitoring.box_summary_plot(
        period,
        run,
        pars,
        det_info,
        fep_mean_results,
        utils.MTG_PLOT_INFO["FEP_variation"],
        output_folder,
        data_type,
        save_pdf,
    )

    with open(usability_map_file, "w") as f:
        yaml.dump(output, f)
