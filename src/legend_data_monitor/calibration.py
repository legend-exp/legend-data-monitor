import glob
import os

import awkward as ak
import lh5
import numpy as np
import pandas as pd
import yaml

from . import monitoring, utils
from .contract import writer as contract_writer


# -------------------------------------------------------------------------
def get_partitions_params(
    ge_keys: list, detector_status: dict, run_dict: dict, hit_map: dict, dsp_map: dict
) -> dict:
    """
    Build per-detector calibration and analysis parameters across runs.

    Returns a nested dictionary: det -> parameter -> peak -> run_key -> value

    Parameters
    ----------
    ge_keys : list of str
        Detector names.
    detector_status : dict
        Detector status per period-run: detector_status[det][period-run]['processable'/'usability'].
    run_dict : dict
        Mapping period to list of runs.
    hit_map : dict
        Mapping (period, run) to hit file path.
    dsp_map : dict
        Mapping (period, run) to dsp file path.
    """
    all_params_ch = {}

    hit_cache = {}
    dsp_cache = {}

    ref_cal_pars_map = {}

    for det_name in ge_keys:
        all_params_ch[det_name] = {
            "mus_peaks": {},
            "mus_err_peaks": {},
            "mus_keV_peaks": {},
            "mus_keV_err_peaks": {},
            "mus_keV_first_cal_peaks": {},
            "mus_keV_first_cal_err_peaks": {},
            "fwhms_peaks": {},
            "fwhms_err_peaks": {},
            "gains": {},
            "gains_err": {},
            "ctc_alpha_par": {},
            "bl_std": {},
            "bl_max": {},
            "bl_std_err": {},
            "bl_max_err": {},
            "aoe_mu": {},
            "aoe_mu_err": {},
            "aoe_sigma": {},
            "aoe_sigma_err": {},
            "cusp_sigma": {},
            "etrap_rise": {},
            "zac_sigma": {},
            "pz_tau": {},
            "cal_params": {},
            "residuals": {},
        }

    for period, runs in run_dict.items():
        for run in runs:
            key = f"{period}-{run}"

            hit_fname = hit_map.get((period, run))
            dsp_fname = dsp_map.get((period, run))

            if hit_fname is None:
                continue

            if hit_fname not in hit_cache:
                with open(hit_fname) as f:
                    hit_cache[hit_fname] = yaml.safe_load(f)
            data_ph = hit_cache[hit_fname]

            if dsp_fname:
                if dsp_fname not in dsp_cache:
                    with open(dsp_fname) as f:
                        dsp_cache[dsp_fname] = yaml.safe_load(f)
                data_pd = dsp_cache[dsp_fname]
            else:
                data_pd = {}

            for det_name in ge_keys:
                if not detector_status[det_name]["processable"].get(key, False):
                    continue

                try:
                    pars_per_ch = all_params_ch[det_name]
                    data_det = data_ph[det_name]
                    ecal = data_det["results"]["ecal"]
                    estimator, results = monitoring.find_energy_key(ecal)
                    if not results:
                        continue
                    peak_fits = results["pk_fits"]
                    cal_op = data_det["pars"]["operations"][estimator]["parameters"]
                    cal_pars = list(cal_op.values())

                    pars_per_ch["gains"][key] = results["eres_linear"]["parameters"][
                        "b"
                    ]
                    pars_per_ch["gains_err"][key] = results["eres_linear"][
                        "uncertainties"
                    ]["b"]
                    pars_per_ch["ctc_alpha_par"][key] = data_det["pars"]["operations"][
                        monitoring.uncalibrated_variable(estimator)
                    ]["parameters"]["a"]

                    if det_name not in ref_cal_pars_map:
                        ref_cal_pars_map[det_name] = {}

                    # set reference only once per period, using first GOOD run
                    if (
                        period not in ref_cal_pars_map[det_name]
                        and detector_status[det_name]["usability"].get(key) == "on"
                    ):
                        ref_cal_pars_map[det_name][period] = cal_pars

                    # fallback if no "on" run was found yet
                    if period in ref_cal_pars_map[det_name]:
                        ref_cal_pars = ref_cal_pars_map[det_name][period]
                    else:
                        ref_cal_pars = cal_pars  # fallback to current run

                    for peak, peak_data in peak_fits.items():

                        if peak not in pars_per_ch["mus_keV_peaks"]:
                            for field in [
                                "mus_peaks",
                                "mus_err_peaks",
                                "mus_keV_peaks",
                                "mus_keV_err_peaks",
                                "mus_keV_first_cal_peaks",
                                "mus_keV_first_cal_err_peaks",
                                "fwhms_peaks",
                                "fwhms_err_peaks",
                                "residuals",
                            ]:
                                pars_per_ch[field][peak] = {}

                        peak_tmp = peak_data["position"]
                        peak_tmp_err = peak_data["position_uncertainty"]
                        fwhm_tmp = peak_data["fwhm_in_kev"]
                        fwhm_err_tmp = peak_data["fwhm_err_in_kev"]

                        peak_kev = np.polynomial.polynomial.polyval(peak_tmp, cal_pars)
                        peak_kev_err = np.polynomial.polynomial.polyval(
                            peak_tmp_err, cal_pars
                        )
                        peak_kev_first = np.polynomial.polynomial.polyval(
                            peak_tmp, ref_cal_pars
                        )
                        peak_kev_first_err = np.polynomial.polynomial.polyval(
                            peak_tmp_err, ref_cal_pars
                        )

                        pars_per_ch["mus_peaks"][peak][key] = peak_tmp
                        pars_per_ch["mus_err_peaks"][peak][key] = peak_tmp_err
                        pars_per_ch["mus_keV_peaks"][peak][key] = peak_kev
                        pars_per_ch["mus_keV_err_peaks"][peak][key] = peak_kev_err
                        pars_per_ch["mus_keV_first_cal_peaks"][peak][
                            key
                        ] = peak_kev_first
                        pars_per_ch["mus_keV_first_cal_err_peaks"][peak][
                            key
                        ] = peak_kev_first_err
                        pars_per_ch["fwhms_peaks"][peak][key] = fwhm_tmp
                        pars_per_ch["fwhms_err_peaks"][peak][key] = fwhm_err_tmp
                        pars_per_ch["residuals"][peak][key] = peak_kev - float(peak)

                    pars_per_ch["cal_params"][key] = cal_pars

                    mon = ecal["monitoring_parameters"]
                    pars_per_ch["bl_std"][key] = mon["bl_std"]["mode"]
                    pars_per_ch["bl_max"][key] = mon["baselineEmax"]["mode"]
                    pars_per_ch["bl_std_err"][key] = mon["bl_std"]["stdev"]
                    pars_per_ch["bl_max_err"][key] = mon["baselineEmax"]["stdev"]

                    aoe_block = data_det["results"]["aoe"]["1000-1300keV"]
                    ts = next(iter(aoe_block))
                    aoe = aoe_block[ts]

                    pars_per_ch["aoe_mu"][key] = aoe["mean"]
                    pars_per_ch["aoe_mu_err"][key] = aoe["mean_err"]
                    pars_per_ch["aoe_sigma"][key] = aoe["sigma"]
                    pars_per_ch["aoe_sigma_err"][key] = aoe["sigma_err"]

                    if dsp_fname and det_name in data_pd:
                        det_dsp = data_pd[det_name]

                        if "cusp" in det_dsp:
                            pars_per_ch["cusp_sigma"][key] = float(
                                det_dsp["cusp"]["sigma"].split("*")[0]
                            )
                        if "etrap" in det_dsp:
                            pars_per_ch["etrap_rise"][key] = float(
                                det_dsp["etrap"]["rise"].split("*")[0]
                            )
                        if "pz" in det_dsp:
                            pars_per_ch["pz_tau"][key] = (
                                float(det_dsp["pz"]["tau1"].split("*")[0]) / 1000
                            )

                except (KeyError, TypeError) as e:
                    utils.logger.error(f"Error with {det_name}")
                    utils.logger.error(e)
                    continue

    return all_params_ch


def check_escale(
    auto_dir_path: str,
    cal_path: str,
    output_folder: str,
    period: str,
    current_run: str,
    det_info: dict,
) -> dict:
    """
    Run energy-scale calibration checks and generate detector plots.

    Parameters
    ----------
    auto_dir_path : str
        Path to tmp-auto public data files (eg /data2/public/prodenv/prod-blind/tmp-auto).
    cal_path : str
        Path to the directory containing calibration runs (eg /data2/public/prodenv/prod-blind/tmp-auto/generated/par/<tier>/cal/<period>).
    output_folder : str
        Path to output folder where the summary plots will be stored.
    period : str
        Period to inspect.
    current_run : str
        Run to inspect.
    det_info : dict
        Dictionary containing detector metadata.

    Returns
    -------
    detector_status : dict
        Usability per detector and period-run (feeds the escale renderer).
    """
    utils.logger.debug("...inspecting energy-scale stability in cal runs")
    hit_map = utils.build_file_map(auto_dir_path, tier="hit")
    dsp_map = utils.build_file_map(auto_dir_path, tier="dsp")

    run_dict = {period: sorted(r for r in os.listdir(cal_path) if "_old" not in r)}
    detectors_name = list(det_info["detectors"].keys())

    detector_status = utils.build_detector_info_per_period(
        auto_dir_path, run_dict, period
    )

    partitions_params = get_partitions_params(
        detectors_name, detector_status, run_dict, hit_map, dsp_map
    )
    write_escale_summary(output_folder, period, current_run, partitions_params)

    output_dir_run = os.path.join(output_folder, period, current_run)
    os.makedirs(os.path.join(output_dir_run, "mtg"), exist_ok=True)
    usability_map_file = os.path.join(
        output_dir_run, f"l200-{period}-{current_run}-qcp_summary.yaml"
    )
    escale_data = utils.load_yaml_or_default(usability_map_file, det_info["detectors"])

    for det_name in detectors_name:
        eval_result = evaluate_escale_metrics(
            det_name,
            partitions_params[det_name],
            detector_status[det_name]["usability"],
            period,
            current_run,
        )
        for metric, verdict in eval_result.items():
            utils.update_evaluation_in_memory(
                escale_data, det_name, "cal", metric, verdict
            )

    with open(usability_map_file, "w") as f:
        yaml.dump(escale_data, f, sort_keys=False)

    return detector_status


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

        for idx, det in enumerate(detectors_list):
            det_key = det if det in run_data else detectors_name[idx]

            aoe_times = utils.deep_get(
                run_data or {}, [det_key, "results", "aoe", "1000-1300keV"], {}
            )
            if not aoe_times:
                pars = {}
            else:
                # take whichever time key
                time_key = next(iter(aoe_times))
                pars = aoe_times[time_key]

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


def write_psd_stability(
    output_folder: str,
    period: str,
    current_run: str,
    det_name: str,
    run_labels: list,
    mean_vals,
    mean_errs,
    sigma_vals,
    sigma_errs,
    eval_result: dict,
    data_type: str = "cal",
) -> str | None:
    """
    Write the per-run A/E fit means and sigmas behind the PSD stability figure.

    One row per (detector, run): the fit values with errors plus the shift
    verdicts, so the figure and the usability evaluation can be reproduced
    without unpickling anything.

    Parameters
    ----------
    output_folder : str
        Monitoring output root (the folder containing ``<period>/``).
    period, current_run : str
        Run the frame is written under.
    det_name : str
        Detector the rows belong to.
    run_labels : list
        Runs covered by the fit results.
    mean_vals, mean_errs, sigma_vals, sigma_errs : array-like
        A/E fit results per run.
    eval_result : dict
        Output of :func:`evaluate_psd_performance`.
    data_type : str
        Data type key of the period contract file.

    Returns
    -------
    key: str or None
        The key written, or None when there was nothing to write.
    """
    if not run_labels:
        return None
    slow_failed = set(eval_result.get("slow_shift_fail_runs") or [])
    sudden_failed = set(eval_result.get("sudden_shift_fail_runs") or [])
    frame = pd.DataFrame(
        {
            "detector": det_name,
            "run": run_labels,
            "mean": np.asarray(mean_vals, dtype=float),
            "mean_err": np.asarray(mean_errs, dtype=float),
            "sigma": np.asarray(sigma_vals, dtype=float),
            "sigma_err": np.asarray(sigma_errs, dtype=float),
            "slow_shift": [r in slow_failed for r in run_labels],
            "sudden_shift": [r in sudden_failed for r in run_labels],
            "status": str(eval_result.get("status")),
        }
    )
    path = monitoring.period_contract_path(output_folder, period, data_type)
    return contract_writer.write_frame(
        path, f"psd_stability/{current_run}/{det_name}", frame
    )


def record_psd_detail(
    period: str, current_run: str, det_name: str, run_labels: list, eval_result: dict
) -> None:
    """
    Stash the z-score behind a failed AoE_stab verdict for the issue record.

    Run-axis quantities: no excursion (its ``longest_s`` means seconds), so
    severity grades on the distance past the band.

    Parameters
    ----------
    period, current_run : str
        Keys of the verdict being explained.
    det_name : str
        Detector.
    run_labels : list
        Runs in the evaluation, aligned with the shift arrays.
    eval_result : dict
        :func:`evaluate_psd_performance` output.
    """
    if current_run not in run_labels:
        return
    i = run_labels.index(current_run)
    slow = eval_result.get("slow_shifts") or []
    sudden = eval_result.get("sudden_shifts") or []
    if current_run in eval_result.get("sudden_shift_fail_runs", []) and i < len(sudden):
        observed, threshold = sudden[i], [None, 0.25]
    elif i < len(slow):
        observed, threshold = slow[i], [-0.5, 0.5]
    else:
        return
    if observed is None or np.isnan(observed):
        return  # missing fit pars: the verdict carries no magnitude
    utils.issues.record_detail(
        period,
        current_run,
        "cal",
        det_name,
        "AoE_stab",
        observed=float(observed),
        threshold=threshold,
        unit="sigma",
    )


def evaluate_psd_usability(
    period: str,
    current_run: str,
    fit_results_cal: dict,
    det_name: str,
    output_dir: str,
    psd_data: dict,
):
    """Evaluate PSD stability across runs and publish the numbers behind it.

    Data-only: the figure is drawn from the contract by
    ``plots.calib.plot_psd_stability``.
    """
    run_labels = sorted(fit_results_cal.keys())

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
    if eval_result["status"] is False:
        record_psd_detail(period, current_run, det_name, run_labels, eval_result)

    # output_dir is <monitoring root>/<period> here; the writer joins period
    write_psd_stability(
        os.path.dirname(os.path.normpath(output_dir)),
        period,
        current_run,
        det_name,
        run_labels,
        mean_vals,
        mean_errs,
        sigma_vals,
        sigma_errs,
        eval_result,
    )

    # update psd status
    utils.update_evaluation_in_memory(
        psd_data, det_name, "cal", "AoE_stab", eval_result["status"]
    )


def check_psd(
    auto_dir_path: str,
    cal_path: str,
    pars_files_list: list,
    output_dir: str,
    period: str,
    current_run: str,
    det_info: dict,
):
    """
    Evaluate the PSD usability for a set of detectors based on calibration results; save results in a YAML summary file and publish the per-detector stability data to the cal period contract.

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

    psd_data = utils.load_yaml_or_default(usability_map_file, det_info["detectors"])

    cal_runs = sorted(os.listdir(cal_path))
    if len(cal_runs) == 1:
        utils.logger.debug(
            "Only one available calibration run. Save all entries as None and exit."
        )
        for det_name in detectors_name:
            utils.update_evaluation_in_memory(
                psd_data, det_name, "cal", "AoE_stab", None
            )

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

    utils.logger.debug("...inspecting PSD stability in cal runs")
    for det_name in detectors_name:
        evaluate_psd_usability(
            period,
            current_run,
            cal_psd_info[det_name],
            det_name,
            os.path.join(output_dir, period),
            psd_data,
        )

    with open(usability_map_file, "w") as f:
        yaml.dump(psd_data, f, sort_keys=False)


def compute_fep_gain_variation(
    timestamps: np.ndarray,
    values: np.ndarray,
    bin_size: int = 600,
    min_counts: int = 5,
    escale: float = 2039.0,
) -> dict:
    """Bin FEP energies in time and express the drift in keV at ``escale``.

    Pure computation behind the FEP gain-stability figure: bins of
    ``bin_size`` seconds, per-bin mean/std/count, bins with fewer than
    ``min_counts`` entries blanked, and the drift of each bin mean from the
    run's baseline (the first valid bin, else the last).

    Returns
    -------
    dict
        ``bins`` (edges), ``stats`` (per-bin time/mean/std/count),
        ``baseline`` and ``drift`` (keV at ``escale``; ``None`` when no bin has
        enough entries to define a baseline).
    """
    bins = np.arange(0, timestamps.max() + bin_size, bin_size)
    bin_idx = np.digitize(timestamps, bins) - 1  # shift to 0-based

    df = pd.DataFrame({"time": timestamps, "value": values, "bin": bin_idx})
    stats = df.groupby("bin")["value"].agg(["mean", "std", "count"]).reset_index()
    stats["time"] = bins[stats["bin"]] + bin_size / 2
    stats.loc[stats["count"] < min_counts, ["mean", "std"]] = np.nan

    valid_means = stats["mean"].dropna()
    baseline = None
    drift = None
    if not valid_means.empty:
        baseline = (
            stats["mean"].iloc[0]
            if pd.notna(stats["mean"].iloc[0])
            else valid_means.iloc[-1]
        )
        drift = (stats["mean"] - baseline) / baseline * escale
    return {"bins": bins, "stats": stats, "baseline": baseline, "drift": drift}


def fep_gain_variation(
    period: str,
    run: str,
    pars: dict,
    chmap: dict,
    timestamps: np.ndarray,
    values: np.ndarray,
) -> tuple:
    """
    Compute the FEP gain variation for a single detector.

    Data-only: the numbers land in the contract via
    :func:`write_fep_gain_contract` and the figure is drawn from there by
    ``plots.stability.plot_fep_gain``.

    Parameters
    ----------
    period, run : str
        Run to inspect.
    pars : dict
        Calibration results dictionary for a given detector.
    chmap : dict
        Detector info with 'name', 'string', 'position'.
    timestamps : np.ndarray
        Event timestamps for the detector.
    values : np.ndarray
        FEP energies for the detector.

    Returns
    -------
    means, computed : tuple
        Per-bin drift (None when no bin has enough entries) and the full
        :func:`compute_fep_gain_variation` result.
    """
    computed = compute_fep_gain_variation(
        timestamps, values, bin_size=600, min_counts=5
    )
    if computed["stats"]["mean"].dropna().empty:
        return None, computed
    return computed["drift"], computed


#: lmon's own event pass — see read_dataflow_stability
DATAFLOW_STABILITY_MIN_BINS = 5


def read_dataflow_stability(
    tmp_auto_dir: str,
    period: str,
    run: str,
    detector: str,
    estimator: str | None = None,
    key: str = "2614_stability",
    data_type: str = "cal",
) -> dict | None:
    """FEP/pulser stability arrays already computed by production dataflow.

    dataflow bins the same quantity lmon re-derives (``bin_stability`` in
    legend-dataflow-scripts) and ships it in the plt tier as
    ``{time, energy, spread}``. Reusing it would remove lmon's per-detector
    event pass entirely.

    **Not usable as of prod-blind auto/v2.0.0**: the arrays are binned in 180 s
    slices, which for a single calibration run leaves almost every bin empty --
    measured on p22/r012, the median detector has 1 populated bin out of 87 and
    *no* detector reaches ``DATAFLOW_STABILITY_MIN_BINS``. Once dataflow bins
    coarsely enough (or exports counts per bin), this becomes the fast path;
    until then callers fall back to the event pass. Returns None when the
    shelve, the detector or the key is missing.
    """
    import shelve

    pattern = os.path.join(
        tmp_auto_dir, "generated/plt/hit", data_type, period, run, "*-plt_hit.dat"
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    try:
        with shelve.open(matches[0].removesuffix(".dat"), "r") as shelf:
            entry = shelf.get(detector)
    except Exception as exc:  # unpicklable (matplotlib version drift), corrupt, ...
        utils.logger.debug("...cannot read dataflow stability: %s", exc)
        return None
    if not entry:
        return None
    ecal = entry.get("ecal", {})
    estimators = [estimator] if estimator else utils.EXPERIMENT["energy"]["estimators"]
    arrays = next(
        (a for e in estimators if (a := (ecal.get(e, {}) or {}).get(key))), None
    )
    if not arrays:
        return None
    return {name: np.asarray(values) for name, values in arrays.items()}


def dataflow_stability_usable(
    arrays: dict | None, min_bins: int = DATAFLOW_STABILITY_MIN_BINS
) -> bool:
    """Whether dataflow's arrays carry enough populated bins to be trusted."""
    if not arrays or "energy" not in arrays:
        return False
    energy = np.asarray(arrays["energy"], dtype=float)
    return int(np.count_nonzero(~np.isnan(energy))) >= min_bins


def read_channel_events(files: list, channel: str, fields: list):
    """Read one channel's cal events across ``files``.

    Reads a file per call and concatenates: handing ``lh5.read_as`` the whole
    file list is ~10x slower for the same rows (measured on p22/r012, 20
    detectors x 6 hit files: 21.2 s vs 2.1 s).
    """
    chunks = []
    for path in files:
        try:
            chunk = lh5.read_as(
                channel + "/hit/", [path], library="ak", field_mask=fields
            )
        except (KeyError, ValueError) as exc:
            utils.logger.debug(
                "skipping %s in %s: %s", channel, os.path.basename(path), exc
            )
            continue
        if chunk is not None and len(chunk):
            chunks.append(chunk)
    if not chunks:
        return None
    return ak.concatenate(chunks) if len(chunks) > 1 else chunks[0]


def check_calibration(
    tmp_auto_dir: str,
    output_folder: str,
    period: str,
    run: str,
    first_run: bool,
    det_info: dict,
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
    """
    detectors = det_info["detectors"]
    usability_map_file = os.path.join(
        output_folder, period, run, f"l200-{period}-{run}-qcp_summary.yaml"
    )
    output = utils.load_yaml_or_default(usability_map_file, detectors)
    fep_mean_results = {}
    fep_stats = {}

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

    os.makedirs(os.path.join(output_folder, period, run, "mtg"), exist_ok=True)
    utils.logger.debug("...inspecting FEP, calib peaks, stability in calibrations")

    # estimator column and peak windows (settings/experiment.yaml)
    energy_column = utils.EXPERIMENT["energy"]["estimators"][-1]
    fep_low, fep_high = utils.EXPERIMENT["energy"]["fep_window_kev"]
    fep_search = utils.EXPERIMENT["energy"]["peaks"]["fep"]["search_kev"]
    low_search = utils.EXPERIMENT["energy"]["peaks"]["low"]["search_kev"]

    hit_files = sorted(
        glob.glob(
            os.path.join(tmp_auto_dir, "generated/tier/hit/cal", period, run, "*")
        )
    )

    available_channels = set(lh5.ls(hit_files[0], ""))

    for ged, item in detectors.items():
        if not item["processable"]:
            continue

        # avoid cases where the detector is not present in the output files
        if item["channel_str"] not in available_channels:
            continue

        hit_files_data = read_channel_events(
            hit_files,
            item["channel_str"],
            [energy_column, "timestamp", "is_valid_cal"],
        )
        if hit_files_data is None:
            continue

        mask = (
            hit_files_data.is_valid_cal
            & (hit_files_data[energy_column] > fep_low)
            & (hit_files_data[energy_column] < fep_high)
        )
        timestamps = hit_files_data[mask].timestamp.to_numpy()
        if timestamps.size == 0:
            continue
        t_first = float(timestamps[0])
        timestamps -= timestamps[0]
        energies = hit_files_data[mask][energy_column].to_numpy()

        fep_mean_results[ged], fep_stats[ged] = fep_gain_variation(
            period,
            run,
            pars=pars[ged],
            chmap=item,
            timestamps=timestamps,
            values=energies,
        )

        # build summary in memory
        ecal_results = pars[ged]["results"]["ecal"]
        ecal = monitoring.get_energy_key(
            ecal_results
        )  # check for cuspEmax_ctc_runcal or cuspEmax_ctc_cal
        pk_fits = monitoring.get_energy_key(ecal_results).get("pk_fits", {})

        operations = pars[ged]["pars"]["operations"]
        operations_ecal = monitoring.get_energy_key(
            operations
        )  # check for cuspEmax_ctc_runcal or cuspEmax_ctc_cal

        # find FEP and low-E peaks (keys digits changed in the past, so let's be generic)
        fep_peaks = [p for p in pk_fits if fep_search[0] < p < fep_search[1]]
        low_peaks = [p for p in pk_fits if low_search[0] < p < low_search[1]]

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
        utils.update_evaluation_in_memory(output, ged, "cal", "npeak", overall_valid)

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
        utils.update_evaluation_in_memory(output, ged, "cal", "FEP_gain_stab", stable)
        if not stable and fep_stats.get(ged):
            record_fep_detail(period, run, "cal", ged, fep_stats[ged], t_first)

        if fwhm_ok:
            # bsln stability (only if not first run)
            if not first_run:
                # channel might not be present in the previous run, leave it None if so
                if ged in prev_pars:
                    gain = operations_ecal["parameters"]["b"]
                    prev_gain = monitoring.get_energy_key(
                        prev_pars[ged]["pars"]["operations"]
                    )["parameters"]["b"]
                    gain_dev = abs(gain - prev_gain) / prev_gain * 2039
                    utils.update_evaluation_in_memory(
                        output, ged, "cal", "const_stab", gain_dev <= 2
                    )
                    if gain_dev > 2:
                        utils.issues.record_detail(
                            period,
                            run,
                            "cal",
                            ged,
                            "const_stab",
                            observed=float(gain_dev),
                            threshold=[None, 2.0],
                            unit="keV",
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
    )

    write_fep_gain_contract(output_folder, period, run, fep_stats)

    with open(usability_map_file, "w") as f:
        yaml.dump(output, f)


def record_fep_detail(
    period: str, run: str, datatype: str, ged: str, computed: dict, t_first: float
) -> None:
    """
    Stash the magnitudes behind a failed FEP_gain_stab verdict for the issue record.

    Parameters
    ----------
    period, run, datatype : str
        Keys of the verdict being explained.
    ged : str
        Detector.
    computed : dict
        :func:`compute_fep_gain_variation` output (``stats`` with ``time`` in
        seconds since the first event, ``drift`` in keV).
    t_first : float
        Unix time of the first event, to anchor the window in absolute time.
    """
    stats = computed["stats"]
    # a TimedeltaIndex, so the excursion's longest_s really is seconds
    series = pd.Series(
        np.asarray(computed["drift"], dtype=float),
        index=pd.to_timedelta(stats["time"].to_numpy(dtype=float), unit="s"),
    )
    valid = series.dropna()
    if valid.empty:
        return
    worst = valid.iloc[int(np.argmax(np.abs(valid.to_numpy())))]
    start = pd.Timestamp(t_first, unit="s", tz="UTC")
    utils.issues.record_detail(
        period,
        run,
        datatype,
        ged,
        "FEP_gain_stab",
        observed=float(worst),
        threshold=[-2.0, 2.0],
        unit="keV",
        window=[str(start + valid.index[0]), str(start + valid.index[-1])],
        excursion=utils.issues.evaluate_excursion(series, -2.0, 2.0),
    )


_PEAKS = utils.EXPERIMENT["energy"]["peaks"]
_FIXED_THR = utils.EXPERIMENT["energy"]["escale_fixed_threshold_kev"]
_ERR_MULT = utils.EXPERIMENT["energy"]["escale_error_multiplier"]

ESCALE_METRICS = {
    # metric -> (parameter, peak energy, fixed threshold, error multiplier,
    #            one-sided). Peaks and band widths come from
    #            settings/experiment.yaml; resolution is graded one-sided, since
    #            a detector whose FWHM improved is not a problem.
    "escale_fwhm_FEP": (
        "fwhms_peaks",
        _PEAKS["fep"]["energy_kev"],
        None,
        _ERR_MULT,
        True,
    ),
    "escale_fwhm_583": (
        "fwhms_peaks",
        _PEAKS["low"]["energy_kev"],
        None,
        _ERR_MULT,
        True,
    ),
    "escale_FEP_pos": (
        "mus_keV_first_cal_peaks",
        _PEAKS["fep"]["energy_kev"],
        _FIXED_THR,
        None,
        False,
    ),
    "escale_SEP_residual": (
        "residuals",
        _PEAKS["sep"]["energy_kev"],
        _FIXED_THR,
        None,
        False,
    ),
}
_ESCALE_ERR_FIELD = {"fwhms_peaks": "fwhms_err_peaks"}


def evaluate_escale_metrics(
    det_name: str,
    det_results: dict,
    usability: dict,
    period: str,
    current_run: str,
) -> dict:
    """
    Evaluate the four energy-scale metrics behind the qcp verdicts.

    Reproduces the numbers the legacy figure computed while drawing: for each
    metric, the mean over every run where the detector is usable ("on"), a
    band of either a fixed width or a multiple of the mean fit error around
    it, and whether the current run's value falls inside. Resolution metrics
    use only the upper half of that band (see ESCALE_METRICS): an improved
    FWHM is not an issue. The band magnitudes are stashed via
    ``issues.record_detail`` for the issue records.

    Parameters
    ----------
    det_name : str
        Detector under evaluation.
    det_results : dict
        This detector's :func:`get_partitions_params` entry.
    usability : dict
        period-run -> 'on'/'ac'/'off' for this detector.
    period, current_run : str
        Run being evaluated (``<period>-<current_run>`` is the target key).

    Returns
    -------
    verdicts: dict
        metric -> True (in band) / False (outside) / None (not evaluable).
    """
    all_keys = sorted(usability.keys())
    target = f"{period}-{current_run}"
    on_mask = np.array([usability.get(k) == "on" for k in all_keys])
    verdicts = {}
    for metric, (
        parameter,
        peak,
        fixed_thr,
        err_thr,
        one_sided,
    ) in ESCALE_METRICS.items():
        entry = det_results.get(parameter, {}).get(peak, {})
        vals = np.array([float(entry.get(k, np.nan)) for k in all_keys])
        err_field = _ESCALE_ERR_FIELD.get(parameter)
        errs = None
        if err_field is not None:
            err_entry = det_results.get(err_field, {}).get(peak, {})
            errs = np.array([float(err_entry.get(k, np.nan)) for k in all_keys])

        verdicts[metric] = None
        valid = ~np.isnan(vals)
        good = valid & on_mask
        if not good.any() or target not in all_keys:
            continue
        mean = np.nanmean(vals[good])
        if fixed_thr is not None:
            lower, upper = mean - fixed_thr, mean + fixed_thr
        elif errs is not None:
            mean_err = np.nanmean(errs[good])
            lower, upper = mean - err_thr * mean_err, mean + err_thr * mean_err
        else:
            continue
        val = vals[all_keys.index(target)]
        if one_sided:
            lower = None
        ok = bool(val <= upper if one_sided else lower <= val <= upper)
        # a NaN target counts as out of band either way
        verdicts[metric] = ok
        if not ok:
            utils.issues.record_detail(
                period,
                current_run,
                "cal",
                det_name,
                metric,
                observed=float(val),
                threshold=[None if lower is None else float(lower), float(upper)],
                unit="keV",
                reference=float(mean),
            )
    return verdicts


def write_escale_summary(
    output_folder: str,
    period: str,
    run: str,
    partitions_params: dict,
    data_type: str = "cal",
) -> str | None:
    """
    Write the per-detector multi-run energy-scale arrays behind the escale figures.

    Flattens ``get_partitions_params`` output (det -> parameter [-> peak]
    -> period-run -> scalar) into one long frame, so every panel of the
    escale figure can be re-drawn (or re-checked) without unpickling it.
    Calibration polynomial coefficients are expanded as ``cal_params_c<i>``.

    Parameters
    ----------
    output_folder : str
        Monitoring output root (the folder containing ``<period>/``).
    period, run : str
        Run the summary is written under (data covers all runs in the file).
    partitions_params : dict
        Output of :func:`get_partitions_params`.
    data_type : str
        Data type key of the period contract file.

    Returns
    -------
    key: str or None
        The key written, or None when nothing was flattened.
    """
    rows = []
    for detector, params in (partitions_params or {}).items():
        for parameter, entry in params.items():
            if parameter == "cal_params":
                for period_run, coefficients in entry.items():
                    for i, value in enumerate(coefficients or []):
                        rows.append(
                            {
                                "detector": detector,
                                "parameter": f"cal_params_c{i}",
                                "peak": "",
                                "period_run": period_run,
                                "value": float(value),
                            }
                        )
                continue
            for level_key, level_value in entry.items():
                if isinstance(level_value, dict):  # peak-resolved parameter
                    for period_run, value in level_value.items():
                        rows.append(
                            {
                                "detector": detector,
                                "parameter": parameter,
                                "peak": str(level_key),
                                "period_run": period_run,
                                "value": float(value),
                            }
                        )
                elif isinstance(level_value, (int, float, np.floating, np.integer)):
                    rows.append(
                        {
                            "detector": detector,
                            "parameter": parameter,
                            "peak": "",
                            "period_run": str(level_key),
                            "value": float(level_value),
                        }
                    )
    if not rows:
        return None
    path = monitoring.period_contract_path(output_folder, period, data_type)
    return contract_writer.write_frame(path, f"escale/{run}", pd.DataFrame(rows))


def write_fep_gain_contract(
    output_folder: str, period: str, run: str, fep_stats: dict, data_type: str = "cal"
) -> str | None:
    """Write per-detector FEP gain stability into the period contract file.

    The same numbers the FEP figure is drawn from, in a form that can be read
    without unpickling a matplotlib figure (see contract/reader.read_frame).
    """
    rows = []
    for detector, computed in fep_stats.items():
        if not computed:
            continue
        stats = computed["stats"]
        drift = computed["drift"]
        for i, bin_row in stats.iterrows():
            rows.append(
                {
                    "detector": detector,
                    "run": run,
                    "time_s": float(bin_row["time"]),
                    "mean": float(bin_row["mean"]),
                    "std": float(bin_row["std"]),
                    "count": int(bin_row["count"]),
                    "drift_kev": (
                        float(drift.iloc[i]) if drift is not None else float("nan")
                    ),
                }
            )
    if not rows:
        return None

    file_path = os.path.join(
        output_folder, period, f"l200-{period}-{data_type}-monitoring.hdf"
    )
    key = f"fep_gain_stab/{run}"
    contract_writer.write_frame(file_path, key, pd.DataFrame(rows))
    utils.logger.debug("...wrote %s to %s", key, file_path)
    return file_path


def check_calibration_lac_ssc(
    tmp_auto_dir: str,
    output_folder: str,
    period: str,
    run: str,
    run_to_apply: str,
    first_run: bool,
    det_info: dict,
    data_type="cal",
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
    """
    detectors = det_info["detectors"]
    usability_map_file = os.path.join(
        output_folder, period, run, f"l200-{period}-{run}-qcp_summary.yaml"
    )
    output = utils.load_yaml_or_default(usability_map_file, detectors)
    fep_mean_results = {}
    fep_stats = {}

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
    os.makedirs(os.path.join(output_folder, period, run, "mtg"), exist_ok=True)
    utils.logger.debug("...inspecting FEP, calib peaks, stability in calibrations")

    # estimator column and peak windows (settings/experiment.yaml)
    energy_column = utils.EXPERIMENT["energy"]["estimators"][-1]
    fep_low, fep_high = utils.EXPERIMENT["energy"]["fep_window_kev"]
    fep_search = utils.EXPERIMENT["energy"]["peaks"]["fep"]["search_kev"]
    low_search = utils.EXPERIMENT["energy"]["peaks"]["low"]["search_kev"]

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
    fep_stats = {}

    available_channels = set(lh5.ls(hit_files[0], ""))

    for ged, item in detectors.items():
        if not item["processable"]:
            continue

        # avoid cases where the detector is not present in the output files
        if item["channel_str"] not in available_channels:
            continue

        hit_files_data = read_channel_events(
            hit_files,
            item["channel_str"],
            [energy_column, "timestamp", "is_valid_cal"],
        )
        if hit_files_data is None:
            continue

        mask = (
            hit_files_data.is_valid_cal
            & (hit_files_data[energy_column] > fep_low)
            & (hit_files_data[energy_column] < fep_high)
        )
        timestamps = hit_files_data[mask].timestamp.to_numpy()
        if timestamps.size == 0:
            continue
        t_first = float(timestamps[0])
        timestamps -= timestamps[0]
        energies = hit_files_data[mask][energy_column].to_numpy()

        fep_mean_results[ged], fep_stats[ged] = fep_gain_variation(
            period,
            run,
            pars=pars[ged],
            chmap=item,
            timestamps=timestamps,
            values=energies,
        )

        # build summary in memory
        ecal_results = pars[ged]["results"]["ecal"]
        ecal = monitoring.get_energy_key(
            ecal_results
        )  # check for cuspEmax_ctc_runcal or cuspEmax_ctc_cal
        pk_fits = monitoring.get_energy_key(ecal_results).get("pk_fits", {})

        # find FEP and low-E peaks (keys digits changed in the past, so let's be generic)
        fep_peaks = [p for p in pk_fits if fep_search[0] < p < fep_search[1]]
        low_peaks = [p for p in pk_fits if low_search[0] < p < low_search[1]]

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
        utils.update_evaluation_in_memory(output, ged, data_type, "fwhm_ok", fwhm_ok)

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
        if not stable and fep_stats.get(ged):
            record_fep_detail(period, run, data_type, ged, fep_stats[ged], t_first)

    write_fep_gain_contract(output_folder, period, run, fep_stats, data_type=data_type)

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
    )

    with open(usability_map_file, "w") as f:
        yaml.dump(output, f)
