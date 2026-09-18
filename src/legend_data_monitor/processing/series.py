"""Plot-free helpers for loading and manipulating monitoring time series."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

from .. import utils

IGNORE_KEYS = utils.IGNORE_KEYS


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


def find_hdf_file(
    directory: str, include: list[str], exclude: list[str] | None = None
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
    candidates = [
        f
        for f in sorted(p.name for p in Path(directory).iterdir())
        if f.endswith(".hdf")
        and all(tag in f for tag in include)
        and not any(tag in f for tag in exclude)
    ]

    return str(Path(directory) / candidates[0]) if candidates else None


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
    geds_df_abs = []
    geds_df_abs_corr = []
    puls_df_abs = []

    base_dir = Path(phy_mtg_data) / period
    runs = [p.name for p in base_dir.iterdir()]
    runs = [r for r in runs if re.fullmatch(r"r\d{3}", r)]

    for r in runs:
        if r not in run_list:
            continue
        run_dir = base_dir / r

        # geds file
        hdf_geds = find_hdf_file(run_dir, include=["geds"], exclude=["res", "min"])
        if hdf_geds:
            geds_abs = read_if_key_exists(hdf_geds, f"IsPulser_{parameter}")
            if geds_abs is not None:
                geds_df_abs.append(geds_abs)

            geds_puls_abs = read_if_key_exists(
                hdf_geds, f"IsPulser_{parameter}_pulser01anaDiff"
            )
            if geds_puls_abs is not None:
                geds_df_abs_corr.append(geds_puls_abs)
        else:
            utils.logger.debug("...hdf_geds missing in %s", r)

        # pulser file
        hdf_puls = find_hdf_file(
            run_dir, include=["pulser01ana"], exclude=["res", "min"]
        )
        if hdf_puls:
            puls_abs = read_if_key_exists(hdf_puls, f"IsPulser_{parameter}")
            if puls_abs is not None:
                puls_df_abs.append(puls_abs)
        else:
            utils.logger.debug("...hdf_puls missing in %s", r)

    if not geds_df_abs and not geds_df_abs_corr and not puls_df_abs:
        return None, None, None
    else:
        return (
            (
                pd.concat(geds_df_abs, ignore_index=False, axis=0)
                if geds_df_abs
                else pd.DataFrame()
            ),
            (
                pd.concat(geds_df_abs_corr, ignore_index=False, axis=0)
                if geds_df_abs_corr
                else pd.DataFrame()
            ),
            (
                pd.concat(puls_df_abs, ignore_index=False, axis=0)
                if puls_df_abs
                else pd.DataFrame()
            ),
        )


def get_spike_veto_series(phy_mtg_data: str, period: str, run_list: list):
    """
    Load the PULS01ANA series the pulser spike veto is applied on.

    The veto parameter and its acceptance window live in
    ``settings/experiment.yaml``; the series is the pulser file's
    ``IsPulser_<parameter>`` pivot, i.e. exactly what :func:`get_dfs` returns
    for that parameter.

    Parameters
    ----------
    phy_mtg_data : str
        Path to the base directory containing monitoring HDF5 files.
    period : str
        Period to inspect.
    run_list : list
        List of available runs.

    Returns
    -------
    pandas.DataFrame
        The pulser pivot, empty when the parameter was never written.
    """
    parameter = utils.EXPERIMENT["pulser_spike_veto"]["parameter"]
    key = utils.convert_to_camel_case(parameter, "_")
    _, _, puls_df = get_dfs(phy_mtg_data, period, run_list, key)
    return puls_df if puls_df is not None else pd.DataFrame()


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
    geds_abs: pd.DataFrame,
    channel: str,
    escale: float,
    puls_abs: pd.DataFrame | None = None,
    spike_veto: pd.DataFrame | None = None,
    variations=False,
) -> dict:
    """
    Return resampled geds and pulser series, and the pulser-corrected geds one.

    Parameters
    ----------
    resampling_time : str
        Resampling time, eg '1h' or '10min'.
    period : str | list
        Period or list of periods to inspect.
    geds_abs : pandas.DataFrame
        Absolute values of the inspected parameter, geds x time.
    channel : str
        Channel to inspect.
    escale : float
        Scaling factor used to compute relative differences in gain and calibration constant.
    puls_abs : pandas.DataFrame, optional
        The same parameter for PULS01ANA (single column); without it no
        pulser-corrected series is produced.
    spike_veto : pandas.DataFrame, optional
        PULS01ANA series the spike veto is applied on (see
        :func:`get_spike_veto_series`); without it no spike veto is applied.
    variations : bool
        True if you want to retrieve % variations (default: False).

    Returns
    -------
    dict
        ``ged`` / ``pul`` / ``diff``, each with ``raw``, ``rawdiff``,
        ``kevdiff``, ``kevdiff_av`` and ``kevdiff_std``.
    """
    # geds
    ser_ged = geds_abs[channel].sort_index()
    ser_ged = filter_by_period(ser_ged, period)
    ser_ged = ser_ged[~ser_ged.index.duplicated(keep="first")]  # remove duplicates
    ser_veto_kept = pd.DataFrame()

    if ser_ged.empty:
        utils.logger.debug("...geds series is empty after filtering")
        return None

    # drop pulser spikes, if the veto series was loaded
    if isinstance(spike_veto, pd.DataFrame) and not spike_veto.empty:
        ser_veto = spike_veto.iloc[:, 0].sort_index()  # PULS01ANA: one column
        ser_veto = filter_by_period(ser_veto, period)
        ser_veto = ser_veto[~ser_veto.index.duplicated(keep="first")]

        low_lim, upp_lim = utils.EXPERIMENT["pulser_spike_veto"]["window"]
        ser_veto_kept = ser_veto[(ser_veto > low_lim) & (ser_veto < upp_lim)]

        if not ser_veto_kept.empty:
            valid_idx = ser_ged.index.intersection(ser_veto_kept.index)
            ser_ged = ser_ged.reindex(valid_idx)

    # if before, potential mismatches with the veto series
    ser_ged = ser_ged.dropna()
    # compute average over the first 10% of elements
    n_elements = max(int(len(ser_ged) * 0.10), 1)
    ged_av = np.nanmean(ser_ged.iloc[:n_elements])
    if np.isnan(ged_av):
        utils.logger.debug("...the geds average is NaN")
        return None

    ser_ged_diff, ser_ged_diff_kev = compute_diff_and_rescaling(
        ser_ged, ged_av, escale, variations
    )

    # hour counts masking
    mask = ser_ged.resample(resampling_time).count() > 0

    # resample geds series
    ged_hr_av, ged_hr_std = resample_series(ser_ged_diff_kev, resampling_time, mask)
    ged_index = ged_hr_av.index

    # pulser series
    ser_pul = ser_pul_diff = ser_pul_diff_kev = pul_hr_av = pul_hr_std = None
    ged_corr = ged_corr_kev = ged_cor_hr_av = ged_cor_hr_std = None
    # ...if pulser is available:
    if isinstance(puls_abs, pd.DataFrame) and not puls_abs.empty:
        ser_pul = puls_abs.iloc[:, 0].sort_index()  # PULS01ANA: one column
        ser_pul = ser_pul[~ser_pul.index.duplicated(keep="first")]  # remove duplicates
        ser_pul = filter_by_period(ser_pul, period)

        # pulser average and diffs
        if not ser_pul.empty:
            if not ser_veto_kept.empty:
                valid_idx = ser_pul.index.intersection(ser_veto_kept.index)
                ser_pul = ser_pul.reindex(valid_idx)

            # if before, potential mismatches with the veto series
            ser_pul = ser_pul.dropna()
            n_elements_pul = max(int(len(ser_pul) * 0.10), 1)
            pul_av = np.nanmean(ser_pul.iloc[:n_elements_pul])
            ser_pul_diff, ser_pul_diff_kev = compute_diff_and_rescaling(
                ser_pul, pul_av, escale, variations
            )

            pul_hr_av, pul_hr_std = resample_series(
                ser_pul_diff_kev, resampling_time, mask
            )
            pul_hr_av = pul_hr_av.reindex(ged_index)
            pul_hr_std = pul_hr_std.reindex(ged_index)

            # corrected GED
            common_index = ser_ged_diff.index.intersection(ser_pul_diff.index)
            ged_corr = ser_ged_diff[common_index] - ser_pul_diff[common_index]
            ged_corr_kev = ged_corr * escale
            ged_cor_hr_av, ged_cor_hr_std = resample_series(
                ged_corr_kev, resampling_time, mask
            )
            ged_cor_hr_av = ged_cor_hr_av.reindex(ged_index)
            ged_cor_hr_std = ged_cor_hr_std.reindex(ged_index)

    return {
        "ged": {
            "raw": ser_ged,
            "rawdiff": ser_ged_diff,
            "kevdiff": ser_ged_diff_kev,
            "kevdiff_av": ged_hr_av,
            "kevdiff_std": ged_hr_std,
        },
        "pul": {
            "raw": ser_pul,
            "rawdiff": ser_pul_diff,
            "kevdiff": ser_pul_diff_kev,
            "kevdiff_av": pul_hr_av,
            "kevdiff_std": pul_hr_std,
        },
        "diff": {
            "raw": None,
            "rawdiff": ged_corr,
            "kevdiff": ged_corr_kev,
            "kevdiff_av": ged_cor_hr_av,
            "kevdiff_std": ged_cor_hr_std,
        },
    }
