"""
Reads escale-usability, PSD status, the reason field, and PSD notes from legend-datasets.

Main output is:
data dict keys: (string_num, ged_name, period, run, run_type, field)
        field = "E"      -> "on" | "ac" | "off" | None
        field = "P"      -> PSD status: "valid" | "present" | "missing" | None (complicated for various geds)
        field = "reason" -> str | None  — only at the cal/phy event where it was written
        field = "PSD_note" -> str | None  — formatted per-cut statuses, only at cal events
                                                                        where PSD changed (or first cal run of period)

PSD notes are:
  low_aoe: valid
  high_aoe: present
  lq: valid
  ann: missing (redundant now)
  coax_rt: missing (redundant now)
"""

import glob
import re
from pathlib import Path

import yaml
from dbetto import TextDB
from lgdo import lh5

# LEGEND_DATASETS = Path(__file__).parent / "legend-datasets"
LEGEND_DATASETS = "/global/cfs/cdirs/m2676/data/lngs/l200/public/prodenv/prod-blind/auto/latest/inputs/datasets/"

PSD_RANK = {"valid": 2, "present": 1, "missing": 0}
PSD_FROM_RANK = {2: "valid", 1: "present", 0: "missing"}

# ---------------------------------------------------------------------------
# PSD helpers
# ---------------------------------------------------------------------------


def _psd_status(ged_data: dict) -> str | None:
    """PSD status, or None if no PSD block."""
    psd = ged_data.get("psd")
    if not psd:
        return None
    is_bb_like = psd.get("is_bb_like", "")
    # chosen params vary for diff geds so select ones of interest
    cuts = [c.strip() for c in re.split(r"[&|,]", is_bb_like) if c.strip()]
    if not cuts:
        return None
    status = psd.get("status", {})
    ranks = [PSD_RANK[s] for c in cuts if (s := status.get(c)) in PSD_RANK]
    return PSD_FROM_RANK[min(ranks)] if ranks else None


def _psd_cut_statuses(ged_data: dict) -> dict | None:
    """Return the full status dict {cut: status}, or None if no PSD block."""
    psd = ged_data.get("psd")
    if not psd:
        return None
    return dict(psd.get("status", {})) or None


def _format_psd_note(cut_statuses: dict) -> str:
    """Format {cut: status} as a multi-line note string."""
    return "\n".join(f"{cut}: {status}" for cut, status in cut_statuses.items())


def _build_psd_note_map(
    statuses: TextDB,
    runinfo: dict,
    periods: dict,
) -> dict:
    """Build {(ged_name, cal_timestamp): note_str} for every cal run where PSD cut statuses changed since the previous cal run."""
    psd_note_map: dict = {}

    for period, cols in periods.items():
        if period not in runinfo:
            continue

        # Previous cal snapshot per detector: {ged: {cut: status}}
        prev_cuts: dict[str, dict] = {}

        for run_type, run in cols:
            if run_type != "cal":
                continue
            if run not in runinfo[period]:
                continue
            run_info = runinfo[period][run]
            if "cal" not in run_info:
                continue

            cal_start_key = run_info["cal"]["start_key"]
            snapshot = statuses.on(cal_start_key, system="cal")

            for ged, ged_data in snapshot.items():
                if not isinstance(ged_data, dict):
                    continue
                cuts = _psd_cut_statuses(ged_data)
                if cuts is None:
                    continue

                if cuts != prev_cuts.get(ged):
                    psd_note_map[(ged, cal_start_key)] = _format_psd_note(cuts)
                    prev_cuts[ged] = cuts

    return psd_note_map


def write_runinfo(legend_datasets_path, runinfo):
    datasets = Path(legend_datasets_path)
    with open(datasets / "runinfo.yaml", "w") as yamlfile:
        yaml.safe_dump(runinfo, yamlfile)


def data(typ, tier, run, period, prod_cycle="auto/latest", server="nersc"):
    if server == "nersc":
        return sorted(
            glob.glob(
                f"/global/cfs/cdirs/m2676/data/lngs/l200/public/prodenv/prod-blind/{prod_cycle}/generated/tier/{tier}/{typ}/{period}/{run}/*"
            )
        )
    elif server == "lngs":
        return sorted(
            glob.glob(
                f"/data2/public/prodenv/prod-blind/{prod_cycle}/generated/tier/{tier}/{typ}/{period}/{run}/*"
            )
        )


def get_run_start_timestamp(period, run, run_type):
    dsp_files = data(run_type, "dsp", run, period, prod_cycle="auto/latest")
    timestamp = dsp_files[0].split("-")[-2]
    return timestamp


def get_live_time(period, run):
    evt_files = data("phy", "evt", run, period, prod_cycle="auto/latest")
    evt_data = lh5.read_as(
        "evt",
        evt_files,
        library="ak",
        field_mask=[
            "trigger/is_forced",
            "trigger/timestamp",
            "coincident/puls",
            "geds/quality/is_not_bb_like/is_delayed_discharge",
        ],
    )
    pulser_mask = (
        ~evt_data.trigger.is_forced
        & evt_data.coincident.puls
        & ~evt_data.geds.quality.is_not_bb_like.is_delayed_discharge
    )
    pulser_data = evt_data[pulser_mask]
    live_time = len(pulser_data) * 20  # 0.05 Hz
    min_time = pulser_data.trigger.timestamp.to_numpy()[0]
    max_time = pulser_data.trigger.timestamp.to_numpy()[-1]
    return (live_time, min_time, max_time)


def correct_runinfo(legend_datasets_path, run_info, period, run):
    if period not in run_info.keys():
        run_info[period] = {}
    if run not in run_info[period].keys():
        run_info[period][run] = {}

    for key in ["cal", "fft", "pzc"]:
        if key not in run_info[period][run].keys():
            run_info[period][run][key] = {
                "start_key": get_run_start_timestamp(period, run, "cal")
            }

    if "phy" not in run_info[period][run].keys():
        if len(data("phy", "dsp", run, period, "auto/latest")) > 0:
            run_info[period][run]["phy"] = {
                "start_key": get_run_start_timestamp(period, run, "phy"),
                "livetime_in_s": get_live_time(period, run)[0],
            }
    write_runinfo(legend_datasets_path, run_info)

    return run_info


def _build_reason_map(datasets: Path, validity: list) -> dict:
    """Build {(ged_name, valid_from_timestamp): reason} for entries where a config file explicitly sets a non-empty reason."""
    reason_map: dict = {}
    statuses_dir = datasets / "statuses"

    for entry in validity:
        ts = entry.get("valid_from")
        if not ts:
            continue
        for filename in entry.get("apply", []):
            filepath = statuses_dir / filename
            if not filepath.exists():
                continue
            with open(filepath) as f:
                cfg = yaml.safe_load(f) or {}
            for ged, ged_data in cfg.items():
                if not isinstance(ged_data, dict):
                    continue
                reason = ged_data.get("reason")
                if reason:
                    reason_map.setdefault((ged, ts), reason)

    return reason_map


def get_usability_data(
    strings: dict,
    periods: dict,
    legend_datasets_path: Path = LEGEND_DATASETS,
    alter_mode: bool = False,
) -> dict:
    """
    Build the usability data dict for the Excel dashboard.

    Parameters
    ----------
    strings  : {string_num: [(ged_name, mass_g), ...]}
    periods  : {period: [(run_type, run), ...]}

    Returns
    -------
    dict keyed by (string_num, ged_name, period, run, run_type, field)
      field = "E"      → "on" | "ac" | "off" | None
      field = "P"      → "valid" | "present" | "missing" | None
      field = "reason" → str | None  (only where written)
      field = "PSD_note" → str | None  (only at cal runs where PSD changed)
    """
    datasets = Path(legend_datasets_path)
    statuses = TextDB(str(datasets / "statuses"))

    with open(datasets / "runinfo.yaml") as f:
        runinfo = yaml.safe_load(f)

    with open(datasets / "statuses" / "validity.yaml") as f:
        validity = yaml.safe_load(f)

    reason_map = _build_reason_map(datasets, validity)
    psd_note_map = _build_psd_note_map(statuses, runinfo, periods)

    ged_to_string = {
        ged: snum for snum, detectors in strings.items() for ged, _ in detectors
    }

    data = {}
    for period, cols in periods.items():
        for run_type, run in cols:
            if alter_mode:
                if period not in runinfo:
                    correct_runinfo(legend_datasets_path, runinfo, period, run)
                if run not in runinfo[period]:
                    # try set this in the run info
                    correct_runinfo(legend_datasets_path, runinfo, period, run)
                run_info = runinfo[period][run]
                if run_type not in run_info:
                    correct_runinfo(legend_datasets_path, runinfo, period, run)
                timestamp = run_info[run_type]["start_key"]
            else:
                timestamp = get_run_start_timestamp(period, run, run_type)
            snapshot = statuses.on(timestamp)

            for ged, string_num in ged_to_string.items():
                ged_data = snapshot.get(ged, {})

                data[(string_num, ged, period, run, run_type, "E")] = ged_data.get(
                    "usability"
                )
                data[(string_num, ged, period, run, run_type, "P")] = _psd_status(
                    ged_data
                )
                data[(string_num, ged, period, run, run_type, "reason")] = (
                    reason_map.get((ged, timestamp))
                )

                # PSD_note only populated for cal runs; None for phy
                data[(string_num, ged, period, run, run_type, "PSD_note")] = (
                    psd_note_map.get((ged, timestamp)) if run_type == "cal" else None
                )

    return data
