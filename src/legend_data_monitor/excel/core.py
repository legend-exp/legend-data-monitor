import os
from pathlib import Path

import awkward as ak
from legendmeta import LegendMetadata

from .make_dashboard import make_excel, make_qcp_sheet, make_qcp_sheets_detailed
from .read_qcp import get_qcp_data
from .read_usability import get_usability_data

# ---------------------------------------------------------------------------
# For runlists/runinfo
# ---------------------------------------------------------------------------


def expand_run_list(value: list | str) -> list[str]:
    """Expand a YAML run value to a flat list of run strings."""
    if isinstance(value, list):
        result = []
        for item in value:
            s = str(item)
            if ".." in s:
                start, end = s.split("..")
                result.extend(
                    [f"r{n:03d}" for n in range(int(start[1:]), int(end[1:]) + 1)]
                )
            else:
                result.append(s)
        return result
    s = str(value)
    if ".." in s:
        start, end = s.split("..")
        return [f"r{n:03d}" for n in range(int(start[1:]), int(end[1:]) + 1)]
    return [s]


def get_periods(key: str, datasets_path: Path) -> dict[str, list[tuple[str, str]]]:
    import yaml

    runlists = yaml.safe_load(open(os.path.join(datasets_path, "runlists.yaml")))
    runs_key = runlists[key]
    periods: dict = {}
    for data_type, data_type_item in runs_key.items():
        for period, runs in data_type_item.items():
            run_list = expand_run_list(runs)
            if period not in periods:
                periods[period] = []
            for run in run_list:
                periods[period].append((data_type, run))
            periods[period] = sorted(periods[period], key=lambda x: x[1])
    return periods


def get_geds(key: str, datasets_path: Path) -> dict[int, list[tuple[str, float]]]:
    import yaml
    from legendmeta import LegendMetadata
    from read_usability import correct_runinfo

    runlists = yaml.safe_load(open(os.path.join(datasets_path, "runlists.yaml")))
    runs_key = runlists[key]
    periods = sorted(list(runs_key["cal"].keys()))
    first_run = expand_run_list(runs_key["cal"][periods[0]])[0]

    runinfo = yaml.safe_load(open(os.path.join(datasets_path, "runinfo.yaml")))
    if periods[0] not in runinfo:
        correct_runinfo(datasets_path, runinfo, periods[0], first_run)

    timestamp = runinfo[periods[0]][first_run]["cal"]["start_key"]
    meta = LegendMetadata()
    chmap = meta.channelmap(timestamp)

    strings: dict = {}
    for ged, item in chmap.items():
        if item["system"] != "geds":
            continue
        string = item["location"]["string"]
        position = item["location"]["position"]
        mass = item["production"]["mass_in_g"]
        strings.setdefault(string, []).append((ged, mass, position))

    for string in strings:
        strings[string] = sorted(strings[string], key=lambda x: x[2])
        strings[string] = [(ged, mass) for ged, mass, _ in strings[string]]
    return strings


# ---------------------------------------------------------------------------
# Run discovery - for auto/latest
# ---------------------------------------------------------------------------


def get_runs_for_a_period(
    auto_dir_path: str, period: str
) -> dict[str, list[tuple[str, str]]]:
    """
    Discover all runs for *period* by scanning the DSP tier directories.

    Returns {period: [(run_type, run), ...]} as:
      cal r000, phy r000, cal r001, phy r001, ..., cal rN
    phy is only added when phy DSP data actually exists for that run.
    """
    data_base = os.path.join(auto_dir_path, "generated/tier/dsp")
    cal_dir = Path(os.path.join(data_base, "cal", period))
    phy_dir = Path(os.path.join(data_base, "phy", period))

    if not cal_dir.exists():
        raise SystemExit(f"No cal DSP data found for {period} at {cal_dir}")

    cal_runs = sorted(p.name for p in cal_dir.iterdir() if p.is_dir())
    phy_runs = (
        {p.name for p in phy_dir.iterdir() if p.is_dir()} if phy_dir.exists() else set()
    )

    pairs: list[tuple[str, str]] = []
    for run in cal_runs:
        pairs.append(("cal", run))
        if run in phy_runs:
            pairs.append(("phy", run))

    return {period: pairs}


def generate_dashboard(auto_dir_path: str, period: str, output: str) -> None:
    """
    Generate the LEGEND usability dashboard for one period.

    Parameters
    ----------
    auto_dir_path : str
        Path to tmp-auto public data files (eg /data2/public/prodenv/prod-blind/tmp-auto).
    period: str
        Period to process, eg p16
    output: str
        Directory to write sheet_{period}.xlsx into
    """
    strings_info = get_strings_info(auto_dir_path, period)

    periods = get_runs_for_a_period(auto_dir_path, period)
    usability = get_usability_data(
        strings_info, periods, Path(os.path.join(auto_dir_path, "inputs/datasets"))
    )

    output_path = str(
        os.path.join(output, f"l200-{period}-auto_latest-qcp_summary.xlsx")
    )

    make_excel(strings_info, periods, usability, output_path)

    qcp_data = get_qcp_data(periods)
    make_qcp_sheet(output_path, strings_info, periods, qcp_data)
    make_qcp_sheets_detailed(output_path, strings_info, periods, qcp_data)


def get_strings_info(auto_dir_path: str, period: str) -> dict:
    """Get string info in the desired fashion fashion: {string_number: [(ged_name, mass_g), ...]}, top-to-bottom within each string."""
    strings_info = {}

    lmeta = LegendMetadata(os.path.join(auto_dir_path, "inputs"))
    chmap = lmeta.channelmap("20251211T120001Z")  # if start_key else lmeta.channelmap()
    chmap = ak.Array(chmap.group("system").geds.values())

    sorted_chmap = chmap[ak.argsort(chmap.location.position)]

    for det in sorted_chmap:
        string = int(det.location.string)
        name = str(det.name)
        mass = float(det.production.mass_in_g)

        if string not in strings_info:
            strings_info[string] = []

        strings_info[string].append((name, mass))

    return strings_info
