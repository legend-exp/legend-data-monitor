import argparse
from pathlib import Path

from make_dashboard import make_excel, make_qcp_sheet, make_qcp_sheets_detailed
from read_qcp import ROOT_DIR, get_qcp_data
from read_usability import get_usability_data

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROD_CYCLE = "auto/latest"
DATA_BASE = (
    Path("/global/cfs/cdirs/m2676/data/lngs/l200/public/prodenv/prod-blind")
    / PROD_CYCLE
    / "generated/tier/dsp"
)
OUTPUT_BASE = ROOT_DIR / PROD_CYCLE

# ---------------------------------------------------------------------------
# {string_number: [(ged_name, mass_g), ...]}  top-to-bottom within each string
# ---------------------------------------------------------------------------

STRINGS = {
    1: [("V14654A", 3383), ("V14673A", 3450), ("V13044A", 3809)],
    4: [
        ("B00032C", 743),
        ("V07302A", 1803),
        ("V00050A", 1881.1),
        ("V05267A", 2183),
        ("V00048A", 1918.9),
        ("V00048B", 1815.8),
        ("V01240A", 2100),
        ("V05268A", 2298),
        ("V05261A", 1797),
    ],
    5: [
        ("V03421A", 2700),
        ("V03422A", 3418),
        ("V10447B", 3650),
        ("V08682A", 3340),
        ("V09724A", 2643),
        ("V11924A", 3596),
        ("V09374A", 2648),
    ],
    6: [
        ("V02160A", 1750),
        ("V02162B", 2480),
        ("V05612A", 2201),
        ("V05266A", 2073),
        ("V07647A", 1893),
        ("V07647B", 1779),
        ("V07302B", 1592),
        ("V05266B", 1988),
    ],
    7: [("V14618A", 2983), ("V13049A", 3756), ("V13046A", 3497)],
    8: [
        ("B00000C", 815),
        ("B00061B", 751),
        ("B00076C", 824),
        ("B00079B", 736),
        ("B00079C", 812),
        ("V06649M", 2532),
        ("V01404A", 1548.5),
    ],
    10: [
        ("B00000D", 813),
        ("B00002C", 788),
        ("B00035A", 768),
        ("B00035B", 810),
        ("V00050B", 1928.7),
        ("V04549A", 1943),
        ("V07298B", 2085),
        ("V06643A", 2286.2),
        ("V00074A", 2073),
    ],
    11: [
        ("V08682B", 1517),
        ("V11925A", 3668),
        ("V11947B", 3712),
        ("V09372A", 4046),
        ("V10784A", 3691),
        ("V10437B", 3758),
    ],
    12: [
        ("V02160B", 1719),
        ("V04199A", 2987),
        ("V04545A", 3138),
        ("V05267B", 2362),
        ("V07646A", 2630),
        ("V02166B", 2634),
        ("V05261B", 2393),
        ("V06659A", 2587.6),
    ],
}

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

    runlists = yaml.safe_load(open(datasets_path / "runlists.yaml"))
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

    runlists = yaml.safe_load(open(datasets_path / "runlists.yaml"))
    runs_key = runlists[key]
    periods = sorted(list(runs_key["cal"].keys()))
    first_run = expand_run_list(runs_key["cal"][periods[0]])[0]

    runinfo = yaml.safe_load(open(datasets_path / "runinfo.yaml"))
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


def build_periods(period: str) -> dict[str, list[tuple[str, str]]]:
    """
    Discover all runs for *period* by scanning the DSP tier directories.

    Returns {period: [(run_type, run), ...]} as:
      cal r000, phy r000, cal r001, phy r001, ..., cal rN
    phy is only added when phy DSP data actually exists for that run.
    """
    cal_dir = DATA_BASE / "cal" / period
    phy_dir = DATA_BASE / "phy" / period

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the LEGEND usability dashboard for one period."
    )
    parser.add_argument(
        "-period",
        required=True,
        metavar="pXX",
        help="Period to process, e.g. p16",
    )
    parser.add_argument(
        "-output",
        default=None,
        metavar="DIR",
        help="Directory to write sheet_{period}.xlsx into. "
        "Defaults to the standard output tree under ROOT_DIR.",
    )
    # TODO: add args for experts
    # - sync usability sheet with datasets for git commits
    # -

    args = parser.parse_args()
    period = args.period

    periods = build_periods(period)
    usability = get_usability_data(STRINGS, periods)

    if args.output is not None:
        output_dir = Path(args.output)
    else:
        output_dir = OUTPUT_BASE / "generated" / "plt" / "hit" / "phy" / period
    # output_dir.mkdir(parents=True, exist_ok=True)
    # (OUTPUT_BASE / "generated" / "tmp").mkdir(parents=True, exist_ok=True)

    output_path = str(output_dir / f"l200-{period}-auto_latest-qcp_summary.xlsx")

    make_excel(STRINGS, periods, usability, output_path)

    qcp_data = get_qcp_data(periods)
    make_qcp_sheet(output_path, STRINGS, periods, qcp_data)
    make_qcp_sheets_detailed(output_path, STRINGS, periods, qcp_data)


if __name__ == "__main__":
    main()
