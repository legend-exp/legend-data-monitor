"""
Driver script for make_dashboard.py.

Includes examples for structure of the inputs.
"""

from pathlib import Path

import yaml
from make_dashboard import make_excel
from read_usability import get_usability_data

LEGEND_DATASETS = Path(__file__).parent / "legend-datasets"

# ---------------------------------------------------------------------------
# INPUT 1 — strings
# {string_number: [(ged_name, mass_g), ...]}
# Detectors in top-to-bottom display order within each string.
# ---------------------------------------------------------------------------
example_strings = {
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
# INPUT 2 — periods
# {period: [(run_type, run), ...]}
# Each tuple is one column in the sheet.
# Last run of each period is cal-only (no trailing "phy" entry).
# ---------------------------------------------------------------------------
example_periods = {
    "p16": [
        ("cal", "r000"),
        ("phy", "r000"),
        ("cal", "r001"),
        ("phy", "r001"),
        ("cal", "r002"),
        ("phy", "r002"),
        ("cal", "r003"),
        ("phy", "r003"),
        ("cal", "r004"),
        ("phy", "r004"),
        ("cal", "r005"),
        ("phy", "r005"),
        ("cal", "r006"),
    ],
    "p18": [
        ("cal", "r000"),
        ("phy", "r000"),
        ("cal", "r001"),
        ("phy", "r001"),
        ("cal", "r002"),
        ("phy", "r002"),
        ("cal", "r003"),
        ("phy", "r003"),
        ("cal", "r004"),
        ("phy", "r004"),
        ("cal", "r005"),
        ("phy", "r005"),
        ("cal", "r006"),
    ],
    "p19": [
        ("cal", "r000"),
        ("phy", "r000"),
        ("cal", "r001"),
        ("phy", "r001"),
        ("cal", "r002"),
        ("phy", "r002"),
        ("cal", "r003"),
        ("phy", "r003"),
        ("cal", "r004"),
        ("phy", "r004"),
        ("cal", "r005"),
        ("phy", "r005"),
        ("cal", "r006"),
    ],
}

if __name__ == "__main__":
    with open(LEGEND_DATASETS / "runinfo.yaml") as _f:
        _runinfo = yaml.safe_load(_f)

    data = get_usability_data(example_strings, example_periods)

    # livetimes are an optional thing for exposure related stuff
    livetimes = {
        (period, run): _runinfo[period][run]["phy"]["livetime_in_s"]
        for period in example_periods
        for _, run in example_periods[period]
        if (
            period in _runinfo
            and run in _runinfo[period]
            and "phy" in _runinfo[period][run]
            and "livetime_in_s" in _runinfo[period][run]["phy"]
        )
    }

    make_excel(
        example_strings,
        example_periods,
        data,
        output_path="dashboard_output.xlsx",
        livetimes=livetimes,
    )
