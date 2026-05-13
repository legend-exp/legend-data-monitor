"""
Reads QCP summary YAML files from monitoring/temp/.

Returns qcp_data[period][run][detector][section][check] = True | False | None
  section = "cal" or "phy"
  checks (cal) : FEP_gain_stab, fwhm_ok, npeak, const_stab, PSD
  checks (phy) : baseln_spike, baseln_stab, pulser_stab
"""

from pathlib import Path

import yaml

ROOT_DIR = Path(
    "/global/cfs/cdirs/m2676/users/calgaro/legend-data-monitor/monitoring/automatic_prod/dashboard/"
)


def read_qcp_summary(period, run, prod_cycle="auto/latest"):
    file = (
        ROOT_DIR
        / prod_cycle
        / "generated/plt/hit/phy"
        / period
        / run
        / f"l200-{period}-{run}-qcp_summary.yaml"
    )
    with open(file) as yaml_file:
        qcp_summary = yaml.safe_load(yaml_file)
    return qcp_summary


def get_qcp_data(periods: dict) -> dict:
    qcp_data: dict = {}
    for period in periods:
        seen: set = set()
        for _, run in periods[period]:
            seen.add(run)
        qcp_data[period] = {}
        for run in sorted(seen):
            qcp_data[period][run] = read_qcp_summary(period, run)
    return qcp_data
