"""
Reads QCP summary YAML files from monitoring/temp/.

Returns qcp_data[period][run][detector][section][check] = True | False | None
  section = "cal" or "phy"
  checks (cal) : FEP_gain_stab, fwhm_ok, npeak, const_stab, PSD
  checks (phy) : baseln_spike, baseln_stab, pulser_stab
"""

import os

import yaml


def read_qcp_summary(
    output: str, period: str, run: str, prod_cycle: str = "auto/latest"
) -> dict:
    """
    Read QCP summary YAML file.

    Parameters
    ----------
    output : str
        Root path to the auto/latest directory (e.g., /data2/public/prodenv/prod-blind/auto/latest)
    period : str
        Period identifier (e.g., "p16")
    run : str
        Run identifier (e.g., "r000")
    prod_cycle : str
        Production cycle subdirectory (default: "auto/latest")
    """
    file = os.path.join(output, run, f"l200-{period}-{run}-qcp_summary.yaml")
    if not os.path.exists(file):
        return None
        
    with open(file) as yaml_file:
        qcp_summary = yaml.safe_load(yaml_file)
    return qcp_summary


def get_qcp_data(output: str, periods: dict) -> dict:
    """
    Get QCP data for all periods and runs.

    Parameters
    ----------
    output : str
        Root path to the auto/latest directory
    periods : dict
        {period: [(run_type, run), ...]} structure
    """
    qcp_data: dict = {}
    for period in periods:
        seen: set = set()
        for _, run in periods[period]:
            seen.add(run)
            
        qcp_data[period] = {}
        for run in sorted(seen):
            qcp_data[period][run] = read_qcp_summary(output, period, run)
            
    return qcp_data
