"""
Driver script for make_dashboard.py.

Adds a 'QCP Summary' sheet to an existing dashboard_output.xlsx.
Run this after create_dashboard.py has already produced the file.

Uses the same strings/periods as create_dashboard.py.
"""

from pathlib import Path

from create_dashboard import example_periods, example_strings
from make_dashboard import make_qcp_sheet
from read_qcp import get_qcp_data

OUTPUT = Path(__file__).parent / "dashboard_output.xlsx"

if __name__ == "__main__":
    qcp_data = get_qcp_data(example_periods)
    make_qcp_sheet(str(OUTPUT), example_strings, example_periods, qcp_data)
