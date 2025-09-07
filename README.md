# legend-data-monitor

![GitHub tag (latest by date)](https://img.shields.io/github/v/tag/legend-exp/legend-data-monitor?logo=git)
[![GitHub Workflow Status](https://img.shields.io/github/checks-status/legend-exp/legend-data-monitor/main?label=main%20branch&logo=github)](https://github.com/legend-exp/legend-data-monitor/actions)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Codecov](https://img.shields.io/codecov/c/github/legend-exp/legend-data-monitor?logo=codecov)](https://app.codecov.io/gh/legend-exp/legend-data-monitor)
![GitHub issues](https://img.shields.io/github/issues/legend-exp/legend-data-monitor?logo=github)
![GitHub pull requests](https://img.shields.io/github/issues-pr/legend-exp/legend-data-monitor?logo=github)
![License](https://img.shields.io/github/license/legend-exp/legend-data-monitor)
[![Read the Docs](https://img.shields.io/readthedocs/legend-data-monitor?logo=readthedocs)](https://legend-data-monitor.readthedocs.io)

`legend-data-monitor` is a Python toolkit for monitoring the LEGEND experiment, enabling **inspection, visualization, and quality control** of experimental data. The main features are:

- **dataframe setup**: build dataframes containing channel maps and status for subsystems like auxiliary, HPGe and SiPM channels.
- **parameter access**: retrieve parameters of interest from raw, DSP, hit tiers, or user-defined sources for a given dataset.
- **flexible inspection**: filter and inspect parameters by time intervals, specific runs, or keys.
- **status mapping**: plot channel status maps (ON/OFF/...), highlighting channels exceeding or falling below configurable thresholds.
- **automated monitoring workflows**: semi-offline and continuous performance checks.
- **visualization utilities**: generate time series plots, histograms, and other performance indicators for monitoring.
