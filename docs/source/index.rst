Welcome to legend-data-monitor's documentation!
===============================================

*legend-data-monitor* is a Python package developed to inspect LEGEND data.
In particular, this tool helps:

* set up dataframe objects containing channel map and status for a given subsystems (pulser, geds, spms)
* get data for parameters (from raw/dsp/hit tiers or user defined ones) of interest based on a given dataset
* inspect parameters by providing either a time interval, a list of run(s) or key(s) to inspect
* plotting status maps (e.g., ON/OFF/...) for each channel, spotting those that are problematic when overcoming/undercoming given thresholds

Getting started
---------------
*legend-data-monitor* can be installed with `pip <https://pip.pypa.io/en/stable/getting-started>`_:

.. code-block::

   $ pip install legend-data-monitor@git+https://github.com/legend-exp/legend-data-monitor@main

.. attention::

   Before starting, make sure ``~/.local/bin`` (where you can find the legend-data-monitor executable) is appended to ``$PATH`` after ``legend-data-monitor`` has been installed via pip.

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2

   manuals/index
   Package API reference <api/modules>
