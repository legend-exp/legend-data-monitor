How to produce plots
====================

How to run legend-data-monitor
------------------------------
After the installation, a executable is available at ``~/.local/bin``.
To automatically generate plots, two different methods are available.
All methods rely on the existence of a config file containing the output folder (``output``)
where to store results, the ``dataset`` you want to inspect, and the ``subsystems`` (pulser, geds, spms)
you want to study and for which you want to load data. See next section for more details.

You can either run the code by importing the ``legend-data-monitor`` module:

.. code-block:: python

  import legend-data-monitor as ldm
  user_config = "path_to_config.yaml"
  ldm.control_plots(user_config)

Or run it by parsing to the executable the path to the config file:

.. code-block:: bash

  $ legend-data-monitor user_prod --config path_to_config.yaml

If you want to inspect bunches of data (e.g. useful to avoid the process to get killed
when loading lots of files), you can use

.. code-block:: bash

  $ legend-data-monitor user_bunch --config path_to_config.yaml --n_files N

where ``N`` specifies how many files you want to inspect together at each iteration e.g. ``N=40``.


.. warning::

  Use the ``user_prod`` command line interface for generating your own plots.
  ``auto_prod`` and ``user_rsync_prod`` were designed to be used during automatic data production, for generating monitoring plots on the fly for new processed data. For the moment, no documentation will be provided.


Configuration file
------------------
In the following, we describe the structure of the configuration file in detail.


Example config
~~~~~~~~~~~~~~
.. code-block:: yaml

    output: "<output_path>"  # output folder
    dataset:
      experiment: "L200"
      period: "p09"
      version: "tmp-auto"
      path: "/data2/public/prodenv/prod-blind/"
      type: "phy"  # data type (either cal, phy, or ["cal", "phy"])
      start: "2023-02-07 02:00:00"  # time cut (here based on start+end)
      end: "2023-02-07 03:30:00"
    saving: "overwrite"
    subsystems:
      geds:  # type of subsystem to plot (geds, spms, pulser)
        "Baselines in pulser events":
          parameters: "baseline"
          event_type: "pulser"
          plot_structure: "per channel"
          plot_style: "vs time"
          variation: true
          resampled: "also"
          time_window: "1H"
          status: true


The argument ``output`` is the path where plots and inspected data will be saved.
In particular, ``dataset`` settings are:

- ``experiment``: experiment name (e.g. ``L60`` or ``L200``);
- ``period``: format ``pXX``. Note that this entry is not mandatory when using `DataLoader <https://pygama.readthedocs.io/en/stable/api/pygama.flow.html#pygama.flow.data_loader.DataLoader>`_ as it finds the period of interest automatically based on the provided time selection (see below); the period is useful for output filenames;
- ``version``: version of production cycle, format ``vXX.YY`` or ``ref-vXX.YY.ZZ`` or ``tmp-auto``;
- ``path``: path to ``prod-ref`` folder, e.g. ``/data2/public/prodenv/prod-blind/`` on LNGS cluster or ``/global/cfs/cdirs/m2676/data/lngs/l200/public/prodenv/prod-blind/`` on NERSC cluster;
- ``type``: type of data, either physics (``phy``), calibration (``cal``), or both (``["phy", "cal"]``);
- ``selection``: time window to select data.

.. note::

  Time selection is based on:

  - ``'start': '2023-02-07 02:00:00', 'end': '2023-02-07 03:30:00'`` (start + end) in format ``YYYY-MM-DD hh:mm:ss``;
  - ``'timestamps': ['20230207T103123Z', '20230207T141123Z', ...]`` (list of keys) in format ``YYYYMMDDThhmmssZ``;
  - ``'window': '1d 2h 0m'`` (time window in the past from current time point) in format ``Xd Xh Xm`` for days, hours, minutes;
  - ``'runs': 1`` (one run) or ``'runs': [1, 2, 3]`` (list of runs) in integer format.


A ``saving`` option is available to either ``"overwrite"`` any already present output file (or create a new one if not present) or ``"append"`` new data to the previously obtained output files.

Then, ``subsystems`` can either be ``pulser``, ``geds`` or ``spms`` (note: SiPMs plots are not implemented yet, but DataLoader can load the respective data if needed).

For each subsystem to be plotted, specify

- ``"<some title>"``: the title of the plot you want to generate. eg. "Baselines in pulser events"
- ``parameters``: one or multiple parameters of interest to be plotted for this subsystem. In addition to any parameter present in ``lh5`` files, the following special parameters were implemented and are available for plotting (see provided examples below for more details on how to select these parameters):
    - ``"K_lines"``: events whose energy is contained within 1430 and 1575 keV (40K and 42K regions)
    - ``"FWHM"``: FWHM values for each channel
    - ``"wf_max_rel"``: relative difference between ``wf_max`` and baseline
    - ``"event_rate"``: event rate calculated in windows specified in the field ``"sampling"`` under ``plotting.parameters``
- ``"event_type"``: which events to plot. Choose among ``pulser`` (events flagged as pulser based on AUX channel), ``FCbsln`` (events flagged as FlashCam triggered baseline, i.e. flat events), ``muon`` (events flagged as in coincidence with a muon), ``phy`` (physical, i.e. non-pulser events), ``all`` (pulser + physical events), ``K_events`` (physical events with energies in [1430; 1575] keV)
- ``"plot_structure"``: plot arrangement. Choose among
    - ``per channel`` (geds): group plots by channel, i.e. each channel has its canvas
    - ``per cc4`` (geds): group plots by CC4, i.e. all channels belonging to the same CC4 are in the same canvas
    - ``per string`` (geds): group plots by string, i.e. all channels belonging to the same string are in the same canvas
    - ``array`` (geds): group all channels in the same canvas
    - ``per fiber`` (spms): group channels separating them into the inner barrel (IB) and outer barrel (OB), and put top/bottom channels of a given fiber together to look for correlations within the fiber and among neighbouring fibers
    - ``per barrel`` (spms): group channels separating them into top/bottom IB/OB
- ``"plot_style"``: plot style. Choose among
    - ``vs time``: plot parameter VS time (all timestamps), as well as resampled values in a time window specified in plot settings (see ``time_window``)
    - ``vs ch``: plot parameter VS channel ID
    - ``histogram``: plot distribution of given parameter
    - ``scatter``: plot all entries of a parameter with points
    - ``heatmap``: plot 2d histos, with time on x axis
- ``"variation"``: set it to ``True`` if you want % variation instead of absolute values for your parameter. Percentage variations are evaluated as: ``(param/mean - 1)*100``, where ``mean`` is the mean of the parameter under study evaluated over the first 10% of the time interval you specified in the ``dataset`` entry
- ``"resampled"``: set it to ``"also"`` if you want to plot resampled values for the parameter under study. Resampling is done using the ``"time_window"`` you specify. Possible values are:
    - ``"none"``: do not plot resampled values, i.e. plot only events for each saved timestamps
    - ``"only"``: plot only resampled values, i.e. averaged parameter values after computing an average in a time window equal to ``"time_window"``
    - ``"also"``: plot both resampled and not resampled values
- ``"time_window"``: resampling time (``T``=minutes, ``H``=hours, ``D``=days) used to print resampled values (useful to spot trends over time)
- ``"status"``: set it to ``True`` if you want to generate a GEDs status map for the subsystem and parameter under study. Before using this option, you first need to specify the limits you want to set as a low/high threshold for the parameter under study by adding the % or absolute threshold for the subsystem of interest in ``src/legend-data-monitor/settings/par-setting.yaml``.

.. warning::

  There is no event type selection ready for calibration data.
  This means you always have to use ``"event_type": "all"`` as long as the different event selections are not properly implemented for calibration data too.

..

More than one subsystem can be entered at once, for instance:

.. code-block:: yaml

    subsystems:
      pulser:
        "Pulser event rate":
          parameters: "event_rate"
          event_type: "pulser"
          plot_structure: "per channel"
          plot_style: "vs time"
          variation: false
          time_window: "1H"

        "AUX channel waveform maximum":
          parameters: "wf_max"
          event_type: "all"
          plot_structure: "per channel"
          plot_style: "histogram"
          variation: false

      geds:
        "Baselines in pulser events":
          parameters: "baseline"
          event_type: "pulser"
          plot_structure: "per channel"
          plot_style: "vs time"
          variation: true
          time_window: "1H"

..


Quality cuts
------------
Different methods were implemented to either apply or retrieve quality cuts (QC).

Apply QC
~~~~~~~~
If you are loading a parameter for GEDs channels and you want to apply one or multiple QC flags, you just specify it in the subsystem plot entry:

.. code-block:: yaml

  "subsystems": {
    "geds": {
      "Baselines in pulser events": {
        "parameters": "baseline",
        "event_type": "pulser",
        "plot_structure": "per channel",
        "plot_style": "vs time",
        "variation": true,
        "time_window": "1H",
        "cuts": ["is_valid_bl_slope"]
      }
    }
..

In the above example, you are loading the baseline for pulser events and applying the ``is_valid_bl_slope`` QC as well to remove events for which the baseline slope is not valid.
Any bitmask entry is automatically converted into a boolean entry based on the information stored in legend-metadata.

Retrieve QC
~~~~~~~~~~~
QC are not parameters like the baseline, energy, etc. so there is no purpose in plotting them as they are.
However, QC rates are of fundamental importance as well as distribution of QC classifiers.
Below, we show a way to retrieve all available QC flags and/or classifiers by selecting ``"parameters": "quality_cuts"``:

.. code-block:: yaml

    subsystems:
      geds:
        "Baselines in pulser events":
          parameters: "baseline"
          event_type: "pulser"
          plot_structure: "per channel"
          plot_style: "vs time"
          variation: true
          time_window: "1H"
          cuts:
            - "is_valid_bl_slope"

..

This will create a unique table with QC flags/classifiers as columns, with an entry for each hit in each GEDs detector.
Any bitmask entry is automatically converted into a boolean entry based on the information stored in legend-metadata.

.. warning::

  At the moment, there is no differentiation based on the detector type for the available QC flags/classifiers.
  In other words, to load all QC info we read at the flags/classifiers listed under a path of type ``<path_to_prod_blind>/ref-v2.1.5/inputs/dataprod/config/tier_hit/l200-p01-r%-T%-ICPC-hit_config.yaml``.
  If any of these listed flags/classifiers is not present for a given detector type (eg COAX), then all entries of the flag/classifier are set to ``False`` by default.
  Any difference will be better handled in the future.

..


Removal of pulser effects
-------------------------

When plotting GEDs events that coincide with an injected pulser trace, you might want to remove any effects related to the pulser system (e.g. noise).
To do this, you can configure the config file to adjust a parameter by subtracting or dividing it by the corresponding auxiliary pulser parameter. In the configuration file, you can specify whether to compute either a ratio or a difference relative to the original parameter param:

.. math::

   \text{param\_ratio} = \frac{\text{param}_\text{geds}}{\text{param}_\text{AUX}}

.. math::

   \text{param\_diff} = \text{param}_\text{geds} - \text{param}_\text{AUX}

In the config file, you just need to set either the key ``AUX_ratio`` or ``AUX_diff" to true (note: it's not possible to select both options at the same time):

.. code-block:: yaml

    subsystems:
      geds:
        "Baselines in pulser events":
          parameters: "baseline"
          event_type: "pulser"
          plot_structure: "per channel"
          plot_style: "vs time"
          variation: true
          AUX_ratio: true
          time_window: "1H"
          cuts:
            - "is_valid_bl_slope"

..


.. note::

  The AUX channel selected for retrieving :math:`\text{param}_\text{AUX}` is always ``PULSER01ANA=1027203`` (when available).

..

Special parameters
------------------
More attention must be paid to the following special parameters, for which a particular ``subsystem`` entry is required.

K lines
~~~~~~~
To plot events having energies within 1430 and 1575 keV (ie, around the 40K and 42K regions), grouping channels by string and selecting physical (=not-pulser) events, use

.. code-block:: yaml

    subsystems:
      geds:
        "K events":
          parameters: "K_events"
          event_type: "K_lines"
          plot_structure: "per string"
          plot_style: "scatter"

..


FWHM
~~~~
To plot FWHM values for each channel, grouping them by strings, selecting only pulser events, use

.. code-block:: yaml

    subsystems:
      geds:
        "FWHM in pulser events":
          parameters: "FWHM"
          event_type: "pulser"
          plot_structure: "array"
          plot_style: "vs ch"

..


Relative maximum of the waveform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
To plot the relative difference between ``wf_max`` and ``baseline``, use

.. code-block:: yaml

    subsystems:
      pulser:
        "Relative wf_max":
          parameters: "wf_max_rel"
          event_type: "pulser"  # or phy, all, ...
          plot_structure: "per channel"
          plot_style: "vs time"
          variation: true  # optional
          time_window: "5T"

..

Event rate
~~~~~~~~~~
To plot the event rate, by sampling over a period of time equal to ``<time_window>`` (``T``=minutes, ``H``=hours, ``D``=days), use:

.. code-block:: yaml

    subsystems:
      geds:
        "Event rate":
          parameters: "event_rate"
          event_type: "pulser"
          plot_structure: "per channel"
          plot_style: "vs time"
          resampled: "no"
          variation: false
          time_window: "5T"

..
