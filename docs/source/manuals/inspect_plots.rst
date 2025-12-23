How to inspect plots
====================

Output files (user_prod, user_bunch)
------------------------------------

After you run the code, hdf files containing retrieved data generated for the inspected parameters/subsystems are produced, together with a pdf file containing all the generated plots and a log file.
In particular, the last two items are created for each inspected subsystem (pulser, geds, spms).

Files are usually collected in the output folder specified in the ``output`` config entry.
Then, depending on the chosen dataset (``experiment``, ``period``, ``version``, ``type``, time selection),
different output folders can be created. In general, the output folder is structured as it follows:

.. code-block::

    <output_path>/
        └── <version>/
            └── generated/
                ├── plt/
                │    └── hit/
                │        └── <type>/
                │            └── <period>/
                │                └── <time_selection>/
                │                    └── <experiment>-<period>-<time_selection>-<type>-geds.hdf
                └── tmp/
                    └── mtg/
                        └── <period>/
                            └── <time_selection>/
                                ├── <experiment>-<period>-<time_selection>-<type>.pdf
                                └── <experiment>-<period>-<time_selection>-<type>.log


Output hdf files for ``geds`` have the following dictionary structure, where ``<param>`` is the name of one of the inspected parameters, ``<flag>`` is the event type, e.g. *IsPulser* or *IsBsln*:

- ``<flag>_<param>_info`` = some useful info
- ``<flag>_<param>`` = absolute values
- ``<flag>_<param>_mean`` = average over the first 10% of data (within the selected time window) of ``<flag>_<param>``
- ``<flag>_<param>_var`` = % variations of ``<param>`` wrt ``<flag>_<param>_mean``
- ``<flag>_<param>_pulser01anaRatio`` = ratio of absolute values ``<flag>_<param>`` with PULS01ANA absolute values
- ``<flag>_<param>_pulser01anaRatio_mean`` = average over the first 10% of data (within the selected time window) of ``<flag>_<param>_pulser01anaRatio``
- ``<flag>_<param>_pulser01anaRatio_var`` = % variations of ``<flag>_<param>_pulser01anaRatio`` wrt ``<flag>_<param>_pulser01anaRatio_mean``
- ``<flag>_<param>_pulser01anaDiff`` = difference of absolute values ``<flag>_<param>`` with PULS01ANA absolute values
- ``<flag>_<param>_pulser01anaDiff_mean`` = average over the first 10% of data (within the selected time window) of ``<flag>_<param>_pulser01anaDiff``
- ``<flag>_<param>_pulser01anaDiff_var`` = % variations of ``<flag>_<param>_pulser01anaDiff`` wrt ``<flag>_<param>_pulser01anaDiff_mean``

.. note::

  For entries related to quality cut flags, we do not store any mean or percentage variation key.
  Moreover, no ratio or different with respect to AUX channels is performed.
  No plots (ie pdf files) are generated when loading quality cut flags.

Output files (auto_run)
-----------------------

When running the code via the executable ``auto_run``, the code will automatically produce copies of the original HDF file but resampling the time content by 10 min or 1 hour; these resampled files will speed up the loading step when plots will be uploaded on the Dashboard.
A YAML file for quick access on plotting info is also automatically produced in output.
Monitoring shelve (and pdf) period-based files will be stored under ``<path2>/<ref>/generated/plt/hit/phy/<period>/mtg/``.
Additional monitoring files produced for each run will be stored under ``<path2>/<ref>/generated/plt/hit/phy/<period>/<run>/mtg/``.

Monitoring plots are stored to reflect the period-based and run-base structure.
The structure will look like:

.. code-block:: text

   <output_folder>/
       └── <ref>/
           └── generated/
               ├── plt/
               │    └── hit/
               │        └── phy/
               │            └── <period>/
               │                ├── <run>/
               │                │   ├── l200-<period>-<run>-phy-geds.hdf
               │                │   ├── l200-<period>-<run>-phy-geds-info.yaml
               │                │   ├── l200-<period>-<run>-phy-geds-res_10min.hdf
               │                │   ├── l200-<period>-<run>-phy-geds-res_60min.hdf
               │                │   ├── l200-<period>-<run>-phy-slow_control.hdf
               │                │   └── mtg/
               │                │      ├── l200-<period>-<run>-phy-monitoring.{bak,dat,dir}
               │                │      └── <pdf>/
               │                │         ├── st1/
               │                │         ├── st2/
               │                │         ├── st3/
               │                │         └── ...
               │                └── mtg/
               │                    ├── l200-<period>-phy-monitoring.{bak,dat,dir}
               │                    └── <pdf>/
               │                        ├── st1/
               │                        ├── st2/
               │                        ├── st3/
               │                        └── ...
               └── tmp/
                   └── mtg/
                       └── <period>/
                           └── <run>/
                               ├── last_checked_timestamp.txt
                               ├── new_keys.filekeylist
                               ├── l200-<period>-<run>-phy.pdf
                               └── l200-<period>-<run>-phy.log

where ``<parameter>`` can be ``Baseline``, ``TrapemaxCtcCal``, etc.
The ``<pdf>/`` folders are created only if ``--pdf True``.


Inspect plots
-------------

- Some standard plots to monitor detectors' response can be found online on the `LEGEND Dashboard <https://legend-exp.atlassian.net/wiki/spaces/LEGEND/pages/637861889/Monitoring+Dashboard+Manual>`_
- Some notebooks to interactively inspect plots can be found under the ``notebook`` folder
