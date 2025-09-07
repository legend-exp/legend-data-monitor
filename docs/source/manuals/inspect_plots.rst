How to inspect plots
====================

Output files
------------

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
                │                ├── <time_selection>/
                │                │   ├── <experiment>-<period>-<time_selection>-<type>-geds.hdf
                │                │   └── mtg/
                │                │       └── <parameter>/
                │                │           ├── <experiment>-<period>-<time_selection>-<type>-<parameter>.{bak,dat,dir}
                │                │           └── <pdf>/
                │                │               ├── st1/
                │                │               ├── st2/
                │                │               ├── st3/
                │                │               └── ...
                │                └── mtg/
                │                    ├── <experiment>-<period>-<type>-monitoring.{bak,dat,dir}
                │                    └── <pdf>/
                │                        ├── st1/
                │                        ├── st2/
                │                        ├── st3/
                │                        └── ...
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

Inspect plots
-------------

- Some standard plots to monitor detectors' response can be found online on the `LEGEND Dashboard <https://legend-exp.atlassian.net/wiki/spaces/LEGEND/pages/637861889/Monitoring+Dashboard+Manual>`_
- Some notebooks to interactively inspect plots can be found under the ``notebook`` folder
