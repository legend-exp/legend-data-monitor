Available parameters
====================
| The following table shows which parameter are available (and tested) separately for each detector type.


.. list-table::
  :widths: 30 25 25 25
  :header-rows: 1

  * - Available parameters
    - HPGe
    - SiPM
    - ch000
  * - dsp variables
    - all
    - x
    - all
  * - hit variables
    - all
    - all
    - x
  * - ``event_rate``
    - ✓
    - ✓
    - ✓
  * - ``K_lines``
    - ✓
    - x
    - x
  * - ``FWHM``
    - ✓
    - x
    - x
  * - ``wf_max_rel``
    - ✓
    - x
    - ✓

.. warning::

  It has been found out that no muon signals were being recorded in the auxiliary channel MUON01 for periods p08 and p09 (up to r003 included).
  This means the present code is not able to flag the germanium events for which there was a muon crossing the experiment.
  In other words, the dataframe associated to the ``muon`` events here will be empty.
  Moreover, if you select ``phy`` entries, these will still contain muons since the cut over this does not work.


.. important::

  Special parameters are typically saved under ``src/legend-data-monitor/settings/special-parameters.yaml`` and carefully handled when loading data.
