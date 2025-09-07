Settings Directory
==================

The directory ``src/legend_data_monitor/settings/`` contains YAML configuration files used to customize the behavior of the ``legend-data-monitor`` package.

flags.yaml
----------

- **Purpose:** Defines a conversion dictionary for flagging data entries corresponding to different event types in output files.

ignore-keys.yaml
----------------

- **Purpose:** Contains time ranges to exclude during data retrieval. The file contains one entry for each period.

- **Structure:**

  - ``start_keys``: list of starting cycles (inclusive) from which to remove data
  - ``stop_keys``: list of ending cycles (exclusive) up to which to remove data

no-pulser-dets.yaml
-------------------

- **Purpose:** List of periods where the pulser signal was not injected into a specific HPGe detector. Uses the detector name as the key.

par-settings.yaml
-----------------

- **Purpose:** Contains metadata for various inspected parameters. Uses the parameter name as the key.

- **Includes:**

  - Plotting labels
  - Units
  - Plot colors
  - Threshold limits

parameter-tiers.yaml
--------------------

- **Purpose:** Maps parameters to the data tiers they belong to.

remove-dets.yaml
----------------

- **Purpose:** Provides a list of detectors to be quickly excluded from analysis (i.e. turned OFF), typically used when status maps are not yet updated.

remove-keys.yaml
----------------

- **Purpose:** Specifies individual parameter keys to remove for a specific detector. Example: ``remove-keys-COAXp04.yaml`` contains keys to remove for some COAX detectors in p04.

SC-params.yaml
--------------

- **Purpose:** Contains configuration for retrieving Slow Control parameters. Includes:

  - Database table name
  - Column name
  - Entry name
  - Other relevant SC expressions for data access

special-parameters.yaml
-----------------------

- **Purpose:** Defines special composite parameters defined by the user and their composition. Lists parameter names and the underlying DSP/hit parameters they are based on.
