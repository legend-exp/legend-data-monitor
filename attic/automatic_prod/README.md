# Automatic generation of plots

This basic example file can be used to automatically generate monitoring plots, based on new .lh5 dsp/hit files appearing in the production folder (the processing version can be specified at input).

Before running it, be sure to have installed `legend-data-monitor` in your container (`legendexp_legend-base_latest.sif`).

## How to run it

To run the script, you have to parse different inputs - you can check them via `$ python main_sync_code.py --help`. For automatic generation of plots on lngs, use

```console
$ python main_sync_code.py --cluster nersc
                        --ref_version tmp-auto
                        --output_folder /global/cfs/cdirs/m2676/data/lngs/l200/public/prodenv/prod-blind/
                        --chunk_size 30
                        --pswd_email <password>
```

where

* `cluster` is either `lngs` or `nersc`, ie the name of the cluster where you are working; this already look for the production environment specific for the two clusters, ie `/global/cfs/cdirs/m2676/data/lngs/l200/public/prodenv/prod-blind/` on NERSC or `/data2/public/prodenv/prod-blind/` on lngs-login;
* `ref_version` is the version of processed data to inspect (eg. `tmp-auto` or `ref-v2.1.0`);
* `output_folder` is the path where to store the automatic results, ie plots and summary files;
* `pdf` is True if you want pdf monitoring files in output (default: `False`);
* `sc` is True if you want to retrieve Slow Control parameters (default: `False`);
* `pswd` is the password to access the Slow Control database (NOT available on NERSC); you can find the password on Confluence;
* `port` is the port necessary to retrieve the Slow Control database (default: `8282`);
* `pswd_email` is the password to access the legend.data.monitoring@gmail.com account for sending automatic alert messages (default: `None`);
* `chunk_size` is the maximum integer number of files to read at each loop in order to avoid the process to be killed (default: `20`);
* `p` is the period (eg p03) to inspect; if not specified, the code will retrieve the latest processed period for the specified processing version (default: `None`);
* `r` is the run (eg r000) to inspect; if not specified, the code will retrieve the latest processed run for the specified processing version and period (default: `None`);
* `escale` is the energy scale at which evaluating the gain differences (default: `2039`, ie 76-Ge Q-value).


## Output format

The generated plots are saved in the shelve format in order to directly upload the produced canvas on the Dashboard.
HDF files will be stored under `<path2>/<ref>/generated/plt/hit/phy/<period>/<run>`.
The code will automatically produce copies of the original HDF file but resampling the time content by 10 min or 1 hour; these resampled files will speed up the loading step when plots will be uploaded on the Dashboard.
A YAML file for quick access on plotting info is also automatically produced in output.
Monitoring shelve (and pdf) period-based files will be stored under `<path2>/<ref>/generated/plt/hit/phy/<period>/mtg/`.
Additional monitoring files produced for each run will be stored under `<path2>/<ref>/generated/plt/hit/phy/<period>/<run>/mtg/`.

Monitoring plots are stored to reflect the period-based and run-base structure.
The structure will look like:

```text
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
            │                │       └── <parameter>/
            │                │           ├── l200-<period>-phy-<parameter>.{bak,dat,dir}
            │                │           └── <pdf>/
            │                │               ├── st1/
            │                │               ├── st2/
            │                │               ├── st3/
            │                │               └── ...
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
```

where `<parameter>` can be `Baseline`, `TrapemaxCtcCal`, etc.
The `<pdf>/` folders are created only if `--pdf True`.




# Automatic running

## How to set up a cronjob

You can run this command as a cronejob. On terminal, type

```console
$ crontab -e
```

and add a new line in this file of the following type:

```console
0 */6 * * * python <absolute_path>/main_sync_code.py <parse_inputs>
```

This will automatically look for new processed .lh5 files every 6 hours for instance (parse all the necessary inputs when you run the main code).


## How to set up a bash script
If the crontab command is not available on your cluster, you can use a bash script (`automatic_run.sh`) to run an infinite `while true` loop:

```bash
#!/bin/bash
while true; do
  python <absolute_path>/main_sync_code.py <parse_inputs>
  echo "Running job at $(date)"
  sleep 3600  # every hour
done
```

Fix the absolute path to your `main_sync_code.py`, the inputs to parse to the code, the time (in seconds) every which you want to execute the python command (here set to 1 H).

In order to run the above script in a resilient way such that survives your logout or SSH disconnects, run it like this:

```console
$ nohup ./automatic_run.sh
```

If you want to stop the process, you have first to find the associated process ID (PID) with one of these commands:

```console
ps aux | grep automatic_run.sh
```

or more specifically:

```console
pgrep -f automatic_run.sh
```

This will return the process ID(s) (e.g., `12345`).

Once you have the PID, stop it like this:

```console
kill 12345
```

# Keys to ignore

An external file `inore-keys.yaml` with information about time ranges to remove from inspected data is retrieved from `../src/legend_data_monitor/settings/`.
The structure of this file is of the following type:

```yaml
{
    "p03": {
        "start_keys": ["20230327T145702Z", "20230406T135529Z"],
        "stop_keys": ["20230327T145751Z", "20230406T235540Z"]
    },
    ...
```

where for instance we remove all file keys falling in the time range from `20230327T145702Z` (included) to `20230327T145751Z` (excluded), and `20230406T135529Z` (included) to `20230406T235540Z` (excluded).

This option was implemented for a local running of the code in order to be able to quickly query specific bunches of data.
The final keys to ignore in the analysis will have to be stored in the LEGEND metadata.
