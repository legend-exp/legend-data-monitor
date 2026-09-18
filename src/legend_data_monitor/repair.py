"""Regenerate one parameter's outputs for a run without re-running the pipeline.

When a producer bug corrupts a single parameter (the ``bl_mean`` pivots were
truncated to one row per chunk by a name heuristic in ``save_data.get_pivot``),
re-processing the whole run costs hours. This replays only the configuration
entries for that parameter over the run's recorded chunk lists into a scratch
tree, transplants the resulting keys into the run's v1 files, and refreshes
the matching contract keys in place.
"""

import glob
import importlib.resources
import os
import re
import shutil

import pandas as pd
import yaml

from . import errors, utils
from .contract import build as contract_build

PROD_ROOTS = {
    "lngs": "/data2/public/prodenv/prod-blind/",
    "nersc": "/global/cfs/cdirs/m2676/data/lngs/l200/public/prodenv/prod-blind/",
}


def chunk_lists(generated_path: str, period: str, run: str) -> list:
    """Return the run's chunk key lists in processing order (whole list if unsplit)."""
    mtg = os.path.join(generated_path, "generated", "tmp", "mtg", period, run)
    parts = glob.glob(os.path.join(mtg, "new_keys_part_*.filekeylist"))
    if parts:
        return sorted(parts, key=lambda p: int(re.findall(r"part_(\d+)", p)[0]))
    whole = os.path.join(mtg, "new_keys.filekeylist")
    return [whole] if os.path.exists(whole) else []


def config_entries(parameter: str, subsystem: str = "geds") -> dict:
    """Return the packaged plot-config entries that load ``parameter``."""
    pkg = importlib.resources.files("legend_data_monitor")
    with open(pkg / "settings" / f"{subsystem}-dict.yaml") as f:
        entries = yaml.load(f, Loader=yaml.CLoader)[subsystem]
    return {
        title: entry
        for title, entry in entries.items()
        if entry.get("parameters") == parameter
    }


def family_keys(path: str, camel: str) -> list:
    """Keys of one parameter family in a v1 file (``<Flag>_<Camel>[_suffix]``)."""
    with pd.HDFStore(path, "r") as store:
        keys = [k.lstrip("/") for k in store.keys()]
    return [k for k in keys if k.split("_")[1:2] == [camel]]


def transplant_keys(target: str, source: str, keys: list) -> None:
    """Replace ``keys`` in ``target`` with their ``source`` versions, atomically."""
    tmp = f"{target}.repack.{os.getpid()}"
    try:
        with pd.HDFStore(target, "r") as store:
            existing = [k.lstrip("/") for k in store.keys()]
        for key in existing:
            if key in keys:
                continue
            frame = pd.read_hdf(target, key=key)
            options = {} if key.endswith("_info") else utils.HDF_COMPRESSION
            frame.to_hdf(tmp, key=key, mode="a", **options)
        for key in keys:
            frame = pd.read_hdf(source, key=key)
            options = {} if key.endswith("_info") else utils.HDF_COMPRESSION
            frame.to_hdf(tmp, key=key, mode="a", **options)
        os.replace(tmp, target)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def repair_parameter(
    output_folder: str,
    ref_version: str,
    period: str,
    run: str,
    parameter: str,
    prod_root: str = PROD_ROOTS["lngs"],
    data_type: str = "phy",
    experiment: str = "l200",
) -> list:
    """
    Regenerate one parameter for a run and refresh its v1 and contract keys.

    Parameters
    ----------
    output_folder : str
        Output root given to ``auto_run`` (the folder containing ``<ref_version>/``).
    ref_version : str
        Production reference version (``auto/v2.0.0``).
    period, run : str
        Run to repair.
    parameter : str
        Parameter as named in the plot config (``bl_mean``).
    prod_root : str
        Production environment root (read-only).
    data_type : str
        Data type (``phy``).
    experiment : str
        Experiment prefix in file names.

    Returns
    -------
    replaced : list
        The v1 keys replaced in the run's geds file.
    """
    from . import core

    generated_path = os.path.join(output_folder, ref_version)
    chunks = chunk_lists(generated_path, period, run)
    if not chunks:
        raise errors.ConfigError(f"no chunk lists recorded for {period}-{run}")
    entries = config_entries(parameter)
    if not entries:
        raise errors.ConfigError(f"no plot-config entry loads {parameter!r}")

    scratch = os.path.join(generated_path, "generated", "tmp", "repair", period, run)
    shutil.rmtree(scratch, ignore_errors=True)
    # the pipeline's make_dir is not recursive: the version path must pre-exist
    os.makedirs(os.path.join(scratch, ref_version), exist_ok=True)
    config = {
        "output": scratch,
        "dataset": {
            "experiment": experiment.upper(),
            "period": period,
            "version": ref_version,
            "path": prod_root,
            "type": data_type,
            "runs": int(run[1:]),
        },
        "saving": "append",
        "subsystems": {"geds": entries},
    }
    utils.logger.info(
        "repairing %s for %s-%s over %d chunk(s)", parameter, period, run, len(chunks)
    )
    for chunk in chunks:
        core.auto_control_plots(config, chunk, "", {}, render=False)

    run_dir = os.path.join(generated_path, "generated/plt/hit", data_type, period, run)
    scratch_dir = os.path.join(
        scratch, ref_version, "generated/plt/hit", data_type, period, run
    )
    camel = utils.convert_to_camel_case(parameter, "_")
    replaced = []
    for subsystem in ("geds", "pulser01ana"):
        name = f"{experiment}-{period}-{run}-{data_type}-{subsystem}.hdf"
        source, target = os.path.join(scratch_dir, name), os.path.join(run_dir, name)
        if not (os.path.exists(source) and os.path.exists(target)):
            continue
        keys = family_keys(source, camel)
        transplant_keys(target, source, keys)
        utils.logger.info("replaced %d %s key(s) in %s", len(keys), camel, name)
        if subsystem == "geds":
            replaced = keys

    contract_build.build_contract_files(
        generated_path,
        period,
        run,
        metadata_path=os.path.join(prod_root, ref_version, "inputs"),
        data_type=data_type,
        experiment=experiment,
        keys=replaced,
    )
    shutil.rmtree(scratch, ignore_errors=True)
    return replaced


def refresh_contract(
    output_folder: str,
    ref_version: str,
    period: str,
    run: str,
    prod_root: str = PROD_ROOTS["lngs"],
    data_type: str = "phy",
    experiment: str = "l200",
) -> int:
    """
    Rebuild every contract key that still has v1 backing, keeping the rest.

    A keyed refresh rather than a full rebuild on purpose: the classifier
    pivots are stripped from finished runs, so their contract groups exist
    without a v1 source and a rebuild would lose them.

    Parameters
    ----------
    output_folder : str
        Output root given to ``auto_run``.
    ref_version : str
        Production reference version.
    period, run : str
        Run to refresh.
    prod_root : str
        Production environment root (detector metadata).
    data_type : str
        Data type (``phy``).
    experiment : str
        Experiment prefix in file names.

    Returns
    -------
    n_keys : int
        Number of v1 keys refreshed.
    """
    generated_path = os.path.join(output_folder, ref_version)
    v1_file = os.path.join(
        generated_path,
        "generated/plt/hit",
        data_type,
        period,
        run,
        f"{experiment}-{period}-{run}-{data_type}-geds.hdf",
    )
    with pd.HDFStore(v1_file, "r") as store:
        keys = [k.lstrip("/") for k in store.keys() if not k.endswith("_info")]
    contract_build.build_contract_files(
        generated_path,
        period,
        run,
        metadata_path=os.path.join(prod_root, ref_version, "inputs"),
        data_type=data_type,
        experiment=experiment,
        keys=keys,
    )
    return len(keys)
