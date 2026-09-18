"""Rewrite already-produced v1 monitoring files in the current on-disk layout.

The pipeline used to write its pandas HDF outputs uncompressed and in
float64. That costs about 7x the disk it needs: ~2.6x from the dtype and the
missing compression, and another ~2.7x from slack, because overwriting an
*uncompressed* fixed-format key orphans the block it replaces and every chunk
of a run rewrites every key it touches.

The contract (schema2) files had the same two problems in their own form:
float64 histogram storage, and the slack left behind when the float64 uhi
wrote was narrowed in place.

New runs get the right layout from :data:`utils.HDF_COMPRESSION`, the float32
pivots in :mod:`save_data` and the staged writes in :mod:`contract.writer`.
This module brings existing files over without re-running the pipeline: a
16 GB run file repacks to ~2 GB in minutes, against hours to regenerate.
"""

import glob
import os
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from . import utils
from .contract import writer

# Frames of plotting metadata: a handful of strings, nothing to compress.
_METADATA_SUFFIX = "_info"


def _pandas_hdf_is_current(path: str) -> bool:
    """Is this file already float32 and compressed? Cheap: reads no data."""
    import tables

    with tables.open_file(path) as f:
        for node in f.walk_nodes("/", "Leaf"):
            if not node.name.startswith("block") or not node.name.endswith("_values"):
                continue
            if node._v_parent._v_name.endswith(_METADATA_SUFFIX):
                continue
            if node.filters.complib != utils.HDF_COMPRESSION["complib"]:
                return False
            if node.dtype == np.float64:
                return False
    return True


def repack_pandas_hdf(path: str) -> tuple:
    """Repack one pandas HDF file in place; return ``(before, after)`` bytes.

    Keys are converted one at a time, so peak memory is one key, not one
    file. The result is written beside the original and moved into place only
    once it is complete *and* smaller -- a file already in the current layout
    is left exactly as it was.
    """
    before = Path(path).stat().st_size
    if _pandas_hdf_is_current(path):
        return before, before
    # pid-unique so concurrent invocations cannot clobber each other's tmp
    tmp = f"{path}.repack.{os.getpid()}"

    try:
        with pd.HDFStore(path, "r") as store:
            keys = [key.lstrip("/") for key in store.keys()]
        for key in keys:
            frame = pd.read_hdf(path, key=key)
            options = {}
            if not key.endswith(_METADATA_SUFFIX):
                wide = [
                    column
                    for column in getattr(frame, "columns", [])
                    if frame[column].dtype == "float64"
                ]
                if wide:
                    frame = frame.astype({column: "float32" for column in wide})
                options = utils.HDF_COMPRESSION
            frame.to_hdf(tmp, key=key, mode="a", **options)
        after = Path(tmp).stat().st_size
        if after >= before:
            Path(tmp).unlink()
            return before, before
        Path(tmp).replace(path)
        return before, after
    except BaseException:
        if Path(tmp).exists():
            Path(tmp).unlink()
        raise


def repack_run(
    generated_path: str,
    period: str,
    run: str,
    data_type: str = "phy",
    experiment: str = "l200",
) -> dict:
    """Repack every HDF output of one run; return ``{path: (before, after)}``."""
    run_dir = str(Path(generated_path) / "generated/plt/hit" / data_type / period / run)
    pattern = str(Path(run_dir) / f"{experiment}-{period}-{run}-{data_type}-*.hdf")
    results = {}
    for path in sorted(glob.glob(pattern)):
        if path.endswith("-schema2.hdf"):
            before, after = repack_contract_hdf(path)
        else:
            before, after = repack_pandas_hdf(path)
        results[path] = (before, after)
        if after == before:
            utils.logger.info(
                "%s already in the current layout (%.2f GB)",
                Path(path).name,
                before / 2**30,
            )
        else:
            utils.logger.info(
                "repacked %s: %.2f -> %.2f GB (%.1fx)",
                Path(path).name,
                before / 2**30,
                after / 2**30,
                before / max(after, 1),
            )
    return results


#: v1 keys that exist only as transport to the contract build, per subsystem:
#: geds QC classifiers; for spms every per-event pivot of physics/all events
#: (forced-trigger keys are sparse and stay)
def _transport_keys(keys: list, subsystem: str) -> list:
    if subsystem == "spms":
        return [
            k
            for k in keys
            if k.startswith(("IsPhysics_", "All_"))
            and not k.endswith((_METADATA_SUFFIX, "_mean"))
        ]
    return [k for k in keys if "Classifier" in k]


def strip_classifier_pivots(
    generated_path: str,
    period: str,
    run: str,
    data_type: str = "phy",
    experiment: str = "l200",
) -> tuple:
    """Geds flavour of :func:`strip_transport_pivots` (kept for the CLI and callers)."""
    return strip_transport_pivots(
        generated_path, period, run, data_type, experiment, subsystem="geds"
    )


def strip_transport_pivots(
    generated_path: str,
    period: str,
    run: str,
    data_type: str = "phy",
    experiment: str = "l200",
    subsystem: str = "geds",
) -> tuple:
    """
    Drop the transport-only pivots from a run's v1 file of one subsystem.

    They are event-level values that barely compress (geds: the QC classifier
    pivots, 1.6 GB of a 2.2 GB run file; spms: the per-event ``IsPhysics``/
    ``All`` pivots, 376 of 414 MB) and exist in the v1 file only as the
    transport to the contract build, which bins them into
    ``hist/<key>/<cadence>``. Refuses to
    touch the file unless the contract carries every key about to be removed,
    so a v1 file is never stripped of the only copy of its data. QC flag
    (boolean) keys, ``_mean``/``_var``/``_info`` keys and parameters survive.

    Parameters
    ----------
    generated_path : str
        Output root of a previous run (the folder containing ``generated``).
    period, run : str
        Run whose v1 file to strip.
    data_type : str
        Data type (``phy``, ...).
    experiment : str
        Experiment prefix in file names.
    subsystem : str
        ``geds`` or ``spms``.

    Returns
    -------
    sizes : tuple
        ``(before, after)`` file size in bytes; equal when nothing was done.
    """
    run_dir = str(Path(generated_path) / "generated/plt/hit" / data_type / period / run)
    stem = f"{experiment}-{period}-{run}-{data_type}-{subsystem}"
    v1_file = str(Path(run_dir) / f"{stem}.hdf")
    contract_file = str(Path(run_dir) / f"{stem}-schema2.hdf")
    if not Path(v1_file).exists():
        utils.logger.warning("no v1 file at %s; nothing to strip", v1_file)
        return 0, 0

    before = Path(v1_file).stat().st_size
    with pd.HDFStore(v1_file, "r") as store:
        keys = [key.lstrip("/") for key in store.keys()]
    doomed = _transport_keys(keys, subsystem)
    if not doomed:
        utils.logger.info("%s carries no transport pivots", Path(v1_file).name)
        return before, before

    # the guard: every key being removed must already be binned in the contract
    if not Path(contract_file).exists():
        utils.logger.error(
            "refusing to strip %s: no contract file at %s",
            Path(v1_file).name,
            contract_file,
        )
        return before, before
    with h5py.File(contract_file, "r") as f:
        missing = [key for key in doomed if f"hist/{key}/1min" not in f]
    if missing:
        utils.logger.error(
            "refusing to strip %s: %d key(s) not in the contract (e.g. %s)",
            Path(v1_file).name,
            len(missing),
            missing[0],
        )
        return before, before

    tmp = f"{v1_file}.repack.{os.getpid()}"
    try:
        for key in keys:
            if key in doomed:
                continue
            frame = pd.read_hdf(v1_file, key=key)
            options = {} if key.endswith(_METADATA_SUFFIX) else utils.HDF_COMPRESSION
            frame.to_hdf(tmp, key=key, mode="a", **options)
        after = Path(tmp).stat().st_size
        Path(tmp).replace(v1_file)
    except BaseException:
        if Path(tmp).exists():
            Path(tmp).unlink()
        raise
    utils.logger.info(
        "stripped %d transport pivot(s) from %s: %.2f -> %.2f GB",
        len(doomed),
        Path(v1_file).name,
        before / 2**30,
        after / 2**30,
    )
    return before, after


# Only the contract's own histogram arrays. Everything else in the file is a
# pandas frame, whose datasets carry pytables attributes (CLASS, transposed,
# ...) that recreating them would drop -- and pandas then refuses to read them.
_NARROWABLE = (
    "/storage/counts",
    "/storage/values",
    "/storage/variances",
    "/min",
    "/max",
)


def _narrow_contract_datasets(path: str) -> None:
    """Rewrite float64 histogram storage as float32 (counts as int32), in place."""
    with h5py.File(path, "r+") as f:
        wide = []
        f.visititems(
            lambda name, obj: (
                wide.append(name)
                if isinstance(obj, h5py.Dataset)
                and obj.dtype == np.float64
                and name.startswith("hist/")
                and name.endswith(_NARROWABLE)
                else None
            )
        )
        for name in wide:
            dataset = f[name]
            values = dataset[...]
            options = {
                "compression": dataset.compression,
                "compression_opts": dataset.compression_opts,
            }
            if (
                name.endswith("/storage/counts")
                and np.isfinite(values).all()
                and values.min() >= 0
                and values.max() < 2**31
                and np.array_equal(values, np.rint(values))
            ):
                narrowed = values.astype(np.int32)
            else:
                narrowed = values.astype(np.float32)
            parent, _, leaf = name.rpartition("/")
            group = f[parent] if parent else f
            del group[leaf]
            group.create_dataset(leaf, data=narrowed, **options)


def _compact(src: str, dst: str) -> None:
    """Copy every object into a fresh file, leaving the slack behind.

    Each histogram group's ``axes`` dataset holds object references to its own
    ``ref_axes`` children; the copy preserves paths exactly, so the references
    are re-resolved by name afterwards (see ``contract.writer`` for why not
    ``expand_refs``).
    """
    with h5py.File(src, "r") as source, h5py.File(dst, "w") as target:
        targets = writer.reference_targets(source)
        for name, value in source.attrs.items():
            target.attrs[name] = value
        for name in source:
            source.copy(name, target, name=name, expand_refs=False)
        writer.restore_references(target, targets)


def _contract_hdf_is_current(path: str) -> bool:
    """Is the storage already narrowed AND the file free of slack? Reads no data."""
    stored = 0
    with h5py.File(path, "r") as f:
        wide = []

        def visit(name, obj):
            nonlocal stored
            if not isinstance(obj, h5py.Dataset):
                return
            stored += obj.id.get_storage_size()
            if (
                obj.dtype == np.float64
                and name.startswith("hist/")
                and name.endswith(_NARROWABLE)
            ):
                wide.append(name)

        f.visititems(visit)
    if wide:
        return False
    # narrowed but slacked (e.g. an interrupted earlier repack) still needs work
    return Path(path).stat().st_size < 1.15 * max(stored, 1)


def repack_contract_hdf(path: str) -> tuple:
    """Repack one contract (schema2) file in place; return ``(before, after)``."""
    before = Path(path).stat().st_size
    if _contract_hdf_is_current(path):
        return before, before
    # narrow a scratch copy, never the original: a failure part-way must not
    # leave the real file mutated (or half-narrowed yet reading as "current")
    tmp = f"{path}.repack.{os.getpid()}"
    tmp2 = tmp + ".compact"
    try:
        shutil.copyfile(path, tmp)
        _narrow_contract_datasets(tmp)
        _compact(tmp, tmp2)
        Path(tmp).unlink()
        after = Path(tmp2).stat().st_size
        if after >= before:
            Path(tmp2).unlink()
            return before, before
        Path(tmp2).replace(path)
        return before, after
    except BaseException:
        for leftover in (tmp, tmp2):
            if Path(leftover).exists():
                Path(leftover).unlink()
        raise
