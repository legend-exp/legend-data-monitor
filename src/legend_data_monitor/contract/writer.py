"""The ONLY module that writes monitoring output files (contract v2).

Writes UHI-serialized histograms + small pandas frames into per-run HDF
files, stamps the schema version, and maintains the run manifest. Applies
REMOVE_KEYS filtering (removed data never reaches the files); IGNORE_KEYS
ranges are exported as *flagged* ranges in the manifest instead of dropped.
"""

import json
import os
from datetime import datetime, timezone

import h5py
import numpy as np
import pandas as pd
from uhi.io import hdf5 as uhi_hdf5

from ..config import settings
from . import schema


def reference_targets(root) -> dict:
    """Map each object-reference dataset under ``root`` to the names it points at.

    h5py can copy referenced objects along with a tree (``expand_refs``), but
    that is not idempotent: run over a file it produced itself it fails with
    "destination object already exists". Our references only ever point inside
    the tree being copied, so copy the tree verbatim and re-resolve the
    references by name -- which is what these two helpers do.
    """
    targets = {}
    file = root.file

    def note(path, obj):
        if (
            isinstance(obj, h5py.Dataset)
            and h5py.check_dtype(ref=obj.dtype) is h5py.Reference
        ):
            targets[path] = [file[ref].name for ref in np.ravel(obj[...])]

    root.visititems(note)
    return targets


def restore_references(root, targets: dict) -> None:
    """Re-point copied reference datasets at the objects of ``root``'s file."""
    file = root.file
    for path, names in targets.items():
        dataset = root[path]
        flat = np.ravel(dataset[...])
        for index, name in enumerate(names):
            flat[index] = file[name].ref
        dataset[...] = flat.reshape(dataset.shape)


def write_hist(
    file_path: str,
    key: str,
    hist,
    mins=None,
    maxs=None,
    attrs: dict | None = None,
) -> None:
    """Write one histogram under ``key`` (UHI HDF5 layout + min/max sidecars)."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with h5py.File(file_path, "a") as f:
        f.attrs[schema.ROOT_ATTR] = schema.SCHEMA_VERSION
        if key in f:
            del f[key]
        # Assemble the group in memory and copy the finished, narrowed version
        # across. Letting uhi write float64 straight into the file and then
        # narrowing in place works, but HDF5 does not reclaim the blocks it
        # orphans: measured 1.4x slack over a run's worth of keys.
        with h5py.File(
            "uhi-staging", driver="core", backing_store=False, mode="w"
        ) as staging:
            # stage under the very path it will occupy, so the object
            # references inside resolve by the same absolute name after the copy
            group = staging.create_group(key)
            uhi_hdf5.write(group, hist)
            _narrow_storage(group)
            if mins is not None:
                group.create_dataset(
                    "min", data=np.asarray(mins, dtype=np.float32), compression="gzip"
                )
            if maxs is not None:
                group.create_dataset(
                    "max", data=np.asarray(maxs, dtype=np.float32), compression="gzip"
                )
            for name, value in (attrs or {}).items():
                if value is not None:
                    group.attrs[name] = (
                        json.dumps(value) if isinstance(value, (list, dict)) else value
                    )
            group.attrs["schema"] = schema.SCHEMA_VERSION
            # Spell out the two conventions a plain-h5py reader cannot guess
            # and gets silently wrong: Mean storage keeps means (NOT sums, so
            # never divide by counts), and both axes carry flow bins.
            group.attrs["values_are"] = "mean"
            group.attrs["counts_are"] = "n_entries"
            group.attrs["flow_bins"] = (
                "axis_0: [underflow, ...bins..., overflow]; "
                "axis_1: [...categories..., flow]"
            )

            parent, _, leaf = key.rpartition("/")
            targets = reference_targets(group)
            staging.copy(
                group,
                f.require_group(parent) if parent else f,
                name=leaf,
                expand_refs=False,
            )
            restore_references(f[key], targets)


def _narrow_storage(group) -> None:
    """Store histogram storage arrays at the precision they actually carry.

    boost-histogram's views are float64 throughout, which doubles both the
    file and the time a reader spends inflating it -- for values whose
    relative error at float32 is ~6e-8, and for counts that are integers.
    Halving them is the single biggest lever on how fast the dashboard opens
    a run. Called on the in-memory staging group, so the float64 datasets uhi
    wrote are discarded with it rather than orphaned in the output file.
    """
    storage = group.get("storage")
    if storage is None:
        return
    for name in list(storage):
        dataset = storage[name]
        if not isinstance(dataset, h5py.Dataset) or dataset.dtype != np.float64:
            continue
        values = dataset[...]
        finite = np.isfinite(values)
        if (
            name == "counts"
            and finite.all()
            and values.min() >= 0
            and values.max() < 2**31
            and np.array_equal(values, np.rint(values))
        ):
            # signed on purpose: counts are integers, and a reader that
            # subtracts two of them must get a negative number, not 4e9
            narrowed = values.astype(np.int32)
        else:
            narrowed = values.astype(np.float32)
        del storage[name]
        storage.create_dataset(name, data=narrowed, compression="gzip")


def write_binned_series(
    file_path: str,
    flag: str,
    param: str,
    binned,
    attrs: dict | None = None,
) -> list:
    """Write a BinnedTimeSeries at base cadence + derived rebins; return keys."""
    keys = []
    for cadence in schema.CADENCES:
        if cadence == schema.BASE_CADENCE:
            b = binned
        else:
            factor = (
                schema.CADENCE_SECONDS[cadence]
                // schema.CADENCE_SECONDS[schema.BASE_CADENCE]
            )
            b = binned.rebin(factor)
        key = schema.hist_key(flag, param, cadence)
        write_hist(file_path, key, b.hist, b.mins, b.maxs, attrs)
        keys.append(key)
    return keys


def write_distribution(
    file_path: str, flag: str, param: str, hist, attrs: dict | None = None
) -> str:
    key = schema.dist_key(flag, param)
    write_hist(file_path, key, hist, attrs=attrs)
    return key


def write_distribution_2d(
    file_path: str, flag: str, param: str, hist, attrs: dict | None = None
) -> str:
    key = schema.dist2d_key(flag, param)
    write_hist(file_path, key, hist, attrs=attrs)
    return key


def write_frame(file_path: str, key: str, frame: pd.DataFrame) -> str:
    """Write a small pandas frame (run means, detector map, calib pars)."""
    # same courtesy as write_hist: the period directory may not exist yet
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    frame.to_hdf(file_path, key=key, mode="a")
    with h5py.File(file_path, "a") as f:
        f.attrs[schema.ROOT_ATTR] = schema.SCHEMA_VERSION
    return key


def write_detector_map(file_path: str, detectors: dict) -> str:
    """Write /detector_map from a build_detector_info()-style dict."""
    rows = [
        {
            "name": name,
            "rawid": info.get("daq_rawid"),
            "string": info.get("string"),
            "position": info.get("position"),
            "processable": info.get("processable"),
            "usability": info.get("usability"),
            "mass_in_kg": info.get("mass_in_kg"),
        }
        for name, info in detectors.items()
    ]
    return write_frame(file_path, "detector_map", pd.DataFrame(rows))


def apply_remove_keys(df: pd.DataFrame, period: str, run: str) -> pd.DataFrame:
    """Drop REMOVE_KEYS time ranges per detector (producer-side, permanent).

    ``settings.REMOVE_KEYS`` maps detector name -> list of {from, to, period, run}
    style entries (see settings/remove-keys.yaml); only matching rows of the
    matching detector columns are dropped (set to NaN).
    """
    removals = settings.REMOVE_KEYS or {}
    out = df
    for det, entries in removals.items():
        if det not in out.columns:
            continue
        for entry in entries if isinstance(entries, list) else [entries]:
            # an entry may be scoped to one period/run; skip the others (an
            # unscoped time range would otherwise apply to every run)
            if entry.get("period") not in (None, period):
                continue
            if entry.get("run") not in (None, run):
                continue
            lo = (
                pd.Timestamp(entry.get("from"), tz="UTC") if entry.get("from") else None
            )
            hi = pd.Timestamp(entry.get("to"), tz="UTC") if entry.get("to") else None
            mask = pd.Series(True, index=out.index)
            if lo is not None:
                mask &= out.index >= lo
            if hi is not None:
                mask &= out.index <= hi
            if mask.any():
                out = out.copy()
                out.loc[mask, det] = float("nan")
    return out


def flagged_ranges(period: str) -> list:
    """Return IGNORE_KEYS ranges for a period, exported to the manifest as flagged.

    The ranges stay in the data; consumers display them shaded, not dropped.
    """
    entry = (settings.IGNORE_KEYS or {}).get(period)
    if not entry:
        return []
    starts = entry.get("start_keys") or []
    stops = entry.get("stop_keys") or []
    return [
        {"from": str(lo), "to": str(hi), "reason": "ignore-keys"}
        for lo, hi in zip(starts, stops)
    ]


def write_manifest(
    dir_path: str,
    period: str,
    run: str,
    files: dict,
    package_version: str,
    experiment: str = "l200",
) -> str:
    """Write the run manifest. ``files`` maps file name -> {"keys": [...], "cadences": [...]}."""
    manifest = {
        "schema_version": schema.SCHEMA_VERSION,
        "package_version": package_version,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "period": period,
        "run": run,
        "files": files,
        "cadences": list(schema.CADENCES),
        "key_vocabulary": {
            "flags": settings.FLAGS_RENAME,
            "parameters": {
                name: {
                    "label": info.get("label"),
                    "unit": info.get("unit"),
                }
                for name, info in (settings.PLOT_INFO or {}).items()
                if isinstance(info, dict)
            },
        },
        "flagged_ranges": flagged_ranges(period),
    }
    path = os.path.join(dir_path, schema.manifest_name(period, run, experiment))
    os.makedirs(dir_path, exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=1, default=str)
    return path
