"""Reference reader for contract v2 files.

Used by tests (round-trip verification) and available to any consumer; the
format itself needs only h5py + json (see the compatibility test), so the
dashboard may read files without importing this package.
"""

import json
import os

import boost_histogram as bh
import h5py
import numpy as np
import pandas as pd
from uhi.io import hdf5 as uhi_hdf5

from . import schema


def read_schema_version(file_path: str) -> int | None:
    with h5py.File(file_path, "r") as f:
        version = f.attrs.get(schema.ROOT_ATTR)
    return int(version) if version is not None else None


def read_hist(file_path: str, key: str):
    """Return (histogram, mins, maxs, attrs) for a hist group."""
    with h5py.File(file_path, "r") as f:
        group = f[key]
        hist = bh.Histogram(uhi_hdf5.read(group))
        mins = group["min"][...] if "min" in group else None
        maxs = group["max"][...] if "max" in group else None
        attrs = dict(group.attrs)
    return hist, mins, maxs, attrs


def read_binned_series(file_path: str, flag: str, param: str, cadence: str):
    """Return a BinnedTimeSeries for one (flag, param, cadence)."""
    from ..processing.binning import BinnedTimeSeries

    hist, mins, maxs, _ = read_hist(file_path, schema.hist_key(flag, param, cadence))
    if mins is None:
        n_bins, n_det = hist.view()["count"].shape
        mins = np.full((n_bins, n_det), np.nan)
        maxs = np.full((n_bins, n_det), np.nan)
    return BinnedTimeSeries(hist, mins, maxs)


def read_frame(file_path: str, key: str) -> pd.DataFrame:
    """
    Read one frame-valued contract key.

    Frames inside contract files are stored as pandas HDF today; consumers go
    through here so that storage can change without touching them. v1
    monitoring files are not contract files: read those with ``pd.read_hdf``.

    Parameters
    ----------
    file_path : str
        Contract file (run ``-schema2.hdf`` or period ``-monitoring.hdf``).
    key : str
        Frame key (``detector_map``, ``escale/<run>``, ...).

    Returns
    -------
    pandas.DataFrame
        The stored frame.
    """
    return pd.read_hdf(file_path, key=key)


def read_manifest(
    dir_path: str, period: str, run: str, experiment: str = "l200"
) -> dict:
    path = os.path.join(dir_path, schema.manifest_name(period, run, experiment))
    with open(path) as f:
        return json.load(f)


def list_hist_keys(file_path: str) -> list:
    """All hist/... group paths present in a file."""
    keys = []
    with h5py.File(file_path, "r") as f:
        if "hist" not in f:
            return keys

        def visit(name, obj):
            if isinstance(obj, h5py.Group) and (
                "storage" in obj or "stack" in obj.attrs.get("type", "")
            ):
                keys.append(f"hist/{name}")
                return None
            return None

        f["hist"].visititems(visit)
    return keys
