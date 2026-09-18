"""Direct LH5 loading of per-channel monitoring parameters.

pygama's ``DataLoader`` builds a file database and a per-key *entry list* so
that event-level cuts can be applied while loading. Monitoring never uses that:
it wants whole channels for a known list of files, and the entry-list
construction dominates the cost — measured on p22/r012, loading 59 geds tables
x 5 files x 5 parameters took 39 s through ``DataLoader`` (21 s of it building
entry lists) versus 2.3 s reading the same data with ``lh5.read_as``.

This module is that direct path. It returns one tidy frame per tier with
explicit ``channel`` and ``timestamp`` columns, so callers merge tiers on those
keys instead of relying on two loaders emitting identically ordered rows.
"""

import os

import numpy as np
import pandas as pd
from lgdo import lh5
from lh5.io.exceptions import LH5DecodeError

from .. import utils


def list_channels(file_path: str) -> list:
    """Channel table names (``chNNNNNNN``) present in an LH5 file."""
    return [name for name in lh5.ls(file_path) if name.startswith("ch")]


def load_channel_frame(
    files: list,
    tier: str,
    channels: list,
    params: list,
) -> pd.DataFrame:
    """Read ``params`` for ``channels`` across ``files`` from one tier.

    Parameters
    ----------
    files : list
        LH5 files of a single tier, in the order they should be concatenated.
    tier : str
        Tier name, used to build the in-file path ``<channel>/<tier>/``.
    channels : list
        Channel table names to read (``chNNNNNNN``).
    params : list
        Field names to read; ``timestamp`` is always included.

    Returns
    -------
    pandas.DataFrame
        The requested ``params`` plus ``timestamp`` and ``channel`` (integer
        rawid). Channels missing from the files are skipped with a warning
        rather than raising, mirroring the tolerance of the loader it replaces.
    """
    if not files or not channels:
        return pd.DataFrame()

    from ..processing import spms

    fields, derived = spms.expand_fields(list(params) + ["timestamp"])
    frames = []
    # the dtype each field actually has in the tier, before any concatenation
    # gets a chance to widen it (see utils.narrow_to_native_dtypes)
    native = {}
    for channel in channels:
        # read one file per call and concatenate: handing lh5.read_as the whole
        # file list is ~9x slower for the same rows (measured on p22/r012,
        # 30 channels x 10 dsp files: 14.3 s vs 1.6 s)
        per_file = []
        for path in files:
            try:
                frame = lh5.read_as(
                    f"{channel}/{tier}/", [path], library="pd", field_mask=fields
                )
            except (KeyError, ValueError, LH5DecodeError) as exc:
                utils.logger.debug(
                    "skipping %s in tier '%s' of %s: %s",
                    channel,
                    tier,
                    os.path.basename(path),
                    exc,
                )
                continue
            if frame is not None and len(frame):
                # reduce before concatenating: pandas concatenates ragged
                # (awkward-backed) columns in Python, ~100x slower than scalars
                frame = spms.reduce_frame(frame, derived, list(params))
                native.update(frame.dtypes.items())
                per_file.append(frame)
        if not per_file:
            continue
        frame = (
            pd.concat(per_file, ignore_index=True) if len(per_file) > 1 else per_file[0]
        )
        # rawids are 7-digit: int32 halves what is otherwise one of the widest
        # columns in the frame, one entry per event
        frame["channel"] = np.int32(int(channel[2:]))
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return utils.narrow_to_native_dtypes(pd.concat(frames, ignore_index=True), native)


def merge_tiers(frames: dict) -> pd.DataFrame:
    """Merge per-tier frames on ``(channel, timestamp)``.

    The loader this replaces concatenated tiers positionally
    (``pd.concat(..., axis=1)``), which is only correct while every tier emits
    rows in exactly the same order; merging on the keys is equivalent when that
    holds and correct when it does not.
    """
    present = [frame for frame in frames.values() if frame is not None and len(frame)]
    if not present:
        return pd.DataFrame()

    merged = present[0]
    for frame in present[1:]:
        overlap = [
            col
            for col in frame.columns
            if col in merged.columns and col not in ("channel", "timestamp")
        ]
        merged = merged.merge(
            frame.drop(columns=overlap), on=["channel", "timestamp"], how="outer"
        )
    # the outer merge re-widens any column it had to NaN-fill
    native = {}
    for frame in present:
        native.update(frame.dtypes.items())
    return utils.narrow_to_native_dtypes(merged, native)


def resolve_files(
    path: str,
    version: str,
    tier: str,
    datatype: str,
    period: str,
    timerange: dict,
    experiment: str = "l200",
) -> list:
    """Resolve a monitoring time range into the tier files it covers.

    Replaces ``DataLoader``'s filedb query: the caller already knows which
    keys/runs it wants, so the files are found by globbing the production tree
    instead of building and querying a file database.

    ``timerange`` is the structure produced by :func:`utils.get_query_times`:
    ``{"timestamp": [keys]}``, ``{"run": [runs]}`` or
    ``{"<word>": {"start": ..., "end": ...}}``.
    """
    import glob
    import os

    base = os.path.join(path, version, "generated/tier", tier, datatype, period)
    exp = experiment.lower()

    if not timerange:
        return []
    word = list(timerange)[0]
    selection = timerange[word]

    files: list = []
    if isinstance(selection, dict):
        # start/end window: take everything in the period and filter by key
        start, end = selection.get("start"), selection.get("end")
        for candidate in sorted(glob.glob(f"{base}/*/{exp}-*-tier_{tier}.lh5")):
            key = _key_of(candidate)
            if key and (start is None or key >= start) and (end is None or key <= end):
                files.append(candidate)
    elif word == "run":
        for run in selection:
            files += sorted(glob.glob(f"{base}/{run}/{exp}-*-tier_{tier}.lh5"))
    else:
        for key in selection:
            files += sorted(glob.glob(f"{base}/*/{exp}-*-{key}-tier_{tier}.lh5"))

    # de-duplicate while keeping order (a key matches exactly one file)
    return list(dict.fromkeys(files))


def _key_of(file_path: str) -> str | None:
    """Extract the ``YYYYmmddTHHMMSSZ`` key from a tier file name."""
    import re

    match = re.search(r"\d{8}T\d{6}Z", file_path)
    return match.group(0) if match else None
