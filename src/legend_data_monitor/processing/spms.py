"""Per-event reductions of the ragged SiPM fields.

The spms dsp/hit tiers store pulses per event as ``VectorOfVectors``
(``energy_in_pe``, ``trigger_pos``, ``is_valid_hit``). Monitoring works on
one float per event, so the loader reads those fields, reduces them here with
the recipes in ``settings/spms-reductions.yaml`` and never hands a ragged
column to the rest of the pipeline.
"""

import awkward as ak
import numpy as np
import pandas as pd

from .. import utils

OPS = {"count_true", "sum_masked", "max_masked", "min_masked"}


def is_reduction(param: str) -> bool:
    """Whether ``param`` is a derived scalar defined in spms-reductions.yaml."""
    return param in utils.SPMS_REDUCTIONS


def expand_fields(params: list) -> tuple:
    """
    Split requested parameters into lh5 fields to read and reductions to run.

    Parameters
    ----------
    params : list
        Parameter names as requested by the caller.

    Returns
    -------
    tuple
        ``(fields, derived)``: the lh5 fields to read (requested plain
        parameters plus the sources of every derived one, deduplicated) and
        the names of the derived parameters to compute afterwards.
    """
    fields, derived = [], []
    for param in params:
        if is_reduction(param):
            derived.append(param)
            fields += utils.SPMS_REDUCTIONS[param]["fields"]
        else:
            fields.append(param)
    return list(dict.fromkeys(fields)), derived


def _as_awkward(column: pd.Series) -> ak.Array:
    array = column.array
    if hasattr(array, "_data"):  # awkward-pandas extension array
        return array._data
    return ak.Array(column.tolist())


def reduce_frame(frame: pd.DataFrame, derived: list, requested: list) -> pd.DataFrame:
    """
    Compute the derived columns and drop the ragged sources.

    Parameters
    ----------
    frame : pandas.DataFrame
        Frame as read from lh5, ragged fields included.
    derived : list
        Derived parameter names to compute (keys of spms-reductions.yaml).
    requested : list
        Parameters the caller asked for; ragged sources not in this list are
        dropped after the reductions.

    Returns
    -------
    pandas.DataFrame
        ``frame`` with one float32 column per derived parameter.
    """
    if not derived:
        return frame
    out = frame.copy()
    for name in derived:
        recipe = utils.SPMS_REDUCTIONS[name]
        op = recipe["op"]
        if op not in OPS:
            raise utils.errors.ConfigError(f"unknown spms reduction op '{op}'")
        values = _as_awkward(frame[recipe["fields"][0]])
        if op == "count_true":
            result = ak.sum(values, axis=1)
        else:
            mask = _as_awkward(frame[recipe["fields"][1]])
            masked = values[mask]
            if op == "sum_masked":
                result = ak.sum(masked, axis=1)
            elif op == "max_masked":
                result = ak.fill_none(ak.max(masked, axis=1), np.nan)
            else:
                result = ak.fill_none(ak.min(masked, axis=1), np.nan)
        out[name] = ak.to_numpy(result).astype(np.float32)
    sources = {f for name in derived for f in utils.SPMS_REDUCTIONS[name]["fields"]}
    return out.drop(columns=[c for c in sources if c not in requested])
