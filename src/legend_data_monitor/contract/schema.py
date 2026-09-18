"""File-contract v2 schema: key grammar, cadences, attrs, manifest.

Contract summary (schema version 2, fully binned):

Per (period, run, datatype) the writer produces
``l200-<period>-<run>-<datatype>-<subsystem>.hdf`` containing

- ``hist/{flag}_{param}/{cadence}`` — UHI-serialized boost-histogram over
  (regular UTC time axis × detector-name StrCategory axis) with Mean storage
  (count / mean / variance per bin), plus sidecar datasets ``min``/``max``
  in the same group. Cadences: ``1min`` (base fill) and lossless rebins
  ``10min``, ``60min``.
- ``hist/{flag}_{param}_dist`` — 1-D distribution histogram (all samples).
- small pandas frames: ``{flag}_{param}_mean`` (one row of per-detector run
  means) and ``/detector_map`` (name, rawid, string, position, usability).
- every file carries the root attr ``lmon_schema_version = 2``.

A ``l200-<period>-<run>-manifest.json`` beside the files inventories keys,
cadences, and the key vocabulary, and lists IGNORE_KEYS time ranges as
*flagged* (data kept, display shaded) — REMOVE_KEYS data is dropped by the
writer and never reaches the files.

Everything is readable with plain h5py + json (no lmon import).
"""

SCHEMA_VERSION = 2

CADENCES = ("1min", "10min", "60min")
BASE_CADENCE = "1min"
CADENCE_SECONDS = {"1min": 60, "10min": 600, "60min": 3600}

ROOT_ATTR = "lmon_schema_version"

# attrs stored on every hist group / frame key
KEY_ATTRS = ("unit", "label", "limits", "event_type", "schema")


def hist_key(flag: str, param: str, cadence: str) -> str:
    """HDF group path of a time-binned histogram."""
    return f"hist/{flag}_{param}/{cadence}"


def dist_key(flag: str, param: str) -> str:
    """HDF group path of a 1-D distribution histogram."""
    return f"hist/{flag}_{param}_dist"


def dist2d_key(flag: str, param: str) -> str:
    """HDF group path of a per-detector distribution histogram."""
    return f"hist/{flag}_{param}_dist2d"


def mean_key(flag: str, param: str) -> str:
    """Pandas key of the per-detector run-mean frame."""
    return f"{flag}_{param}_mean"


def run_file_name(
    period: str, run: str, datatype: str, subsystem: str, experiment: str = "l200"
) -> str:
    return f"{experiment}-{period}-{run}-{datatype}-{subsystem}.hdf"


def manifest_name(period: str, run: str, experiment: str = "l200") -> str:
    return f"{experiment}-{period}-{run}-manifest.json"
