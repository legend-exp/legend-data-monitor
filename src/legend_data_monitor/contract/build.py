"""Build contract-v2 files from a run's monitoring output.

Reads the per-run v1 HDF (wide frames, rawid columns), converts to the fully
binned v2 contract: detector-name axes, (time × detector) Mean-storage
histograms at 1min/10min/60min + min/max sidecars, per-detector run-mean
frames, a /detector_map, and the run manifest.

During the migration window the v2 file coexists with the v1 outputs under
the name ``...-geds-schema2.hdf``; consumers discover the exact file names
from the manifest, so the eventual rename (when the v1 writer is retired) is
transparent.
"""

import os

import pandas as pd

from .. import utils
from ..config import settings
from ..processing import binning
from . import schema, writer


def _camel(param: str) -> str:
    return "".join(word.capitalize() for word in param.split("_"))


_AUX_RELATION = {
    "_pulser01anaRatio": " / pulser01ana",
    "_pulser01anaDiff": " - pulser01ana",
}


def _param_attrs(key: str) -> dict:
    """Best-effort unit/label/limits lookup for a v1 key name."""
    body = key.lstrip("/")
    flag, _, rest = body.partition("_")
    # peel suffixes to a fixed point: ``Baseline_pulser01anaRatio_var`` carries two
    relation = ""
    stripped = True
    while stripped:
        stripped = False
        for suffix in ("_var", "_mean", *_AUX_RELATION):
            if rest.endswith(suffix):
                rest = rest.removesuffix(suffix)
                relation = _AUX_RELATION.get(suffix, relation)
                stripped = True
    camel_to_snake = {_camel(p): p for p in (settings.PLOT_INFO or {})}
    info = (settings.PLOT_INFO or {}).get(camel_to_snake.get(rest, ""), {})
    is_var = body.endswith("_var")
    label = info.get("label")
    unit = info.get("unit")
    if relation == _AUX_RELATION["_pulser01anaRatio"]:
        unit = "a. u."  # a ratio of two ADC quantities is dimensionless
    attrs = {
        "unit": "%" if is_var and info else unit,
        "label": f"{label}{relation}" if label else None,
        "event_type": flag,
    }
    try:
        keyword = "variation" if is_var else "absolute"
        attrs["limits"] = info["limits"]["geds"][keyword]
    except (KeyError, TypeError):
        pass
    return {k: v for k, v in attrs.items() if v is not None}


def build_contract_files(
    generated_path: str,
    period: str,
    run: str,
    metadata_path: str | None = None,
    data_type: str = "phy",
    experiment: str = "l200",
    keys: list | None = None,
) -> str | None:
    """Produce the v2 contract file + manifest for one (period, run).

    Parameters
    ----------
    generated_path : str
        Output root (the directory containing ``generated/``).
    period, run : str
        Run to convert.
    metadata_path : str, optional
        LEGEND metadata root (``<prod>/inputs``) for the rawid->name map;
        columns stay rawid-labelled strings when not given.
    data_type : str
        Data type (``phy``, ...).
    keys : list, optional
        v1 key bodies (e.g. ``IsPulser_BlMean``) to refresh in place; the rest
        of an existing contract file is kept. Default rebuilds everything.

    Returns the manifest path, or None when the v1 input file is absent.
    """
    run_dir = os.path.join(generated_path, "generated/plt/hit", data_type, period, run)
    v1_file = os.path.join(run_dir, f"{experiment}-{period}-{run}-{data_type}-geds.hdf")
    if not os.path.exists(v1_file):
        utils.logger.debug("no v1 monitoring file at %s; skipping v2 build", v1_file)
        return None

    # rawid -> detector-name mapping (fall back to string rawids)
    rename = {}
    detectors = {}
    if metadata_path is not None:
        det_info = utils.build_detector_info(metadata_path)
        detectors = det_info["detectors"]
        rename = {info["daq_rawid"]: name for name, info in detectors.items()}

    v2_name = f"{experiment}-{period}-{run}-{data_type}-geds-schema2.hdf"
    v2_file = os.path.join(run_dir, v2_name)
    if keys is None and os.path.exists(v2_file):
        os.remove(v2_file)
    wanted = None if keys is None else {k.lstrip("/") for k in keys}

    written_keys = []
    with pd.HDFStore(v1_file, "r") as store:
        for key in sorted(store.keys()):
            if key.endswith(("_info",)):
                continue  # replaced by per-key attrs + manifest vocabulary
            if wanted is not None and key.lstrip("/") not in wanted:
                continue
            frame = store[key]
            frame.columns = [rename.get(c, str(c)) for c in frame.columns]
            body = key.lstrip("/")
            flag, _, rest = body.partition("_")
            attrs = _param_attrs(key)

            if key.endswith("_mean"):
                written_keys.append(writer.write_frame(v2_file, body, frame))
                continue

            frame = writer.apply_remove_keys(frame, period, run)
            binned = binning.frame_to_binned(frame)
            written_keys.extend(
                writer.write_binned_series(v2_file, flag, rest, binned, attrs)
            )
            # 1-D distribution over all samples (cal-view replacement)
            written_keys.append(
                writer.write_distribution(
                    v2_file,
                    flag,
                    rest,
                    binning.fill_distribution(frame.to_numpy().ravel()),
                    attrs,
                )
            )
            # classifiers also get a per-detector distribution: the QC view
            # draws one histogram per detector, and after the transport strip
            # this is the only surviving event-level record of their shape.
            # Fixed +-15 range (classifier values are sigma-like); outliers
            # land in the flow bins, so in-range fractions stay derivable.
            if "Classifier" in rest:
                written_keys.append(
                    writer.write_distribution_2d(
                        v2_file,
                        flag,
                        rest,
                        binning.fill_distribution_2d(
                            frame, n_bins=76, value_range=(-15.0, 15.4)
                        ),
                        attrs,
                    )
                )

    if detectors:
        # also on a keyed refresh: the map is tiny, and a file that lost it
        # would otherwise stay incomplete (the repack below compacts the slack)
        written_keys.append(writer.write_detector_map(v2_file, detectors))

    if keys is not None:
        # rewriting groups in place leaves their old blocks behind; compact, and
        # list the manifest from what the file holds rather than this pass
        from .. import repack

        repack.repack_contract_hdf(v2_file)
        written_keys = _keys_in_file(v2_file)

    from .._version import version

    manifest_path = writer.write_manifest(
        run_dir,
        period,
        run,
        {v2_name: {"keys": sorted(written_keys), "cadences": list(schema.CADENCES)}},
        package_version=version,
        experiment=experiment,
    )
    utils.logger.info("v2 contract file written: %s", v2_file)
    return manifest_path


def _keys_in_file(v2_file: str) -> list:
    """Every manifest-worthy key a contract file holds (hist groups + frames)."""
    import h5py

    from . import reader

    keys = reader.list_hist_keys(v2_file)
    with h5py.File(v2_file, "r") as f:
        keys += [
            name
            for name, obj in f.items()
            if name != "hist"
            and isinstance(obj, h5py.Group)
            and "pandas_type" in obj.attrs
        ]
    return keys
