"""Golden-snapshot tool for the refactor baseline (Phase 0).

Walks a pipeline output tree and records, for every file: relative path, size
(bytes) for binary artifacts, and for HDF files the sorted key list plus a
per-key checksum of the numeric payload. Shelve/PDF/PNG artifacts are recorded
by path only (their bytes are not stable across runs).

Usage:
    python tests/baseline/snapshot.py <output_tree> <golden.json>   # write
    python tests/baseline/snapshot.py <output_tree> <golden.json> --check
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _df_checksum(df: pd.DataFrame) -> str:
    """Checksum of the numeric content of a frame, robust to file rewrites."""
    h = hashlib.sha256()
    h.update(",".join(map(str, df.columns)).encode())
    h.update(",".join(map(str, df.index[:5])).encode())
    values = df.to_numpy()
    if values.dtype == object:
        h.update(str(values.tolist()).encode())
    else:
        h.update(np.ascontiguousarray(np.nan_to_num(values)).tobytes())
    return h.hexdigest()[:16]


def snapshot(tree: Path) -> dict:
    out = {}
    for path in sorted(tree.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(tree))
        if path.suffix == ".hdf":
            entry = {"keys": {}}
            with pd.HDFStore(path, "r") as store:
                for key in sorted(store.keys()):
                    entry["keys"][key] = _df_checksum(store[key])
            out[rel] = entry
        elif path.suffix in (".yaml", ".yml", ".json"):
            out[rel] = {"sha": hashlib.sha256(path.read_bytes()).hexdigest()[:16]}
        else:  # shelve/pdf/png/log: existence only
            out[rel] = {"present": True}
    return out


def main() -> int:
    tree, golden = Path(sys.argv[1]), Path(sys.argv[2])
    current = snapshot(tree)
    if "--check" in sys.argv:
        reference = json.loads(golden.read_text())
        problems = []
        for rel, entry in reference.items():
            if rel not in current:
                problems.append(f"MISSING {rel}")
            elif current[rel] != entry:
                problems.append(f"DIFFERS {rel}")
        for rel in current:
            if rel not in reference:
                problems.append(f"EXTRA   {rel}")
        report = "\n".join(problems) if problems else "OK: snapshot matches golden"
        print(report)  # noqa: T201
        return 1 if problems else 0
    golden.parent.mkdir(parents=True, exist_ok=True)
    golden.write_text(json.dumps(current, indent=1, sort_keys=True))
    print(f"wrote {golden} ({len(current)} files)")  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
