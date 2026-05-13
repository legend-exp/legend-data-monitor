"""Low-level I/O for legend-datasets status config files and validity.yaml."""

from pathlib import Path

import yaml


def _dump(data) -> str:
    """Serialise `data` to a YAML string with double-quoted strings."""
    return yaml.dump(
        data,
        default_flow_style=False,
        sort_keys=False,
        allow_unicode=True,
    )


# ---------------------------------------------------------------------------
# Config file helpers
# ---------------------------------------------------------------------------


def read_config(path: Path) -> dict:
    """Load a YAML config file.  Returns {} if the file does not exist."""
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def append_to_config(path: Path, ged: str, entry: dict) -> None:
    """
    Append a new detector block to a config file.

    Raises ValueError if `ged` is already present in the file — call
    update_in_config instead.
    """
    if ged in read_config(path):
        raise ValueError(f"{ged} already present in {path.name} — use update_in_config")
    with open(path, "a") as f:
        f.write(_dump({ged: entry}))


def update_in_config(path: Path, ged: str, entry: dict) -> None:
    """
    Update (or insert) a detector block in a config file.

    Reads the full file, merges `entry` into the existing detector block
    and rewrites the file.
    """
    cfg = read_config(path)
    existing = cfg.get(ged)
    if isinstance(existing, dict):
        existing.update(entry)
        cfg[ged] = existing
    else:
        cfg[ged] = entry
    # if file doesn't exist then create
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        _dump(cfg, f)


def remove_from_config(path: Path, ged: str) -> bool:
    """
    Remove a detector block from a config file.

    Returns True if the entry was found and removed, False if not present.
    """
    cfg = read_config(path)
    if ged not in cfg:
        return False
    del cfg[ged]
    with open(path, "w") as f:
        _dump(cfg, f)
    return True


# ---------------------------------------------------------------------------
# Validity helpers
# ---------------------------------------------------------------------------


def validity_blocked(validity: list, timestamp: str) -> str | None:
    """
    Return a reason if a new 'append' entry cannot safely be added at `timestamp`, or None if the timestamp is clear to write.

    Blocked conditions:
      - A 'remove' entry already exists at this timestamp.
      - More than one entry already exists at this timestamp (ambiguous merge).
    """
    at_ts = [e for e in validity if e["valid_from"] == timestamp]
    if len(at_ts) > 1:
        return f"multiple existing entries at {timestamp}"
    if at_ts and at_ts[0].get("mode") == "remove":
        return f"existing 'remove' entry at {timestamp} — needs manual reset"
    return None


def ensure_validity_entry(
    validity: list,
    timestamp: str,
    config_name: str,
    categories: list,
) -> bool:
    """
    Insert or update a validity 'append' entry for `config_name` at `timestamp`.

    The validity file looks as follows:
    - valid_from: YYYYMMDDTHHMMSSZ
        category:
            - all
            - cal
            - fft
        mode: append
        apply:
            - l200-pXX-rYYY-T%-{all/phy}-config.yaml

    If an entry already exists at `timestamp`, `config_name` is added to its
    apply list (if not already present).  Otherwise a new entry is inserted
    in chronological order.

    Returns True if `validity` was modified.

    Call validity_blocked() first to ensure the timestamp is safe to write.
    """
    at_ts = [e for e in validity if e["valid_from"] == timestamp]

    if at_ts:
        apply_list = at_ts[0].setdefault("apply", [])
        if config_name not in apply_list:
            apply_list.append(config_name)
            return True
        return False

    new_entry = {
        "valid_from": timestamp,
        "category": list(categories),
        "mode": "append",
        "apply": [config_name],
    }
    for i, e in enumerate(validity):
        if e["valid_from"] > timestamp:
            validity.insert(i, new_entry)
            return True
    validity.append(new_entry)
    return True


def remove_from_validity_apply(
    validity: list,
    timestamp: str,
    config_name: str,
) -> bool:
    """
    Remove `config_name` from the apply list of the validity entry at `timestamp`.

    Returns True if the list was modified.  Does not remove the validity
    entry itself if the apply list becomes empty — leave that to a human.
    """
    for entry in validity:
        if entry["valid_from"] != timestamp:
            continue
        apply_list = entry.get("apply", [])
        if config_name in apply_list:
            apply_list.remove(config_name)
            return True
    return False
