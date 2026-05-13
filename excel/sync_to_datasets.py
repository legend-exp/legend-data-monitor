"""
Sync the Excel usability dashboard back to legend-datasets.

Builds a Transition history from both the on-disk config files and the
Excel sheet, diffs them per detector, and applies any differences.

Three kinds of change
---------------------
    ADD    — Excel records a transition that disk does not -> write config + validity
    UPDATE — Both record a transition at the same run, but value or reason differs
            -> update the existing source config file
    REMOVE — Disk records a transition that Excel does not want → remove detector
           entry from its source config file

"""

from pathlib import Path

import yaml
from config_io import (
    append_to_config,
    ensure_validity_entry,
    read_config,
    remove_from_config,
    update_in_config,
    validity_blocked,
)
from detector_history import (
    History,
    Transition,
    _ordered_entries,
    build_from_disk,
    build_from_excel,
)

LEGEND_DATASETS = Path(__file__).parent / "legend-datasets"

_PERIOD_CATEGORIES: dict[str, list[str]] = {
    "p16": ["all", "cal", "fft", "lac"],
}
_DEFAULT_CATEGORIES = ["all", "cal", "fft"]


def _categories_for(period: str) -> list[str]:
    return _PERIOD_CATEGORIES.get(period, _DEFAULT_CATEGORIES)


def _cal_config_removed_at_phy(
    period: str, run: str, validity: list, runinfo: dict
) -> bool:
    """Return True if validity already strips the cal-config at the phy start of this run.

    When this is the case, a cal-only usability change should be written
    into the cal-config rather than the all-config: the remove entry will
    automatically revert the value for the phy run without needing a
    separate phy-config override.
    """
    phy_ts = runinfo.get(period, {}).get(run, {}).get("phy", {}).get("start_key")
    if not phy_ts:
        return False
    cal_config = f"l200-{period}-{run}-T%-cal-config.yaml"
    return any(
        e["valid_from"] == phy_ts
        and e.get("mode") == "remove"
        and cal_config in e.get("apply", [])
        for e in validity
    )


def _target_config_name(transition: Transition, validity: list, runinfo: dict) -> str:
    """
    Derive the intended config file name for a new (ADD) transition.

    Cal transitions go to all-config by default so the change persists into
    the phy run.  Exception: if validity already has a remove entry that
    strips the cal-config at phy start, the cal change is cal-only and
    should live in the cal-config instead — the remove handles the revert.
    """
    if transition.run_type == "phy":
        return f"l200-{transition.period}-{transition.run}-T%-phy-config.yaml"
    if _cal_config_removed_at_phy(transition.period, transition.run, validity, runinfo):
        return f"l200-{transition.period}-{transition.run}-T%-cal-config.yaml"
    return f"l200-{transition.period}-{transition.run}-T%-all-config.yaml"


def _needs_update(disk_t: Transition, excel_t: Transition) -> bool:
    """
    Return True if the disk transition should be updated to match Excel.

    A reason difference only triggers an update when Excel explicitly
    provides a reason — we never write a config entry purely to clear a
    reason that Excel left blank.
    """
    if disk_t.usability != excel_t.usability:
        return True
    if excel_t.reason and excel_t.reason != disk_t.reason:
        return True
    return False


def _build_entry(transition: Transition) -> dict:
    entry: dict = {"usability": transition.usability}
    if transition.reason:
        entry["reason"] = transition.reason
    return entry


# ---------------------------------------------------------------------------
# Diff
# ---------------------------------------------------------------------------


def _diff(
    disk_usab: dict,
    excel_usab: dict,
    disk_hist: History,
    excel_hist: History,
    all_geds: list[str],
    excel_periods: dict,
    runinfo: dict,
) -> tuple[list, list, list]:
    """
    Compare disk and Excel transition histories per detector.

    Only ADD/UPDATE/REMOVE transitions that fall within the Excel period window.
    Disk transitions outside that window are never touched.

    Returns (adds, updates, removes) where:
      adds    — list[Transition]          (Excel transition, no disk match)
      updates — list[tuple[Transition, Transition]]  (disk, excel)
      removes — list[Transition]          (disk transition, no Excel match)
    """
    adds, updates, removes = [], [], []
    excel_keys = set(_ordered_entries(excel_periods, runinfo))

    for ged in all_geds:
        if disk_usab.get(ged) == excel_usab.get(ged):
            continue  # usability identical for all Excel-period events — skip

        disk_by_key = {(t.period, t.run, t.run_type): t for t in disk_hist.get(ged, [])}
        excel_by_key = {
            (t.period, t.run, t.run_type): t for t in excel_hist.get(ged, [])
        }

        for key in set(disk_by_key) | set(excel_by_key):
            if key not in excel_keys:
                continue  # never modify transitions outside the Excel window

            d = disk_by_key.get(key)
            e = excel_by_key.get(key)

            if e and not d:
                adds.append(e)
            elif d and not e:
                removes.append(d)
            elif d and e and _needs_update(d, e):
                updates.append((d, e))

    return adds, updates, removes


# ---------------------------------------------------------------------------
# Apply changes
# ---------------------------------------------------------------------------


def _apply_adds(
    adds: list[Transition],
    statuses_dir: Path,
    validity: list,
    runinfo: dict,
) -> tuple[int, bool]:
    written = 0
    validity_changed = False

    for transition in adds:
        config_name = _target_config_name(transition, validity, runinfo)
        config_path = statuses_dir / config_name
        categories = _categories_for(transition.period)  # all, cal, fft etc
        entry = _build_entry(transition)

        # check this logic
        blocked = validity_blocked(validity, transition.timestamp)
        if blocked:
            # print(f"  ADD    {transition.ged:12s}  {transition.period} {transition.run} {transition.run_type:3s}"
            #      f"  → {transition.usability!r}  SKIP (validity blocked: {blocked})")
            continue

        cfg = read_config(config_path)
        if transition.ged in cfg:
            update_in_config(config_path, transition.ged, entry)
        else:
            append_to_config(config_path, transition.ged, entry)

        if ensure_validity_entry(
            validity, transition.timestamp, config_name, categories
        ):
            validity_changed = True

        # print(f"  ADD    {transition.ged:12s}  {transition.period} {transition.run} {transition.run_type:3s}"
        #      f"  → {transition.usability!r}  [{config_name}]")
        written += 1

    return written, validity_changed


def _apply_updates(
    updates: list[tuple[Transition, Transition]],
    statuses_dir: Path,
    validity: list,
    runinfo: dict,
) -> tuple[int, bool]:
    written = 0
    validity_changed = False

    for disk_transition, excel_transition in updates:
        entry = _build_entry(excel_transition)
        # label = (
        #    f"{disk_transition.usability!r} → {excel_transition.usability!r}"
        #    if disk_transition.usability != excel_transition.usability
        #    else f"{disk_transition.usability!r} (reason update)"
        # )

        if disk_transition.source_file:
            # Update the existing config file in place.
            update_in_config(
                statuses_dir / disk_transition.source_file, disk_transition.ged, entry
            )
            # (f"  UPDATE {disk_transition.ged:12s}  {disk_transition.period} {disk_transition.run}"
            #      f" {disk_transition.run_type:3s}  {label}  [{disk_transition.source_file}]")
            written += 1
        else:
            # No config file at this timestamp — treat as an addition.
            config_name = _target_config_name(excel_transition, validity, runinfo)
            config_path = statuses_dir / config_name
            categories = _categories_for(excel_transition.period)

            blocked = validity_blocked(validity, excel_transition.timestamp)
            if blocked:
                # print(f"  UPDATE {disk_transition.ged:12s}  {disk_transition.period} {disk_transition.run}"
                #      f" {disk_transition.run_type:3s}  {label}  SKIP (validity blocked:"
                #      f" {blocked})")
                continue

            cfg = read_config(config_path)  # {} if does not exist
            if disk_transition.ged in cfg:
                update_in_config(config_path, disk_transition.ged, entry)
            else:
                append_to_config(config_path, disk_transition.ged, entry)

            if ensure_validity_entry(
                validity, excel_transition.timestamp, config_name, categories
            ):
                validity_changed = True

            # print(f"  UPDATE {disk_transition.ged:12s}  {disk_transition.period} {disk_transition.run}"
            #      f" {disk_transition.run_type:3s}  {label}  [{config_name}] (new file)")
            written += 1

    return written, validity_changed


def _is_reset_source(validity: list, config_name: str) -> bool:
    """
    Return True if config_name appears in a mode:reset validity entry.

    Reset configs are the canonical full-state documents written at period
    boundaries by the legend-datasets team.  The sync tool must never remove
    individual detector entries from them — doing so leaves detectors with no
    status and causes blank cells in the dashboard.
    """
    return any(
        e.get("mode") == "reset" and config_name in e.get("apply", []) for e in validity
    )


def _apply_removes(
    removes: list[Transition],
    statuses_dir: Path,
    validity: list,
) -> int:
    written = 0

    for transition in removes:
        if not transition.source_file:
            # print(f"  REMOVE {transition.ged:12s}  {transition.period} {transition.run}"
            #      f" {transition.run_type:3s}  SKIP (no source file at this timestamp)")
            continue

        if _is_reset_source(validity, transition.source_file):
            # print(f"  REMOVE {transition.ged:12s}  {transition.period} {transition.run}"
            #      f" {transition.run_type:3s}  SKIP (reset config — edit manually if needed)")
            continue

        removed = remove_from_config(
            statuses_dir / transition.source_file, transition.ged
        )
        if removed:
            # print(f"  REMOVE {transition.ged:12s}  {transition.period} {transition.run}"
            #      f" {transition.run_type:3s}  removed from [{transition.source_file}]")
            written += 1
        else:
            # print(f"  REMOVE {transition.ged:12s}  {transition.period} {transition.run}"
            #      f" {transition.run_type:3s}  not found in [{transition.source_file}]")
            pass

    return written


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def sync(
    xlsx_path: str,
    strings: dict,
    periods: dict,
    datasets_path: Path = LEGEND_DATASETS,
) -> None:
    datasets = Path(datasets_path)
    statuses_dir = datasets / "statuses"
    validity_path = statuses_dir / "validity.yaml"

    with open(datasets / "runinfo.yaml") as f:
        runinfo = yaml.safe_load(f)

    with open(validity_path) as f:
        validity = yaml.safe_load(f) or []

    all_geds = [ged for snum in sorted(strings) for ged, _ in strings[snum]]

    disk_usab, disk_hist = build_from_disk(datasets, strings, periods, runinfo)

    # Seed build_from_excel with the on-disk usability at the very first Excel
    # event (typically p16 r000 cal — the reset-config anchor).  This prevents
    # detectors whose cell value already matches the reset baseline from
    # generating spurious transitions, while still recording genuine changes.
    events = _ordered_entries(periods, runinfo)
    initial_prev: dict[str, str | None] = {}
    if events:
        fp, fr, frt = events[0]
        initial_prev = {
            ged: disk_usab.get(ged, {}).get(fp, {}).get(fr, {}).get(frt)
            for ged in all_geds
        }

    excel_usab, excel_hist = build_from_excel(
        xlsx_path, strings, periods, runinfo, prev_val_seed=initial_prev
    )

    adds, updates, removes = _diff(
        disk_usab, excel_usab, disk_hist, excel_hist, all_geds, periods, runinfo
    )

    # Suppress phy ADDs that are already handled by the cal-config + remove
    # pattern: writing the cal change into the cal-config is enough — the
    # existing remove entry at phy start will revert the value automatically.
    cal_only_runs = {
        (a.ged, a.period, a.run)
        for a in adds
        if a.run_type == "cal"
        and _cal_config_removed_at_phy(a.period, a.run, validity, runinfo)
    }
    adds = [
        a
        for a in adds
        if not (a.run_type == "phy" and (a.ged, a.period, a.run) in cal_only_runs)
    ]

    if not adds and not updates and not removes:
        # print("No differences found — legend-datasets is already up to date.")
        return

    # print(f"\n{len(adds)} addition(s)  |  {len(updates)} update(s)"
    #      f"  |  {len(removes)} removal(s)\n")

    added, v1 = _apply_adds(adds, statuses_dir, validity, runinfo)
    updated, v2 = _apply_updates(updates, statuses_dir, validity, runinfo)
    _apply_removes(removes, statuses_dir, validity)

    validity_changed = v1 or v2
    if validity_changed:
        with open(validity_path, "w") as f:
            yaml.dump(
                validity,
                f,
                default_flow_style=False,
                sort_keys=False,
                allow_unicode=True,
            )

    # total = added + updated + removed
    # print(f"\nDone — {total} change(s) applied"
    #      f" ({added} added, {updated} updated, {removed} removed).")


if __name__ == "__main__":
    from create_dashboard import periods, strings

    sync("dashboard_output.xlsx", strings, periods)
