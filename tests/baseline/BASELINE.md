# Phase 0 baseline record (2026-08-10)

Environment: `.venv` (Python 3.11.10, uv, `pip install -e ".[test]"` + undeclared deps
`pytz`, `h5py`, `tables` added manually — Phase 1 declares them properly).

## Pytest baseline

`188 passed, 4 failed` (pre-existing, not env-caused):

- `tests/monitoring/test_get_calib_data_dict.py::test_get_calib_data_dict`
- `tests/monitoring/test_get_calib_data_dict.py::test_channel_name_used_if_not_ch_key`
- `tests/monitoring/test_get_dfs.py::test_get_dfs_with_valid_files`
- `tests/monitoring/test_get_dfs.py::test_geds_pulser_correction_and_empty_pulser`

The `get_dfs` pair fail because `get_dfs` silently returns `None` frames (the
silent-fallback antipattern the refactor removes).

## E2e baseline on the mock production tree (`/Users/georgemarshall/mock_prod`)

Pipeline A (`legend-data-monitor user_prod --config tests/baseline/mock_user_prod.yaml`,
p19/r001 phy, 3 parameters × 200 channels): **14.8 s wall**, produces
`plt/hit/phy/p19/r001/l200-p19-r001-phy-{geds,pulser01ana}.hdf` (30 + 12 keys),
a PDF, and a log under `tmp/mtg/`.

Pipeline B (`monitoring.build_new_files(<out>, "p19", "r001")`): **3.3 s wall**,
produces `-res_10min.hdf` / `-res_60min.hdf` (27 keys each) + `-geds-info.yaml`.

Golden snapshot: `tests/baseline/golden_p19_r001.json` (written and self-checked by
`tests/baseline/snapshot.py`). Regenerate/verify:

```sh
.venv/bin/legend-data-monitor user_prod --config tests/baseline/mock_user_prod.yaml
.venv/bin/python -c "from legend_data_monitor import monitoring; monitoring.build_new_files('<out>', 'p19', 'r001')"
.venv/bin/python tests/baseline/snapshot.py <out> tests/baseline/golden_p19_r001.json --check
```

## Findings / limitations discovered at baseline

- `auto_run` is broken from a clean checkout: it requires `settings/geds-dict.yaml`
  which was absent from the repo (exists only on the cluster?). A minimal
  reconstruction (3 pulser-event parameters matching the mock data) was added at
  `src/legend_data_monitor/settings/geds-dict.yaml`.
- `auto_run --ref_version <absolute path>` escapes the hardcoded cluster dir because
  `os.path.join` drops the base — used for local runs until Phase 3 makes the root a
  proper argument.
- Mock-tree coverage is split: phy tier data exists only for p19; cal par data only
  for p14/p15; no cal LH5 tier data at all. Therefore `check_calib` and the cal
  monitoring stage have **no e2e baseline** — their refactor verification relies on
  unit tests only.
- pandas 3 removed the `H`/`T` offset aliases; configs must use `1h` (the example
  configs in `src/legend_data_monitor/config/` still say `1H` and would crash).
- Timings are for the small mock dataset; the ~2,000-redundant-parse hot loop
  (`plot_time_series`) is in the unexercised cal/monitoring path, so Phase 2 speedups
  must be measured with targeted micro-benchmarks, not just this e2e.
