## General code guidelines

Prefer short, targeted changes. Inline comments should fit on the line next to
the code they refer to; if code needs a long comment, make the code clearer instead.
Docstrings follow numpy convention:

```python
def func(a, b):
    """
    One-line summary.

    Parameters
    ----------
    a : str
        description
    b : float
        description

    Returns
    -------
    int
        description
    """
```

## Github guidelines

Always run `pre-commit run -a` before committing. Keep commit messages, PR bodies,
and history short and clean; avoid too many commits. Substantial AI contributions
must be disclosed in the PR (see `AI_POLICY.md`).

Push to the `ggmarshall` fork first; upstream is `legend-exp/legend-data-monitor`.

## Dev commands

- Venv: uv-managed `.venv` (Python 3.11), package installed editable.
  Recreate: `uv venv --python 3.11 .venv && uv pip install -p .venv/bin/python -e ".[test]"`
- Tests: `.venv/bin/python -m pytest -q` (single test: `pytest tests/<dir>/<file>.py::<test> -q`)
- pytest is strict: warnings are errors, `--strict-markers --strict-config`. A new
  FutureWarning from pandas/numpy fails the suite. Use `1h` not `1H` offset aliases.
- 4 pre-existing failures in `tests/monitoring/` are known baseline — see
  `tests/baseline/BASELINE.md`; don't treat them as new breakage.
- Behavior changes must keep the golden snapshot bit-identical:
  `.venv/bin/python tests/baseline/snapshot.py <out> tests/baseline/golden_p19_r001.json --check`
- Lint is black + ruff (`.ruff.toml`; ruff covers isort and pyupgrade too),
  numpy docstrings enforced, `print()`
  banned in `src/` (use the logger). mypy is manual-stage only.

## Architecture invariants

- Only `contract/writer.py` writes monitoring output files; plots never write HDF.
- `legend_data_monitor/__init__.py` imports lazily — a bare import must stay ~0.01 s
  with no matplotlib. Don't add eager imports there.
- Library code raises typed exceptions from `errors.py`, never `sys.exit`;
  `run.py` is the sole exit-code owner (0 ok, 1 task failure, 2 config/env error).
- Root `issues.py` / `tasks.py` are re-export shims — edit `contract/issues.py` /
  `orchestration/tasks.py` instead.
- The v1 pandas writer (`save_data.save_hdf` pivots) still runs alongside the
  contract writer. Retiring it -- and with it the `-schema2` infix and the
  plotting/save_hdf coupling -- waits on every consumer reading v2.

## Gotchas

- `.gitignore` blanket-ignores `*.yaml`/`*.json` repo-wide (exceptions only for
  packaged settings/examples). New YAML/JSON elsewhere needs `git add -f` —
  check `git status --ignored` before assuming a file is committed.
- Cluster prodenv roots are hardcoded in `automatic_run.py`; use `--prod_root`
  for local/mock trees. Full e2e coverage (and any cal-tier data) exists only
  on-cluster.

## Repo specific

Outputs feed the Panel dashboard (legend-exp/legend-monitor-dashboard), which reads
contract v2 with plain h5py/json — keep outputs lightweight so it stays fast. Runs
on a shared machine and single runs are hours long / multi-GB: memory or disk
regressions are bugs.

The log/issue formats consumed by auto-giorgio (ggmarshall/auto-giorgio) are a
frozen contract — see `docs/auto-giorgio-integration.md` before changing log lines,
`ERROR ... END ERROR` / `ISSUE ... END ISSUE` blocks, or the issues JSONL schema.
Warnings and flags for detector instabilities should stay clear and machine-parseable.
