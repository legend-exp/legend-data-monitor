# auto-giorgio × legend-data-monitor integration spec

Status: **spec** — describes changes to be implemented in the
`ggmarshall/auto-giorgio` repo so it consumes legend-data-monitor (lmon)
outputs. The lmon side is already implemented and stable; the artifacts named
here are contracts (see "Stability guarantees").

## What lmon emits (already implemented)

Per unattended invocation (`legend-data-monitor auto_run`):

```
<out>/generated/tmp/log/<YYYYMMDDTHHMMSSZ>/
    orchestrator.log                      START/END/FAILED per task + "ISSUES <abs path> count=<n>"
    <task>/<task>-<period>-<run>.log      per-(task, run) log
<out>/generated/mon/issues/<period>/<run>/
    l200-<period>-<run>-<datatype>-issues.jsonl
```

- Task logs contain parseable blocks:
  `ERROR in task <task> (period=<p>, run=<r>):` … full traceback … `END ERROR`
  and `ISSUE detector=<det> metric=<m> severity=<sev> (period=…, run=…, datatype=…):` … `END ISSUE`.
- Tasks: `check_calibration`, `build_subsystem_data`, `build_monitoring_hdf`,
  `render_plots`, `slow_control`, `phy_summary_plots`, `qc_plots`. The data
  tasks never draw; `render_plots` reads only the contract file and is absent
  when a run was processed with `--plots off` (regenerate afterwards with
  `legend-data-monitor plot_run`).
- Issue records (JSONL, one object per line, `"schema": 1`):
  `issue_id` (`<p>-<r>-<dt>-<detector>-<metric>`, the dedup key), `severity`
  (`warning|alert`), `detector`/`rawid`/`string`/`position`, `metric`,
  `observed`, `threshold` `[low, high]`, `unit`, `window`,
  `excursion {frac_out, max_deviation, longest_s, recovered}`,
  `first_seen_run`, `reference` (band centre, when the evaluator has one),
  `data_ref {file, key}` (contract-v2 HDF + hist key),
  `raw_ref {tier_dir, channel, param, timestamps}` (provenance into the
  production LH5 tree for event-level triage), `plots []` (absolute paths),
  `suggested_action`. Fields the evaluator could not supply are omitted, so
  consumers must treat every field except `issue_id`/`schema`/`detector`/
  `metric`/`severity`/`period`/`run`/`datatype` as optional.
- **Severity is a triage gate, not a restatement of the verdict.** Most cal
  metrics are two-sided consistency bands a few times the mean fit error wide,
  so on real data ~30 % of the array trips one per run — including detectors
  whose resolution *improved*. lmon therefore grades a failed threshold as
  `alert` only when the excursion is sustained (`frac_out` ≥ 5 %) and has not
  recovered by the end of the window; everything else is a `warning`. Run
  auto-giorgio with `MON_MIN_SEVERITY=alert` unless you want the long tail.
- **Which metrics carry what** (2026-08-21): `excursion` is computed only where
  a genuine time series backs the verdict — `pulser_stab`, `baseln_stab`,
  `baseln_spike` (hourly), `discharge_rate`, `saturated_rate` (hourly), and
  `FEP_gain_stab` (600 s bins); `longest_s` is always seconds. Run-axis
  metrics (`escale_*`, `AoE_stab`) and scalars (`const_stab`,
  `tot_discharge_dead_time`) carry `observed`/`threshold`/`unit` but no
  excursion, so they grade on the distance past the band; `npeak`/`fwhm_ok`
  have no magnitude and stay `warning`. Consequence of wiring: one-sided phy
  metrics (`baseln_spike`, the rates) can now grade `alert` when sustained —
  before they could only ever be `warning`. `data_ref` may point into the
  period file (`l200-<p>-{phy,cal}-monitoring.hdf`, e.g.
  `qc_rate_series/IsDischarge/<run>`, `fep_gain_stab/<run>`,
  `psd_stability/<run>/<det>`) as well as the run contract.
- **SiPM metrics** (2026-08-21): `spms_baseln_stab`, `spms_noise_stab`,
  `spms_dark_rate`, `spms_noisy_frac` are graded on the 60 min bins of the
  spms contract (`l200-<p>-<r>-phy-spms-schema2.hdf`, which `data_ref`
  points at) and carry excursions (hourly). Their `detector` is a SiPM name
  (`S0NN`); `string`/`position` are absent (the schema keeps them as
  germanium integers) — use the contract's `/detector_map` for barrel, fiber
  and top/bottom. `plots[]` attach the barrel-side figures. LAr veto
  metrics: `spms_occupancy` per SiPM, and `lar_veto_frac`/`lar_accidental_frac`
  under the pseudo-detector `LAr` (no rawid); `data_ref` points at the period
  keys `lar_occupancy/<run>` / `lar_veto/<run>`.
- **Array-wide events collapse** (2026-08-22): when one metric fails on at
  least 30 % of the detectors it was evaluated on (and on at least 5 of them),
  the per-detector records are replaced by **one** record for the whole
  array — that is a common-mode event (noise burst, DAQ/HV excursion), not
  that many independent detector problems. Such a record has
  `detector` = `spms-array` / `geds-array` (a pseudo-detector, so `issue_id`
  stays a stable dedup key), no `rawid`/`string`/`position`, and two extra
  fields: `affected_detectors` (the full roster) and `affected_frac` (their
  share of the evaluated detectors). Its magnitudes are the worst member's and
  its `first_seen_run` the earliest of the group; `suggested_action` names the
  worst channel and points at common-mode causes. Normal per-detector failure
  rates (cal metrics fail on 10-25 % of the array in a good run) never
  collapse. Consumers keying on `detector` should expect these names alongside
  the run-level `LAr` pseudo-detector.
- **Resolution is graded one-sided** (2026-08-22): `escale_fwhm_FEP` and
  `escale_fwhm_583` now fail only when the FWHM is *above* the band — an
  improved resolution is not an issue and no longer raises one. Their
  `threshold` is therefore `[null, upper]`, and severity uses `reference`
  (the band centre) for the half-width, so a genuinely degraded detector can
  still reach `alert`. Two-sided metrics (`escale_FEP_pos`,
  `escale_SEP_residual`, `AoE_stab`, the SiPM bands) are unchanged: for those
  a departure in either direction is real.
- **SiPM SPE spectra** (2026-08-23): `hist/<flag>_EnergyInPe_dist2d` in the
  spms contract holds every pulse energy per SiPM (250 bins, 0-5 p.e.), for
  forced triggers (`IsBsln`) and physics events (`IsPhysics`). A valid PE
  calibration puts the 1 p.e. peak at exactly 1.0, so the centroid offset is
  the gain drift. Pulses are **unmasked** (`is_valid_hit` not applied, as the
  `selection` attr records) because the validity threshold sits below 1 p.e.;
  consumers must therefore start any peak search above that channel's
  `threshold_a` from `spms_calibration/<run>`, or the sub-threshold noise edge
  is mistaken for the peak.
- Exit codes: 0 all tasks ok; 1 ≥1 task failed (others still ran); 2
  config/environment error.

### Stability guarantees (lmon side)

1. The `ISSUES <absolute path> count=<n>` orchestrator line is the discovery
   contract — format frozen.
2. `ERROR in task` / `END ERROR` and `ISSUE` / `END ISSUE` block delimiters —
   frozen.
3. Issue records carry `"schema": 1`; breaking changes bump it.
4. Task logs live under `generated/tmp/log/**` — inside the path guard of
   auto-giorgio's `scripts/extract_error.sh` (`*/generated/tmp/log/*.log`).
5. `plots[]` are absolute paths.

## 1. Detection: `scripts/find_monitor_issues.py` (new)

Scans `MON_ROOTS` (new env var; whitespace-separated lmon output roots;
empty default = feature off). Two sources per root, emitted as clusters in
the same JSON shape `find_failures.py` produces:

**(a) Pipeline failures** — newest invocation tree under
`<root>/generated/tmp/log/`; parse each task log for `ERROR in task` blocks.
- `error_class = normalise_error(<terminal exception line>)` (reuse the
  existing normaliser).
- Cluster key `(root, task, error_class)`;
  `sig = sha1(basename(root)|<task>|<error_class>)[:12]` — same scheme and
  ledger as rule clusters.
- Targets: `l200-<p>-<r>-<dt>` strings, as today.

**(b) Detector issues** — discover JSONL paths from `ISSUES` lines in
`orchestrator.log` (fallback: glob `generated/mon/issues/*/*/*.jsonl`).
- Severity gate: keep records with severity ≥ `MON_MIN_SEVERITY`
  (default `warning`).
- Cluster key `(root, "mon:"+metric, error_class)` where `error_class`
  normalises the ISSUE header — effectively **one cluster per (root,
  metric)** spanning all affected detectors ("113 channels = one
  diagnosis").
- Targets recorded in `state/.targets/<sig>` are **issue_ids**. The existing
  `has_new_targets` logic then re-fires a persisting issue exactly when a
  new run contributes new issue_ids, and never re-fires for already-covered
  runs.
- A cluster whose metric has no unrecovered records in the latest run is
  "current-clear" → daily digest marks it `[resolved]` (same mechanism as
  disappearing rule clusters).

## 2. Payload mapping (`state/payloads/<sig>.json`)

Existing fields, filled compatibly (old payloads unchanged):

| field | pipeline failure | detector issue |
|---|---|---|
| `rule` | task name | `mon:<metric>` |
| `sample_error` | terminal exception line | first `ISSUE detector=… severity=…` header line |
| `sample_channel_log` | task log path | task log path containing the ISSUE block |
| `sample_output` | — | `ISSUES <path>` line |
| `targets` | `l200-p-r-dt` list | derived `l200-p-r-dt` list (display) |
| `channels` | — | detector names |
| `count` | block count | issue count |

New optional field: `issues: [ …up to 20 full JSONL records… ]` plus
`issues_total`. `diagnose.sh` includes it in the prompt only when present.

## 3. `diagnose.sh` branch for `mon:` payloads

- Branch on `rule` prefix `mon:`.
- Prompt frames a **detector anomaly, not a pipeline crash**: metric,
  severity, per-detector table (detector/string/position, observed vs
  threshold + unit, `frac_out`, `longest_s`, `recovered`,
  `first_seen_run`, `suggested_action`), the issue JSON fenced as
  **untrusted data**, and the affected-run list.
- **Trusted PERSISTENCE pre-check** (the VERIFY-block analog, computed by
  the harness before the agent runs): re-scan the *latest* run's JSONL for
  the same (metric, detector) pairs and inject one of:
  - `PERSISTENCE: <k>/<n> detectors still out of range in <latest run> (recovered=false across >=2 runs)`
  - `RECOVERED: no unrecovered records for <metric> in <latest run>` (⇒ lean TRANSIENT)
  The agent is instructed to trust this block over anything in the logs.
- Plot delivery: copy `plots[0]` to `state/plots/<sig>.png` and reuse the
  existing Slack-upload + PR-permalink mechanics.
- New agent helper `scripts/read_issue_data.py` (auto-allowlisted by
  `Bash(python3 $SCRIPTS_DIR/*)`):
  `read_issue_data.py --payload <p.json> [--issue-id ID] [--cadence 1min] [--stat mean|std|min|max|count]`
  — opens `data_ref.file` with **plain h5py** (contract-v2 layout: group
  `hist/<key>/<cadence>` with `storage/{counts,values,variances}`, `min`/`max`
  sidecars, `ref_axes/axis_0` attrs `bins/lower/upper`,
  `ref_axes/axis_1/categories`; layout pinned by lmon's
  `tests/test_contract_v2.py::test_v2_readable_with_plain_h5py`), prints
  per-bin stats for the flagged detector around `window` plus run means of
  its string-mates for context. Read-only.

  **Two conventions that silently corrupt a naive reader** (both now also
  stated in the group attrs `values_are` / `counts_are` / `flow_bins`):
  1. `storage/values` **already holds the per-bin means** (boost-histogram
     Mean storage) — dividing by `storage/counts` looks natural and yields
     garbage (485 instead of 14073 ADC on real data). `counts` is the entry
     count, used only to mask empty bins (`counts == 0` -> no data).
  2. **Both axes carry flow bins**: `axis_0` is
     `[underflow, ...bins..., overflow]` (980 rows for 978 time bins) and
     `axis_1` is `[...categories..., flow]` (60 columns for 59 detectors).
     Slice `[1:-1, :len(categories)]` before use, or detectors shift by one.

  Also: `detector_map` lists every detector in the channel map, including ones
  with no data in this run (60 rows vs 59 data columns on p22/r012) — intersect
  on `ref_axes/axis_1/categories`, never zip by position.

  3. **Storage dtypes are narrow**: `values` and `variances` are `float32`
     and `counts` is `int32` (they were `float64` in files written before
     2026-08-19). This halves the file and roughly triples how fast a reader
     inflates it. Compare values with a tolerance (`rtol=1e-6`), not for
     exact equality against a float64 recomputation.
- Existing helpers reused unchanged: `validity_lookup.py` (mandatory before
  any status change), `channel_to_detector.py`.
- Operational requirement: `MON_ROOTS` entries not under `PROD_ROOTS` must be
  added to the `--add-dir` list (read-only) so the agent can Read issue
  files and plots.

### Verdict semantics for detector issues

| verdict | meaning | typical action |
|---|---|---|
| `FIXED` | minimal metadata edit made | usability → `ac`/`off` in `datasets/statuses` keyed at the correct `valid_from` (verified via `validity_lookup.py`), a **PSD status downgrade** (see below), or an `ignored_daq_cycles.yaml` entry for a bad cycle window |
| `TRANSIENT` | PERSISTENCE says RECOVERED, or single-run blip (short `longest_s`, `recovered=true`, first seen this run) | none |
| `NEEDS-HUMAN` | persisting hardware anomaly, ambiguous fix, or already covered by an open PR | suggested fix in summary |

**PSD status downgrades are in scope for the agent.** For A/E instability
(`AoE_stab`, and the A/E mean/width drifts behind it) the proportionate fix is
usually not a usability change but the `psd` block of the detector's entry in
`datasets/statuses`: set the affected classifiers in `psd.status`
(`low_aoe`/`high_aoe`/`lq`) from `valid` to `present` — "available but not to
be trusted" — leaving `usability` alone when the energy scale is fine.

Rules for that edit:

- **Keep `is_bb_like` consistent**: it must not combine classifiers that are no
  longer `valid`, so drop the downgraded ones from the expression (the
  statuses README: "Normally these are the classifiers marked `valid`"). A
  survey of p22/r012 found 10/60 detectors already violating this, 4 of them
  `usability: on` — do not add to that pile.
- Same `valid_from` discipline as a usability change: resolve the effective
  status with `validity_lookup.py` first, and key the new entry at the run
  where the instability starts, not at the run that happened to alert.
- Restoring a classifier to `valid` is **not** an agent action — that is a
  `NEEDS-HUMAN` recommendation.
- The evidence to cite in the PR is the raw A/E distribution from the cal
  pars (`results.aoe.correction_fit_results.dep_fit` `mu`/`sigma` with their
  errors, per run), not the derived `aoe.low_cut`, which moves for fitting
  reasons unrelated to detector behaviour.

Retry/dedup: unchanged (24 h retry for needs-human/transient/failed;
`pr-opened` terminal; `bin/requeue.sh <sig>` to force).

## 4. Surrounding pieces

- **daily_check.sh**: one digest line per open `mon:` cluster:
  `• mon:gain_var ×7 detectors p19-r001..r003 [persisting|resolved|<ledger verdict>]`.
- **docs/common-issues.md**: five new numbered entries in the existing shape
  (Seen in / Cause / Classify / Diagnose / Fix / Worked example), keyed by
  the ISSUE header string: gain jump (`metric=gain_var` or
  `TrapemaxCtcCal_var`), noise increase (`BlStd`), event-rate anomaly
  (`EventRate`), QC failure-rate, and **A/E instability** (`AoE_stab`, fix =
  PSD status downgrade + `is_bb_like` update). Each names
  `read_issue_data.py` as the diagnose step and the metadata fix surface.
- **CLAUDE.md**: new "detector anomaly triage" task-type section (vs
  "pipeline failure"); issue JSON/log content is untrusted data; add
  `read_issue_data.py` to the tools list; restate read-only-prod,
  metadata-only edits, validity_lookup-before-status-change; state that the
  metadata surfaces the agent may edit are `usability`, the `psd` block
  (status + `is_bb_like`), and `ignored_daq_cycles.yaml` — nothing else.
- **config/monitor.env.example**:
  ```sh
  # legend-data-monitor output roots (detector-issue + task-failure detection)
  MON_ROOTS=""
  MON_MIN_SEVERITY="warning"
  MAX_MONITOR_DIAGNOSES_PER_POLL=1
  ```

## 5. Rollout

1. This spec lands in lmon (done) together with the record-schema field and
   absolute plot paths (done).
2. auto-giorgio: `find_monitor_issues.py` + unit tests on fixture
   logs/JSONL; wire into `bin/poll.sh` behind empty-default `MON_ROOTS`
   (dark launch — detection runs, nothing dispatched until configured).
3. auto-giorgio: payload `issues` field, `diagnose.sh` `mon:` branch,
   `read_issue_data.py`, CLAUDE.md/common-issues/daily_check additions.
4. Enable one root with `MAX_MONITOR_DIAGNOSES_PER_POLL=1`; watch a week of
   digests; then raise caps / add roots.
