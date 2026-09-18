# Refactor status (2026-08-10)

Plan: `~/.claude/plans/i-want-to-refactor-hidden-trinket.md`. Verification harness:
`tests/baseline/` (golden snapshot of the p19/r001 e2e run on the mock tree).

## Completed

**Phase 0 — baseline.** Golden harness, timings, pytest baseline (4 pre-existing
failures recorded in `tests/baseline/BASELINE.md`).

**Phase 1 — de-bloat.** `excel/` (2.6k LOC) and dead code deleted; real deps declared
(`requires-python >=3.10`); `rstrip` suffix bugs fixed; lazy `__init__` (bare import:
0.01 s, no matplotlib); rcParams no longer mutated at import; pandas-3 alias fixes.

**Phase 2 — performance.** Memoized calib-file/validity/run-times/status/detector-info
loaders (the per-(channel×run) ~2,000-redundant-parse hot loop now hits caches);
`lh5.ls` and `shelve.open` hoisted out of per-detector loops; awkward event loop
vectorized; concat-in-loop and 5×-copy patterns removed. Golden outputs bit-identical.

**Phase 3 — errors/logging/issues (auto-giorgio).** All 45 `sys.exit` in library code →
typed exceptions (`errors.py`); CLI sole exit-code owner (0/1/2). `auto_run` restructured
into isolated tasks (`orchestration/tasks.py`) with per-task logs under
`generated/tmp/log/<ts>/<task>/` and parseable `ERROR in task … END ERROR` blocks —
verified live (cal task fails on mock tree, remaining tasks still run, exit 1).
Detector issues: `contract/issues.py` (JSONL + `ISSUE … END ISSUE` blocks + excursion
evaluator with frac_out/longest/recovered triage fields); `send_email_alert` deleted.
`--prod_root` flag for local/mock production trees.

**Phase 4 — module split.** `monitoring.py` 2,974→~2,000: plot-free code moved verbatim
to `processing/series.py` + `loading/calib_files.py` (re-export shims keep old imports
working). Settings constants → `config/settings.py` (single source of key vocabulary).
`contract/`, `orchestration/`, `plots/` packages created; example configs → `examples/`.

**Phase 5 (core) — file contract v2.** Fully binned backend:
- `processing/binning.py` — (time × detector-name) Mean-storage boost-histograms
  (count/mean/variance per bin) + min/max sidecars; cadence-aligned so 10min/60min are
  exact lossless rebins of the 1min base; period merge = histogram sum.
- `contract/{schema,writer,reader}.py` — UHI-HDF5 serialization, per-key unit/label/
  limits attrs, `/detector_map`, root attr `lmon_schema_version=2`, run `manifest.json`
  with key inventory + vocabulary + IGNORE_KEYS ranges exported as *flagged* (kept,
  shaded) while REMOVE_KEYS is applied writer-side (dropped).
- `contract/build.py` — converts a run's v1 HDF into the v2 file
  (`…-geds-schema2.hdf` during the migration window); wired into the
  `build_monitoring_hdf` task; verified e2e on the mock tree (60 detectors, 82 keys).
- `plots/timeseries.py` — PNG renderer (mean ± σ band + min/max envelope) with
  `SAVED_PLOT` log lines.
- 21 contract tests incl. plain-h5py (no lmon import) compatibility.

## Full-run verification (2026-08-18)

p22/r012 rerun end to end into a fresh tree and diffed against the pre-change
golden snapshot: **exit 0, 3 h 11 m (was 5 h 00 m), peak RSS 17.3 GB (was 24
GB)**. 196/200 v1 HDF keys byte-identical, 340/340 contract hist keys and the
manifest identical, plus the new period-contract keys and figures.

Two differences, both explained:

1. **A data-corruption bug in the old loader, fixed.** The 4 non-identical keys
   are all `*_IsValidBlPolyRmsClassifier`, differing only for 6 detectors
   (rawids 1080003, 1105602, 1107203, 1108802, 1112000, 1112001). Those
   channels genuinely lack `is_valid_bl_poly_rms_classifier` in the hit tier
   (they carry `is_valid_bl_poly_rms`). The DataLoader path filled them with
   uninitialised memory — denormals around 1.5e-319 against a real
   -5.19..6315.89 range — which reached the v1 file, the contract and the
   dashboard. The direct loader yields NaN; pinned by
   `tests/test_direct_loader.py`.
2. `qcp` cal differs for one detector (V09724A) because `check_escale` builds
   each detector's multi-run band from `os.listdir()` over the **live** `/data2`
   tree, and r014 landed after the golden was taken.

**Golden diffs must therefore separate static outputs (r012 phy data, which
must match) from cal-history-derived outputs (which legitimately drift as
production adds runs).**

## Output size and memory (2026-08-19)

The pipeline was writing about 7x the disk it needed and holding about 2.3x
the memory it needed. Both are now fixed, verified value-for-value against
the previous output on a p22/r012 chunk: **200/200 v1 keys agree to float32
epsilon (worst relative difference 6e-8), same key set, same indices**; the
rebuilt contract file matches the old one across all 4006 datasets to the
same tolerance.

Measured on one 10-file chunk (production settings, `--plots off`):

| | before | after |
|---|---|---|
| peak RSS | 3.23 GB | **1.42 GB** |
| v1 file | 0.367 GB | **0.152 GB** |
| wall | 307 s | 224 s |

Per run at production chunk size the v1 file goes from **15.7 GB to ~2.2 GB**
(the extra factor over the chunk measurement is slack, below) and the contract
file from **1.06 GB to ~0.58 GB**, which also makes it ~3x faster to inflate --
the number the dashboard feels.

Size, in order of size:

1. **Slack, 2.7x.** An uncompressed fixed-format pandas key is a *contiguous*
   array, so overwriting it orphans the block it replaces -- and the append
   path rewrites every key it touches once per chunk. 15.34 GB of file for
   5.63 GB of data. Compressed keys are chunked and HDF5 hands the freed
   blocks straight back, so turning compression on removes the slack as a
   side effect (reproduced both ways in `tests/test_output_layout.py`).
2. **blosc:lz4 (`utils.HDF_COMPRESSION`), 2.6x with the dtype below.** Chosen
   over zstd/zlib, which are ~5% smaller but read twice as slowly: these files
   are re-read on every chunk, by the contract build, and by the dashboard.
3. **float32 pivots.** Values whose relative error at float32 is 6e-8, stored
   at float64.
4. **Contract storage narrowed** to float32 values/variances and int32 counts
   (`contract.writer._narrow_storage`). boost-histogram views are float64
   throughout; counts are integers. The group is assembled in an in-memory
   HDF5 file and copied across (`expand_refs` remaps `axes`' object
   references), because narrowing what uhi already wrote into the output
   leaves the float64 blocks orphaned there — 1.37x slack, measured.

Memory:

1. **Dead mean/variation columns for the QC entries.** `save_hdf` writes
   "absolute values ONLY" for `quality_cuts`, but `AnalysisData` still built
   `<param>_mean` and `<param>_var` for ~30 flags/classifiers -- 60 extra
   full-length columns (107 columns instead of 43) plus two whole-frame copies
   in the channel-mean join, all discarded. Skipped now.
2. **Two entries' frames alive at once.** Rebinding `data_analysis` in the
   plot loop frees the previous frame only *after* the next one is fully
   built, so the two largest objects in the run coexisted (~1 GB each).
   Explicit `del` at the end of each iteration.
3. **Rows before columns.** `params_to_get` carries every QC flag in the
   frame, so copying the subsystem and *then* filtering to pulser events cost
   ~15x what the entry needs.
4. **Native tier dtypes restored** (`utils.narrow_to_native_dtypes`): a
   channel missing a field NaN-fills it, and pandas widens the column to
   float64 even when every tier that has it stores float32 -- six classifiers
   on p22. Only that widening is undone: parameters the tier really stores as
   float64 keep their precision, because `value/mean - 1` in float32 loses
   most of the significant digits of a % variation (measured: 1e-5 % error on
   `TrapemaxCtcCal_var`, against a 0.05 % limit -- fine, but free to avoid).
5. **One fewer copy in the append path** (`del existing_data`; the `_var`
   recompute now overwrites the frame it just read instead of copying it).

`legend-data-monitor repack --output_folder ... --p p22 --r r000 ...` brings
runs produced before this over without re-running the pipeline: ~5 min a run
against ~3 h to regenerate, and it covers both file kinds (v1 pivots 7.1x,
contract 1.8x). It is idempotent, atomic per file, and never replaces a file
it did not manage to shrink.

## Classifier pivots stripped from the v1 file (2026-08-20)

Of the repacked 2.2 GB v1 file, **1.61 GB (73%) was the 28 QC classifier
pivots** -- 7 classifiers x 4 event types of event-level continuous float32
that barely compresses. They exist in the v1 file only as the *transport* to
the contract build, which bins them into `hist/<key>/<cadence>`; res_10min /
res_60min keep their own resampled copies. Decision (2026-08-19): classifiers
live only in the contract, so `repack.strip_classifier_pivots` removes the
pivots after the contract build.

- **Guarded**: refuses to touch the file unless the contract carries
  `hist/<key>/1min` for every key about to be removed -- a v1 file is never
  stripped of the only copy of its data. QC *flag* (boolean) keys,
  `_mean`/`_var`/`_info` keys and parameters all survive.
- Wired as a final `strip_transport` task that strips the period's
  *previous* runs only: qc_plots also reads the pivots (so the strip must run
  last), and the current run is still appending -- stripping it mid-run would
  make the contract rebuild bin only post-strip chunks. A finished run is
  stripped on the first invocation that processes a later run; a period's
  last run (and any backfill) is caught by `repack --strip-classifiers`.
- Verified on p22/r006: 200 -> 172 keys, all 172 survivors
  checksum-identical, livetime key selection unchanged
  (`IsPulser_AoeCustom` sorts before the classifier keys either way, pinned
  by test), contract classifier hists and the 28 res_10min copies intact,
  second pass a no-op. **v1: 2.20 -> 0.63 GB; a run's full artifact set is
  now ~1.2 GB against the original 17 GB (14x).**
- This intentionally breaks 200-key parity with old production:
  v1-only consumers lose event-level classifiers (binned versions remain in
  the contract and res files). New runs still pay the transient ~1.6 GB of
  pivots during the run -- they are the transport -- and drop them at the end.

Declined for now (2026-08-19): dropping the contract's 1min min/max sidecars
(226 MB, 40% of the file -- envelopes kept at all cadences) and the per-entry
loading refactor that would take peak RSS from ~5.4 to ~3 GB (accepted 5.4 GB:
r008-r012 measured 5.2-5.6 GB at production chunk size vs 14-17 before).

## Dashboard R3 on real data (2026-08-20)

The phy-v2 dashboard runs against real LNGS data (checkout
`/data1/users/marshall/phy-dash/legend-monitor-dashboard`, branch
`feat/contract-v2-phy` + main merge, commit b27ca85). Headless sweep of the
phy view on p22/r012: v2 96 OK / 0 FAIL vs `lmon-v2-p22`; v1 fallback 48 OK /
0 FAIL vs the live production tree (Histogram correctly v2-only). Two fixes
shaken out: contract-reader caches now key on (path, mtime_ns, size) — equal
mtime_ns across a rewrite on coarse-timestamp filesystems served stale
manifests — and the reader roundtrip test now asserts float32 tolerances
(the producer narrowed contract storage on 2026-08-19). Combos that render
empty (IsBsln continuous params, EventRate) are absent from production's
200-key v1 inventory too — widget options exceeding the produced key set is
pre-existing. Serving: legend-login2 port 9000 (v2) / 9001 (v1 production
tree); browser sign-off pending, then the branch gets pushed.

## Phase 5c — figures from the contract; shelve/pickle deleted (2026-08-20)

The eight generators are now compute -> contract write (+ YAML verdicts) only;
every figure is drawn from the contract by `plots/{qc,summary,stability,calib}`
with the legacy PDF names/locations preserved verbatim (the shifter cloud-upload
interface). `shelve`/`pickle` are gone from the package — the one exception is
`calibration.read_dataflow_stability`, which reads an *external* dataflow
shelve. `--write-shelves` is removed; `--plots off` is now truly data-only
(the summary/qc tasks used to draw regardless). `plot_run` regenerates the
complete figure set from the contract in seconds.

New with this pass: classifier `_dist2d` per-detector histograms in the run
contract (fixed ±15 range, flow bins catch outliers); `event_rate_qc`,
`escale/<run>`, `psd_stability/<run>/<det>`, gain/param `_std` companions,
`pul_cusp` trace and `res`/`res_quad` columns on `cal_points` in the period
files; `calibration.evaluate_escale_metrics` (data-only twin of the verdicts
the escale figure used to compute while drawing).

Latent legacy bugs found while porting, all fixed:

1. `plot_variable` iterated `for period in periods` with `periods` being the
   *string* "p22" — the per-period grouping and `exclude_period` never worked
   (single lumped series), and the escale issue details were recorded under
   period `'p'`, so `pop_detail("p22", ...)` never found them: cal issues
   shipped without observed/threshold magnitudes. Verdict math kept
   bit-compatible (global ON-mean); details now attach.
2. `qc_average` reset `dt_condition` per flag, so the IsSaturated iteration
   always overwrote the `tot_discharge_dead_time` verdict with True.
3. The corrected-branch `results` update did `ndarray.values` (AttributeError
   waiting for the first run where PULS01ANA correction applies to
   TrapemaxCtcCal).
4. The per-string classifier-figure shelve key had no string component, so
   every string overwrote the previous figure; and
   `IsValidBlSlopeRmsClassifier` was listed twice (duplicate rows/PDFs).
5. `check_psd`'s shared shelve path was missing the `mtg/` segment its own
   fallback used.

`collect_stability_series` (ex `plot_time_series`) also runs its period pass
once per detector instead of twice (it was repeated per correction type).

Verified on real p22/r012 data: a pre-deletion legacy reference produced 370
shifter PDFs; the new renderers on the same tree reproduce **exactly the same
370 names, every file within 0.87-1.00x size, zero outliers**. The full task
wiring (check_calib + summary_plots + qc_avg_series, render on) produced 657
PDFs across all families with **zero shelve files**, and the resulting
`qcp_summary.yaml` matches the production-era r012 file on **899/899
verdicts**. 329 tests green (+4 pre-existing failures), pre-commit clean.

## Baseline parameters: the bl_mean pivots were never whole (2026-08-20)

The dashboard's "Baseline Mean" view was empty because `IsPulser_BlMean`
really was: 4 one-minute bins for a week of data, its `_var` on a different
13 h axis. `save_data.get_pivot` decided whether a pivot was the absolute
values, the run mean or the % variation by **substring-matching `mean`/`var`
in the parameter name**, so `bl_mean` (and `pz_mean`) were treated as run
means: truncated to one row per chunk, `_var` overwritten per chunk instead
of recomputed, `_mean` never refreshed. Production v1 has carried this since
the heuristic was written (22 rows vs 38 619 for Baseline on p22/r012); the
contract build reproduced it faithfully. The role is now passed explicitly
(`kind="abs"|"mean"|"var"`) by every call site; the new test fails on the
old code for `bl_mean`/`pz_mean` and passes for `baseline`.

Found on the same trail and fixed: `contract/build._param_attrs` peeled one
suffix only, so every `*_pulser01ana{Ratio,Diff}_var` group shipped without
`label`/`unit`; `fill_distribution` ranged on min/max, so one 3000 ADC noise
burst put all of `BlStd_dist` into a single bin (now 0.5-99.5 percentiles,
flow bins keep the rest).

`legend-data-monitor repair_param` regenerates one parameter for finished
runs without the 3 h pipeline: it replays the parameter's config entries over
the run's recorded chunk lists into a scratch tree, transplants the keys into
the v1 files (geds + pulser01ana) and refreshes just those keys in the contract
(`build_contract_files(keys=)`, compacting afterwards). ~30 min per run; p22
r000-r013 repaired this way. Verified on r012: BlMean now 29 127 rows on the
same axis as Baseline, 575 781 populated contract bins for both.

Dashboard-side causes (Noise y-axis pinned to ±150 against 9-47 ADC data,
fixed ranges clipping the min/max envelope, the dead `IsBsln_<param>` menu
branch, `_uncamel` not inverting the producer's camel-caser) were handed to
the dashboard session with file:line references.

## SiPM / LAr monitoring, phase S1 (2026-08-21)

Nothing LAr-related worked end to end before (no spms entries in
`parameter-tiers.yaml`, so `Subsystem("spms").get_data` raised; the per-barrel
plot was an unrunnable draft; nothing was ever produced in production). First
pass, channel health only:

- **Loader reductions** (`processing/spms.py`, `settings/spms-reductions.yaml`):
  the ragged hit fields become per-event scalars (`n_pulses`, `pe_sum`,
  `pe_max`, `first_trigger_ns`) inside `load_channel_frame`, per file *before*
  the concat (pandas concatenates awkward-backed columns in Python: 22 s vs
  0.3 s for 2 channels x 3 files). Nothing ragged leaves the loader.
- **Subsystem**: `barrel` column (spms only) in the channel map; `is_spms()`
  keys on it instead of duck-typing dtypes; `channel_mean` runs for spms
  (so `_var` keys exist); the pulser01ana aux merge is geds-only.
- **Config**: `settings/spms-dict.yaml` (wf_mode/curr_fwhm/wf_lower_hwhm/
  n_pulses in FCbsln = forced triggers, pe_sum/pe_max in phy, has_any_noise in
  all; 10 min, `per barrel`). `auto_run` merges it with `geds-dict.yaml`.
  `check_plot_settings` refuses cross-subsystem structures.
- **Contract flavour**: `build_contract_files(..., subsystem=)` +
  `build_all_contract_files`; `-spms-schema2.hdf` with a
  `name/rawid/barrel/fiber/position/processable/usability` detector map,
  `limits[subsystem]` attrs, manifest merged across subsystem files.
  `refresh_contract` loops over the subsystem files present; the classifier
  strip stays geds-only.
- **Period keys**: `spms_noise/<run>` (hourly `baseline_curr_fwhm` x SiPM from
  `par_dsp_spms.yaml`) and `spms_calibration/<run>` (the `energy_in_pe` a/m and
  `is_valid_hit` threshold overrides in force, with their source file — on
  p22 it is `lar/p19/r005/...`, i.e. the PE calibration is three periods stale).
- Verified on a 3-key p22/r012 chunk: 58 SiPMs, all 7 families populated,
  FCbsln `n_pulses` ~0.2 per 100 us window (~2 kHz dark rate), one SiPM at a
  6 % noisy-waveform fraction against <1 % for the rest; 335 s, 0.75 GB peak.

- **Full run** (p22/r012, spms only, 163 keys in one unchunked pass): 89 min,
  12.3 GB peak (chunked `auto_run` will be far lower), v1 file 414 MB of which
  376 MB are the per-event `IsPhysics_PeSum/PeMax` pivots (+ their unused
  `_var`) — strip them like the classifiers once the dashboard reads spms.
- **Thresholds** (`mtg-plot-settings.yaml` `spms_*` entries, bands from that
  run) via `monitoring.check_spms_thresholds` in `phy_issues`: hourly
  `wf_mode` variation ±0.05 %, `curr_fwhm` variation ±5 %, dark rate 0.002–1
  pulses/window on a 6 h rolling mean (single empty hours are Poisson),
  noisy-waveform fraction < 5 %. On r012: S070 fails `spms_noisy_frac`
  (6.5 % all run, `alert`); everything else passes. Headline PNGs per
  barrel×position; issue records carry the spms `data_ref` and those plots.

- **v1 strip** generalised (`repack.strip_transport_pivots(subsystem=)`):
  spms drops every `IsPhysics_*`/`All_*` per-event pivot (keeps `_mean`),
  geds keeps the classifier rule; `strip_transport` and `repack
  --strip-classifiers` cover both files.

## Phase S2 — LAr veto performance (2026-08-21)

New task `lar_summary` (after `qc_plots`, ~20 s/run): `monitoring.
read_lar_events` reduces the evt tier per file (per-event scalars + a dense
event x SiPM "had a trigger-coincident pulse" matrix from
`spms/is_trig_coin_pulse`), `write_lar_summary` publishes `lar_veto/<run>`
(hourly `n_phys`, `veto_frac` of physics geds events, `accidental_frac` of
forced triggers = the veto's random-coincidence dead time, medians of
`energy_sum`/`multiplicity`, `first_t0_frac`, `classifier_median`) and
`lar_occupancy/<run>` (hourly x SiPM participation) to the period file, plus
`hist/IsPhysics_Lar{EnergySum,Multiplicity,Classifier}_dist` in the spms
contract. p22/r012: veto 0.73–0.85, accidentals 0.02–0.11, occupancy
0.12–0.22 per SiPM. Checks (`mtg-plot-settings.yaml` `lar_*` entries, 6 h
rolling): `lar_veto_frac` 0.6–0.95 and `lar_accidental_frac` < 0.2 under the
pseudo-detector `LAr`, `spms_occupancy` > 0.02 per SiPM; graded by
`check_spms_thresholds`, issue `data_ref` into the period file.

Still to do: p22 spms backfill is running (sequential, ~75 min/run, then the
strip); the dashboard SiPM/LAr pages need a rewrite against the spms contract
and the `lar_*` keys (dashboard session). The LLAMA page has no producer
anywhere (out of scope).

## Issue-stream semantics + p22 re-grade (2026-08-22)

Two changes to how verdicts become issues (`bc1e40a`), then the whole p22 issue
stream re-derived so the tree matches the code.

- **Array-wide collapse** (`issues.collapse_correlated`): one metric failing on
  >=30 % of the detectors it was evaluated on (and >=5 of them) is one
  common-mode event, not N detector problems. Those records become a single
  record on a `spms-array`/`geds-array` pseudo-detector carrying the worst
  member's magnitudes plus `affected_detectors` / `affected_frac`. The 30 %
  floor is calibrated on p22: per-detector cal metrics fail on 10-25 % of the
  array in a normal run (never collapse), the SiPM noise events on 33-100 %.
- **Resolution graded one-sided**: `escale_fwhm_FEP`/`escale_fwhm_583` fail only
  above the band -- an improved FWHM is not an issue. `classify_severity` grew a
  `reference` argument (the band centre the escale evaluator already recorded)
  so a one-sided band still has a half-width and a degraded detector can still
  reach `alert`.

**p22 re-grade** (14 runs, ~24 min each, verdicts re-derived from the
contract/period files -- no raw reloading; the stale stream predated the
excursion wiring and the SiPM checks). Against the pre-regrade backup
(`/data1/users/marshall/lmon-v2-p22-prerregrade-backup`):

- records 420 -> 383, but **magnitudes 93/420 -> 339/383** and **alerts 12 ->
  72**: severity finally discriminates instead of everything being `warning`.
- fwhm records 157 -> 71: **86 of them were detectors whose resolution had
  improved**. Every other cal metric identical (AoE_stab 37, FEP_gain_stab 16,
  const_stab 16, npeak 24...), and geds phy identical (pulser_stab 46,
  baseln_stab 45, baseln_spike 10) -- the re-grade is faithful; only the
  intended thing moved.
- 44 SiPM/LAr records appear (they were graded but never emitted), and
  `tot_discharge_dead_time` -- a *run-level* quantity that duplicates itself
  across all 59 detectors -- now emits once per run instead of 59 times.
- 12 array records stand in for 631 per-detector ones; the phy stream would be
  773 records uncollapsed, it is 154.
- The pass also wrote the period families that were missing since the backfill:
  `cal_points` res/res_quad, `param_stability/*_std`, `gain_shift/*_std`,
  `pul_cusp/kevdiff`, `event_rate_qc` -- all 14 runs.

Known follow-ups: scalar one-sided metrics (`tot_discharge_dead_time`, the
rates) have no `reference`, so they can never grade `alert` however far past
the band -- recording `reference=0` for fractions whose floor is zero would fix
it (and wants a re-grade to stay consistent). SiPM headline PNGs exist only for
r012, so array records attach no plots elsewhere.

## SiPM SPE spectra + manifest inventory (2026-08-23)

- `hist/<flag>_EnergyInPe_dist2d` in every spms contract (`88effcd`): all 58
  SiPMs' pulse energies, 250 bins over 0-5 p.e., forced triggers (`IsBsln`) and
  physics (`IsPhysics`) separately. A separate pass (`monitoring.write_spe_spectrum`,
  shaped like `write_lar_summary`) reads hit `energy_in_pe` + evt
  `trigger/is_forced`, which are row-aligned per file -- 4 min/run against the
  89-min build, so the whole p22 backfill took an hour. Pulses are unmasked on
  purpose (the validity threshold sits below 1 p.e.), which means a peak search
  must start above the channel's own `threshold_a`.
- **Finding**: 1 p.e. centroids span 0.89-1.25 on r012 (11/58 off by >5 %, 5/58
  by >10 %) -- real, previously unmonitored gain drift. The PE calibration in
  force is a spread of eight override files from p15 to p19, **35 of 58 SiPMs
  from p15/r004**; `read_spms_calibration` used to stamp them all with the
  newest file, now fixed to resolve provenance per SiPM.
- **Manifest inventory** (`a8dc122`): writers that add keys after the contract
  build left them out of `manifest["files"][...]["keys"]`, so manifest-trusting
  consumers were blind to the SPE keys *and* to the phase-S2 LAr `_dist` keys.
  `contract.build.refresh_manifest()` re-inventories from the files, and both
  post-build writers call it. All 14 p22 manifests rebuilt: 69 spms keys each.
## De-hardcoding pass (2026-09-18)

Review of the split PRs found the same shape of problem in several places:
L200 constants and one estimator's name inlined across modules. Two rules now:
**derive from the production metadata where it already says it**, and keep the
rest in one packaged file, `settings/experiment.yaml`.

- **Aux channels are derived, never configured.** `utils.aux_channels` reads
  the channel map: `puls` with/without the `ANA` suffix gives pulser and
  pulser01ana, the non-dummy `auxs` entry the muon, the `bsln` entry FCbsln.
  `subsystem.py` selects the same way, so the four rawids (1027200-1027203)
  are gone from the source. Verified identical on p22.
- **Energy estimator and peaks in settings.** `find_energy_key` returns which
  estimator matched and `uncalibrated_variable` derives the expression
  variable from it, so `cuspEmax_ctc` is no longer written down either. Peak
  energies, their pk_fits search windows, the FEP hit-tier window and the
  escale band widths come from the same file.
- **Bug found and fixed**: `extract_resolution_at_q_bb` asked for
  `cuspEmax_ctc_cal` by name, so a par file carrying only the *run*
  calibration returned NaN resolutions silently. It now goes through the
  estimator lookup like everything else.
- **Dead code found and removed**: `get_traptmax_tp0est` loaded four frames
  per run of which callers read exactly one, and its trapTmax key was
  misspelled (`IsPulser_TrapTmax` against the `IsPulser_Traptmax` the
  camel-caser writes), so that half never loaded anything. Its own test passed
  only because it wrote the same misspelling it read. Replaced by
  `get_spike_veto_series`, which reads the one series the spike veto needs;
  the veto parameter and window are settings.
- `processing/series.py` no longer names itself after one estimator
  (`get_pulser_data` takes explicit frames instead of a 7-element positional
  list, and its `ged`/`pul`/`diff` sub-dicts all use raw/rawdiff/kevdiff).
  The published `pul_cusp/...` period keys are unchanged.
- QC pseudo-parameters, headline PNG keys and the production slow-control
  list were each copy-pasted or inlined; all three now live in settings.

**YAML/JSON**: the run info file is named `.yaml` and was written with
`json.dump` — now written as YAML. The issues JSONL stays JSON (frozen
auto-giorgio contract, and line-appendable records are the point), as do the
JSON strings inside HDF5 attrs.

*Proposal, not yet done*: the run manifest could move to YAML too. Every JSON
document is valid YAML, so the safe order is (1) switch all readers -- lmon
and the dashboard's `contract_reader` -- to `yaml.safe_load`, which reads both
formats, (2) release the dashboard, (3) flip the writer and the
`-manifest.json` name. Needs coordination with legend-monitor-dashboard, so it
is deliberately left out of this pass.

## Muon veto (pmts) contract flavour (2026-08-31)

Producer side of the phy-dash request in
`/data1/users/marshall/phy-dash/pmts-contract-spec.md`: the dashboard's dead
Muon page gets rebuilt on contract keys the way the SiPM page was.

- 53 PMTs (pillbox 10 / floor 20 / wall 23). The channel map's `location` is a
  plain string (not the geds/spms dict) and there is no `analysis` block —
  usability/processable come from `datasets/statuses` (52 on / 1 off on p22).
- **The pmts dsp stream is sparse**: one row per muon-DAQ trigger (~0.08 Hz),
  and its timestamps match no other subsystem, so pulser/FCbsln/muon flags
  cannot attach (the flaggers already degrade to all-False on a timestamp
  mismatch); everything runs on `all` events (`settings/pmts-dict.yaml`:
  bl_mean var, bl_sig, pulseHeight, max/mean_peak_height, containsPulse,
  event_rate; grouped `per string` = per location).
- Contract flavour via the existing subsystem machinery: `pmts` added to
  `schema.SUBSYSTEMS` (refresh_contract picks it up), detector map
  `name/rawid/location/processable/usability`, `utils.build_pmts_info`.
- **Two spec adaptations, both from the data**: this production has **no
  `evt_muon` group** anywhere (that list is from a dev processing), and the
  PMT triggers share zero timestamps with the geds evt stream. So
  `muon_veto/<run>` is derived from what exists: muon-DAQ trigger rate,
  per-trigger multiplicity (PMTs with `containsPulse`) and summed light (LSB)
  from the dsp rows themselves, plus ge-coincidence fractions from
  `evt/coincident/muon(_offline)`. `_dist` keys for multiplicity and light
  sum, and the per-PMT `hist/All_Pulseheight_dist2d` (500 bins over a fixed
  (0,100) LSB — the SPP-position analogue of the SiPM SPE spectra; unmasked,
  no `containsPulse` cut) land in the pmts contract. New `muon_summary` task.
- On p22/r012: 0.08 Hz, multiplicity median 4, light sum ~80 LSB,
  ge-coincidence 0.6-0.7 % (offline flag all zero).
- **p22 backfill DONE 2026-09-01 03:09 UTC**, validated 14/14: pmts contracts
  ~62 MB each (manifest now lists three files per run), `muon_veto/<run>` x14.
  Rate 0.088-0.092 Hz on quiet runs; r000/r004/r011 elevated (0.11-0.13 Hz,
  r004 coincides with the SiPM noise burst); ge-coincidence stable 0.60-0.62 %.
  Producer response for the dashboard left at
  `/data1/users/marshall/phy-dash/pmts-contract-landed.md`. v1 pmts files are
  ~90 MB/run (a 1-min pivot of a sparse stream) -- strip/compact candidate.
- Also fixed on the way: a PMT absent from a file raises `LH5DecodeError`
  under current lh5 — now caught in the direct loader too (the old
  `KeyError/ValueError` catch predates it).

## pathlib and ruff (2026-09-18)

- **`os.path` -> `pathlib` throughout**: all 366 call sites across 24 modules.
  Public functions still accept and return `str` where that is a contract --
  the `SAVED_PLOT` / `ISSUES` log lines and the issue records' `plots[]` are
  consumed by auto-giorgio -- so `Path` is an internal representation only.
  `os` survives where it is not about paths (`os.environ`, `os.getpid`).
  Four tests that patched `os.path.exists`/`os.listdir` now build real
  directories instead: they were pinning the filesystem API rather than the
  behaviour, which is also why `test_nonexistent_path` had to patch at all
  (`/nonexistent` exists on some machines; it now uses an absent tmp_path).
- **flake8 -> ruff**, matching legend-dataflow and the other repos:
  `.ruff.toml` selects the families the flake8 plugins covered (bugbear,
  print, numpy docstrings, pep8-naming) plus isort and pyupgrade, so those two
  hooks are gone as well. The ignore list keeps enforcement identical to
  before: flake8-bugbear's opinionated `B9xx` were never enabled here, and the
  RUF-specific rules have no flake8 equivalent. They are listed with a comment
  and are worth enabling one at a time -- `B904` (`raise ... from`) and `B905`
  (`zip(strict=)`) especially.
  Ruff immediately earned its keep: it found `build_spms_info` defined twice
  in `utils.py`, a duplicate introduced by a merge resolution that flake8's
  configuration had not been catching.

## Remaining

1. ~~**Pickled-figure shelve writers**~~ **DONE (2026-08-20)**, see above. Was:
   **Pickled-figure shelve writers** in `monitoring.py` (qc_distributions,
   qc_and_evt_summary_plots, box_summary_plot, qc_average, qc_time_series,
   plot_time_series) and `calibration.py` (fep_gain_variation,
   evaluate_psd_usability_and_plot): split each into data-computation → contract
   writer (period-level `l200-<p>-{phy,cal}-monitoring.hdf`) + optional PNG via
   `plots/`; then delete `shelve`/`pickle.dumps` package-wide. Blocked on real/cal
   test data (mock tree has none) — build against a production tree or extend the
   mock tree with cal fixtures first.
2. ~~**Excursion wiring**~~ **DONE (2026-08-21, 94f7c81)**: every producer
   stashes its magnitudes; excursions where a real time series exists
   (pulser/baseline, qc rates, FEP bins), scalars elsewhere; phy issues emitted
   once by a final `phy_issues` task (the qc-rate verdicts never reached the
   emitter before). On p22/r012 cal issues went from 0 magnitude fields to
   24/26 and 7 now grade `alert` on distance past the band — including
   *improved* resolutions (symmetric bands); open question whether resolution
   metrics should grade one-sided.
3. ~~**slow_control** output under the contract~~ **DONE (2026-08-21, d9c2bbd)**:
   retrieval never worked (the diode-info merge ate every rack/clean-room
   row, so no SC file was ever written anywhere, production included); fixed,
   and each (parameter, run) is published as `slow_control/<param>/<run>` in
   the period contract file (UTC DatetimeIndex, value/unit/limits).
   Live-verified on p22/r012 (8 parameters, 1-9 k readings each); dashboard
   overlay reader change handed to the dashboard session.
4. **Retire v1 writer** (`save_data.save_hdf` pivots) once the dashboard reads v2 —
   then drop the `-schema2` infix and the plotting→save_hdf coupling disappears with it.
5. ~~**Phase 6 — dashboard phy migration**~~ **DONE (2026-08-11)**: the dashboard
   phy view is manifest-driven (`src/legenddashboard/geds/phy/contract_reader.py`,
   plain h5py/json, no lmon import — subprocess-enforced). v2 path: cadence-key
   selection replaces read-time resampling; detector-name columns replace the
   rawid maps; ±σ bands + min/max envelopes + flagged-range shading; working
   Histogram view from `_dist` (the old broken one deleted); v1 fallback kept
   for manifest-less runs. Bugs fixed on the way: missing `period` dependency,
   hardcoded `l200-`, in-place index mutation, indistinguishable empty-figure
   fallbacks. Tests: `tests/test_phy_contract_reader.py` (10) + headless smoke
   across units × cadences × styles on the mock tree.
   Still open on the dashboard side: cal-trend views on the period files
   (blocked on Phase-5c producers) and loosening the Dockerfile matplotlib pin
   (also needs legend-dataflow to stop shipping pickled figures).
6. **auto-giorgio integration**: spec in `docs/auto-giorgio-integration.md`
   (detection source, payload mapping, `mon:` diagnose branch, PERSISTENCE
   pre-check, verdict semantics, rollout). Implementation lives in the
   auto-giorgio repo. lmon-side hardening done: issue records carry
   `"schema": 1`, `plots[]` absolute, `ISSUES` line documented as frozen.
