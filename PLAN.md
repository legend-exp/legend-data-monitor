# Move the contract-v2 stack to LNGS and run on real data

## Context

The lmon refactor (phases 0–5 core) and the dashboard v2 wiring are validated against the local mock tree, but everything sits uncommitted: 56 files in the vendored `legend-data-monitor/` clone (on `main`, tracking `legend-exp/legend-data-monitor`) and the phy-v2 wiring in the dashboard repo (branch `feat/metadata-editor`, which also carries 6 unrelated in-progress modifications that predate this work). Goal: get the WIP onto the LNGS cluster as dev checkouts (no releases yet), run the pipeline against real `/data2` production data, and view the results in the dashboard. User decisions: target = LNGS; deploy = dev checkouts; lmon goes to the personal fork (`ggmarshall/legend-data-monitor`) first.

Two known-unverified areas that real data will exercise for the first time: the calibration task path (mock tree had no cal tier data — expect contained task failures; they are the Phase-5c backlog) and real-size performance.

**STATUS: Phase R2 COMPLETE** (2026-08-15): r010/r011/r012 all exit 0; r011 verified the orchestrator-log ISSUES lines end-to-end; per-run figs/ PNGs render correctly. Multi-run v2 coverage in place at `MON_ROOTS=/data1/users/marshall/lmon-v2-out/auto/v2.0.0`. Next: auto-giorgio integration (R4), dashboard (R3).

**STATUS: Phase R2 first run GREEN** (2026-08-15, on lngs): p22/r012 end-to-end exit 0 on `auto/v2.0.0` (fixes pushed as e3f0ae2); v2 manifest+schema2 validated via contract.reader; PNG renderer wired into build task (was dead code) and verified on real data; ISSUES line now mirrored to orchestrator.log (auto-giorgio discovery contract). Historical production geds-dict.yaml recovered verbatim from the live old deployment (`/data1/users/calgaro/legend-data-monitor`, cron, commit g5e91c04e5) — reproduces production's 200-key inventory exactly. r010/r011 sweep in flight (~5 h/run backfill). Perf findings for 5c: check_calibration 33 min (reuse dataflow `2614_stability` plt-shelve arrays instead), subsystem_plots 2h52 backfill (per-channel NFS reads, aux reloads), build_monitoring_hdf 85 min / 24 GB peak (frame_to_binned materialization); v1 geds.hdf 16 GB/run vs schema2 1.08 GB (15×) vs res_10min 70 MB.

**STATUS: Phase R1 is DONE** (2026-08-13, on the laptop):
- lmon refactor committed as 4 logical commits on `refactor/contract-v2`, pushed to `ggmarshall/legend-data-monitor` (upstream untouched)
- dashboard phy-v2 work committed on `feat/contract-v2-phy`, pushed to `legend-exp/legend-monitor-dashboard` (`boost-histogram`/`uhi` are TEST extras only — the runtime reader is plain h5py)
- `settings/geds-dict.yaml` now ships a full production parameter set; root cause of it missing was a blanket `*.yaml` in lmon's `.gitignore` (fixed with `!settings/*.yaml` exceptions). If the old deployment's historical geds-dict.yaml survives on the cluster, prefer it verbatim.

## Phase R2 — lmon on LNGS (run on lngs-login)

1. Dev area, e.g. `~/lmon-dev/`:
   ```sh
   git clone -b refactor/contract-v2 https://github.com/ggmarshall/legend-data-monitor
   cd legend-data-monitor
   uv venv --python 3.11 .venv && uv pip install -p .venv/bin/python -e ".[test]"
   .venv/bin/python -m pytest tests/ -q          # expect: pass except 4 pre-existing failures
   #   (tests/monitoring/test_get_calib_data_dict.py x2, test_get_dfs.py x2 — silent-None bug, pre-refactor)
   ```
2. First real-data run — **read-only inputs, outputs to scratch**:
   ```sh
   .venv/bin/legend-data-monitor auto_run --cluster lngs \
       --ref_version <current, e.g. tmp-auto or ref-vX.Y.Z> \
       --output_folder ~/lmon-v2-out --p <recent period> --r <recent run>
   ```
   (`--prod_root` exists if the prodenv base differs from `/data2/public/prodenv/prod-blind/`.)
3. Validate:
   - exit code + `~/lmon-v2-out/<ref>/generated/tmp/log/<ts>/` task logs (per-task files, `ERROR in task … END ERROR` blocks for anything that failed);
   - v2 artifacts: `.../plt/hit/phy/<p>/<r>/l200-<p>-<r>-manifest.json` + `-geds-schema2.hdf`; spot-read with `legend_data_monitor.contract.reader` (detector count, cadences, non-empty bins);
   - issues: `generated/mon/issues/...` + ISSUE blocks if thresholds fired;
   - **timings** per task from orchestrator.log — first real-scale performance numbers (compare subjectively against the old production runtime).
4. Triage failures. Expected hot spots, in order:
   - `check_calibration` / `phy_summary_plots` / `qc_plots`: first-ever exercise of the cal/monitoring plot generators — failures land as contained error blocks and feed Phase 5c (see REFACTOR_STATUS.md "Remaining");
   - pandas-3-era alias/API issues in un-exercised paths;
   - memory in `contract/build.py` on big runs (`frame_to_binned` materializes n_events×n_det arrays; if a run OOMs, chunk per key — the writer API already supports incremental writes).
   Fix on the branch, push to the fork, `git pull` on the server, rerun.
5. Once one run is green end-to-end, sweep a few runs/periods to build multi-run v2 coverage for the dashboard.

## Phase R3 — dashboard on LNGS against real data

**STATUS: Phase R3 serving, pending browser sign-off** (2026-08-20): dashboard
checkout `/data1/users/marshall/phy-dash/legend-monitor-dashboard` on
`feat/contract-v2-phy` + main merged (clean; picks up the CSV-race fix). Full
suite 61 passed / 0 skipped after two fixes committed as b27ca85: (1) contract
reader caches now key on (path, mtime_ns, **size**) — coarse fs timestamp
granularity could serve stale manifests; (2) roundtrip test tolerances moved
to float32, matching the producer's narrowed storage. Headless smoke of the
real-config phy view (p22/r012, full flag x value x style x units x cadence
sweep): v2 **96 OK / 0 FAIL** against `lmon-v2-p22`; every EMPTY combo is a
key production never produced either (no `IsBsln_<param>`, no `EventRate` in
production's 200-key v1 file — widget-vocabulary excess, pre-existing). v1
fallback smoke against the live production tree: 48 OK / 0 FAIL, Histogram
correctly reports "requires v2". Serving on legend-login2: port 9000 = v2
(`dashboard-config-lngs.yaml`, base/cal `auto/v2.0.0`, phy `lmon-v2-p22`),
port 9001 = v1 production tree for side-by-side. Remaining: browser
verification via `ssh -L 9000:localhost:9000 legend-login2` (and 9001), then
push the branch. Notes: no slow-control HDF in the v2 tree (overlay absent by
design); llama/muon/spm pages disabled (llama page crashes at build — bug
noted for upstream; muon/spm have no data).

1. On lngs-login:
   ```sh
   git clone -b feat/contract-v2-phy https://github.com/legend-exp/legend-monitor-dashboard
   cd legend-monitor-dashboard && uv venv && uv pip install -e .
   ```
2. Dev config (copy of `dashboard-config.yaml`):
   - `paths.base` / `paths.cal` → the prodenv ref version root (real run discovery: cal validity covers real periods, so the shell's period selector and the phy data line up — no scratch-app workaround needed);
   - `paths.phy` → `~/lmon-v2-out/<ref>` (the lmon output root containing `generated/`).
3. `.venv/bin/dashboard <dev-config>` (or `panel serve`) on a chosen port; from the laptop: `ssh -L 5063:localhost:<port> lngs-login` and browse. Verify: phy view uses v2 (σ bands / min-max envelopes / cadence label on the x-axis) on runs with manifests, falls back to v1 elsewhere; Histogram view works (fed by the v2 `_dist` histograms); string/sort selectors from real metadata; SC overlay if the slow-control HDF exists.
4. Sanity-compare a couple of runs against the production dashboard.

**STATUS: performance + Phase-5c pass verified** (2026-08-18): p22/r012 rerun clean against the golden snapshot — exit 0, **3 h 11 m (was 5 h 00 m)**, peak RSS **17.3 GB (was 24 GB)**, 196/200 v1 keys byte-identical and all contract keys/manifest identical. The 4 differing keys are a **pre-existing data-corruption bug now fixed**: the DataLoader path wrote uninitialised memory (denormals ~1.5e-319) for 6 detectors that lack `is_valid_bl_poly_rms_classifier`; the direct loader yields NaN. Worth raising with the collaboration — that garbage also reached the old dashboard. See REFACTOR_STATUS.md for detail.

**STATUS: baseline parameters fixed (2026-08-20)**: `get_pivot` misread
`bl_mean`/`pz_mean` as run means (one row per chunk) — root cause of the empty
"Baseline Mean" dashboard view, present in production v1 too. Fixed with an
explicit role per call site; aux/variation keys now carry attrs; `_dist`
histograms percentile-ranged; new `repair_param` regenerates a parameter for
finished runs in ~30 min; p22 r000-r013 repaired (see REFACTOR_STATUS.md).
Dashboard-side y-range/menu bugs handed to the dashboard session.

**STATUS: Phase 5c complete (2026-08-20)**: generators are data-only; every
figure draws from the contract (`plots/{qc,summary,stability,calib}`, legacy
shifter PDF names preserved verbatim); shelve/pickle deleted package-wide
(sole exception: the external dataflow shelve reader); `--write-shelves`
removed and `--plots off` now truly data-only; `plot_run` regenerates the
full figure set in seconds. Five latent legacy bugs fixed on the way (see
REFACTOR_STATUS.md). Unblocks the dashboard cal-trend views.

**STATUS: classifier pivots stripped (2026-08-20)**: the v1 file's 28 QC
classifier pivots (1.61 GB of 2.2 GB) are removed once the contract holds
their binned copies (res files keep theirs). Guarded
(`repack.strip_classifier_pivots` refuses unless every key is in the
contract), wired as the final `strip_transport` task covering the period's
*finished* runs (the live run is left alone), old runs via
`repack --strip-classifiers`. Applied to all of p22 r000–r013. **v1 2.2 ->
~0.6 GB; a run's artifact set is ~1.2 GB against the original 17 GB (14x).**
Declined for now: dropping the contract's 1min min/max sidecars; per-entry
loading (peak stays ~5.4 GB — r008–r012 measured 5.2–5.6 GB vs 14–17
before).

**STATUS: output size + memory pass (2026-08-19)**: v1 file **15.7 -> ~2.2 GB
per run**, contract file **1.06 -> ~0.58 GB** and ~3x faster to inflate, peak
RSS **2.3x lower** — verified value-for-value against the previous output
(200/200 v1 keys and all 4006 contract datasets agree to float32 epsilon).
Root causes and fixes in REFACTOR_STATUS.md; `legend-data-monitor repack`
migrates runs produced before the change. The p22 backfill picks the new
layout up from r008 onward; r000–r007 were repacked.

## Phase R4 — follow-ups unlocked by this move (not in this pass)

- Phase 5c: rewrite the cal/monitoring pickled-shelve generators using the real cal data now available; the R2 error blocks define the worklist.
- **legend-dataflow request**: dataflow's stability arrays are binned in 180 s slices, which for a single calibration run leaves almost every bin empty — measured on p22/r012, the median detector has **1 populated bin out of 87** and no detector reaches 5, so they cannot replace lmon's own FEP pass yet. Ask dataflow to bin coarsely enough for per-run use (or export per-bin counts so consumers can rebin). lmon is ready for it: `calibration.read_dataflow_stability` + `dataflow_stability_usable` read and gate the arrays, and switch on automatically once they are populated.
- Phase 5c / performance (superseded by the above): `check_calibration`'s FEP stability re-reads all cal hit events per detector but production dataflow already ships the same measurement — `bin_stability` arrays (`{time, energy, spread}`, 180 s slices, median-based) under `[det]["ecal"][<estimator>]["2614_stability"]` / `"pulser_stability"` in `generated/plt/hit/cal/<p>/<r>/*-plt_hit` shelves. Rewire `fep_gain_variation` to read those (seconds instead of minutes), keep the event-level path as fallback; caveat: the arrays share shelve values with pickled matplotlib Figures (version-fragile), so also ask dataflow for a non-pickle export of the stability arrays.
- auto-giorgio: implement `docs/auto-giorgio-integration.md` (in the lmon repo) in the auto-giorgio repo; point `MON_ROOTS` at `~/lmon-v2-out/<ref>` (ensure the monitor user has read access, or move outputs to shared scratch).
- Releases: once validated at LNGS, PR the fork branch upstream, release lmon + dashboard to PyPI, bump the Spin Dockerfile pin (NERSC deployment path per `Dockerfile` / `distribute.yml`).

## Verification summary

- R2: pytest green on server (minus the 4 pre-existing); auto_run exit 0 (or 1 with only known-unverified cal tasks failing); manifest + schema2 file readable; timings recorded.
- R3: browser via SSH tunnel shows v2 plots on real runs, v1 fallback on old runs.

## Risks / notes

- **Never write under `/data2`** — outputs go to user scratch only.
- lmon pins `pygama==2.4.1`, requires-python `>=3.10`: use uv-managed Python on the server (system python may be old).
- `check_calibration` may be slow on first run (partition data); task isolation keeps the phy chain unaffected.
- LNGS shared filesystem: the dashboard's mtime-keyed caches handle re-generated files correctly; NFS mtime granularity can delay invalidation by a second or two — harmless.
- `geds-dict.yaml` drives which parameters the pipeline computes; prefer the historical production copy if it can be recovered from the old deployment.
