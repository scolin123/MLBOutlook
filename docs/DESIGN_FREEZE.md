# Design freeze and the fold 3 holdout

**Status: not frozen.** Nothing in this file has been executed. Fold 3 (2026) has never been
run, `configs/folds.yaml` still carries `frozen: false` for it, and no tag exists. This is the
checklist to work through, the inventory of what the tag is supposed to pin, and the pass/fail
rules written down while the result is still unknown.

The tag and the fold 3 run are done by hand, in that order, by the repo owner.

## 1. Why this exists

Folds 1 and 2 have been looked at repeatedly: three model decisions, four interval approaches,
six feature-group ablations, and now two probability labels at three thresholds each. Every one
of those looks spends a little of the evidence in fold 2. Fold 3 is the only test season nothing
has been fitted to, chosen, or argued about, and it is worth exactly one look. The purpose of
the freeze is to make that single look mean something: after the tag there is no configuration
left to choose, so the fold 3 number cannot be the best of several.

## 2. What the tag pins

Everything below is what the holdout tests. A change to any of it after the tag invalidates the
run.

**Point forecaster** -- `ridge_rel`: `models.relative.relative_ridge`, `alpha = 10.0`, a
`LeagueRelative` wrapper around `RidgeForecaster`. Fitted on `y_woba - y_lg_woba`, re-levelled
on `predict` with `lg_woba_hat` (the prior season's league wOBA, a random walk). Pipeline:
median `SimpleImputer`, `StandardScaler`, `Ridge`. Imputer and scaler are fitted inside the
training window only.

**Features** -- `models.ridge.RELATIVE_FEATURES`, 31 columns: `woba_rel`, `xwoba_rel`,
`avg_ev_rel`, `barrel_pct_rel`, `hard_hit_pct_rel`, `pa`, `k_pct`, `bb_pct`, `miss`, `short`
at lags 1-3, plus `age_target`. No park features (deferred to v2). No empirical Bayes prior.

**Training window** -- `extended`: the fold's own `train_targets` plus
`extended.extra_train_targets` (2016-2019, 2021-2022), minus any extra at or after the fold's
test target. For fold 3 that is targets 2016-2019 and 2021-2025. 2020 is never a target and is
flagged per lag through `short_lag{k}`.

**Population** -- `min_pa.feature_season = 100` and `min_pa.target_season = 100`, from
`folds.yaml`. Features for target season T come only from seasons strictly before T, enforced by
`tests/test_no_leakage.py`.

**Intervals** -- `models.intervals.ResidualQuantileIntervals` (`resid_q_pa`): empirical residual
quantiles of the full-data fit, bucketed by `pa_lag1` at edges `(300, 500)`, `min_bucket = 50`
with a pooled fallback, `level = 0.80`. Split conformal stays in the repo as the documented
fallback with a finite-sample guarantee; it is not the selected approach and the criterion is not
evaluated against it.

**Probabilities** -- `models.probabilities.BucketedResidualDistribution` with the same factory,
edges, and `min_bucket`; baseline `weighted3_baseline` (`Weighted3`, weights 5/4/3, PA-weighted,
no mean regression, no age adjustment); threshold `0.030`, closed on both sides. Definitions as
registered in `PROJECT_PLAN.md` section 6.

**Baselines** -- `LastSeason`, `Weighted3`, and `MarcelBaseline` (weights 5/4/3,
`regression_pa = 1200`, `age_factor` 0.006/0.003 either side of 29). Marcel is the one the
criterion is measured against.

**Comparison machinery** -- `paired_bootstrap_mae_diff`, `n_boot = 1000`, `seed = 0`, 95%
percentile CI on the same test rows.

**Data snapshot** -- `data/processed/hitter_seasons.parquet` as built by
`python -m mlb_outlook.ingestion.savant --start 2015 --end 2026`: 9,804 player-seasons,
2,814 players, 2015-2026, columns `player_id, name, season, age, pa, woba, xwoba, k_pct, bb_pct,
avg_ev, barrel_pct, hard_hit_pct`. The parquet is git-ignored, so the freeze commit records its
SHA-256 and row count instead; a re-pull that changes either is a different experiment. Savant
revises historical lines, so this matters.

## 3. Open items to settle before the tag

Each needs a decision recorded in `PROJECT_PLAN.md`, not necessarily a code change. "Accept and
report" is a valid decision; leaving one undecided is not, because it turns into a temptation to
re-open the configuration after seeing fold 3.

**All six are now decided.** The decisions, with the reasoning, are in `PROJECT_PLAN.md` section
6 under "Design-freeze decisions on the six open items". None of them changed the configuration
this file pins. The items stay listed below as the questions they answer.

1. **The absolute-feature ridge.** It degrades on the wide window (fold 2 bias -0.0141) and
   blows up on synthetic two-season windows. Decide: keep it as the absolute-parameterisation
   reference, or drop it. It is not a candidate either way.
2. **The `lg_woba_lag` question.** The `league_rel` ablation showed absolute and relative rate
   lags carry the same information once the target is league-centred, which points at the
   league-level columns or the uncentred target as what breaks the absolute ridge -- unresolved.
   Record it as an open question for v2 rather than investigating on fold 2 again.
3. **Contact quality.** Adopted on +0.0008 MAE whose paired bootstrap CI touches zero
   (-0.0001 to +0.0017). The group stays, on the stated grounds that dropping it also widens the
   intervals and pushes coverage to 0.831. Confirm that reading, in writing, before the tag.
4. **League-level estimator.** `lg_woba_hat` is a random walk and fold 1 showed what that costs
   when the environment moves (bias +0.0111). A better estimator fitted on the full 2015-2026
   series would peek at fold 3's test season, so it is v2. Confirm it is not attempted now.
5. **Breakout under-confidence.** The top decile of P(breakout) observed 0.649 against 0.496
   predicted on fold 2. No calibration map is fitted (it would need its own holdout). Confirm
   that it ships uncorrected and is reported.
6. **Park factors.** Deferred to v2, with the limitation written in the README. Nothing to do
   except not quietly re-open it.

## 4. Pre-freeze checklist

- [x] All six items in section 3 decided and written into `PROJECT_PLAN.md`.
- [ ] `pytest` green and `ruff check src scripts tests` clean on the commit being tagged.
- [ ] `git status` clean -- no uncommitted or untracked source changes.
- [ ] `artifacts/fold_results.csv`, `ablation_results.csv`, `probability_results.csv` and
      `probability_reliability.csv` regenerated from that commit, and the numbers in the README
      and `PROJECT_PLAN.md` matching them.
- [ ] Data snapshot SHA-256 and row count recorded in the freeze commit message.
- [x] `scripts/run_holdout.py` written, reviewed, and tested on fold 2 before the tag -- not
      written after the flag is flipped. It must be the single entry point for the holdout, and
      it must: run fold 3 once on the extended window; report `ridge_rel` MAE and bias against
      all three baselines; run the paired bootstrap against Marcel; report `resid_q_pa` coverage,
      mean width and per-bucket coverage; report breakout and regression Brier, skill against the
      training base rate, and the reliability deciles; print the pass/fail verdict from section 6;
      write `artifacts/fold3_*.csv`. No model selection, no threshold sweep, no second window.

      Done, with precision@10 and @25 added to the probability block. Its gate refuses fold 3
      unless `fold3.frozen` is `true` **and** HEAD is tagged exactly `design-freeze-v1`, so a
      flipped flag in an uncommitted file is not enough. `--rehearse fold2` is the test-only
      path it was exercised on: it reproduces every published fold 2 number (MAE 0.0263, bias
      -0.0013, bootstrap -0.0023 with CI -0.0036 to -0.0008, coverage 0.815, per-bucket
      0.800 / 0.798 / 0.844, the two Brier/skill rows, the reliability deciles), and it refuses
      any fold carrying a `frozen` key so it cannot be used to reach the holdout.

      Two things the flip would otherwise have broken, fixed in the same commit and rehearsed
      by flipping the flag locally and running `pytest`: `scripts/run_folds.py` skipped the
      holdout on `frozen is False`, which means it would have *run* fold 3 the moment the flag
      became `true` -- it now skips any fold carrying the key at all; and four tests read the
      unfrozen holdout out of `folds.yaml`, so they would have failed or stopped testing their
      guard after the flip -- they now build an unfrozen fold of their own.
- [ ] `configs/folds.yaml`: `fold3.frozen` flipped `false` -> `true`, in the freeze commit, with
      nothing else in that file changed. This is what opens the guard in `run_fold`,
      `run_ablations` and `run_probability_fold`; `allow_unfrozen=True` stays a test-only escape
      hatch and is not used from a script.
- [ ] Tag created on that commit: `git tag -a design-freeze-v1 -m "..."`, message naming the
      selected configuration, the data snapshot hash, and the criteria. Nothing is pushed or run
      until the tag exists.

## 5. How fold 3 is run

Once. In this order.

```bash
git describe --tags --exact-match          # must print design-freeze-v1
git status --porcelain                     # must print nothing
pytest -q && ruff check src scripts tests
python -c "import hashlib,pathlib; print(hashlib.sha256(pathlib.Path('data/processed/hitter_seasons.parquet').read_bytes()).hexdigest())"
                                           # must match the hash in the freeze commit
uv run python scripts/run_holdout.py       # the single look
```

Then, without changing anything: commit the artifacts and the write-up in one commit whose
message states the verdict, including a failure. The model card and `PROJECT_PLAN.md` get the
fold 3 table next to the fold 2 one, so the two are read together.

**Rules of the run.**

- One invocation. If it crashes before producing output, the bug is fixed, the fix is committed,
  the tag is moved with a note saying why, and the run happens again -- and the write-up says
  that happened. A crash is not a look at the data; a result is.
- No re-fitting, no re-tuning, no threshold change, no "let's also try" after seeing any fold 3
  number. If a genuine bug is found *after* a result is in hand, the honest report is both
  numbers and the reason, not the better one.
- Fold 3 is not used to choose between the interval approaches, the ablated feature sets, the
  probability thresholds, or the two training windows. Those are all decided.
- No ablations, no sensitivity sweeps, and no per-player inspection on fold 3 beyond what
  `run_holdout.py` prints. Curiosity about 2026 is what the production 2027 forecast is for.

## 6. Pass / fail

The criteria are the ones already in `configs/folds.yaml` under `success_criteria`, written
before fold 2 was run and unchanged since:

| # | Criterion | Fold 2 (known) | Fold 3 (unknown) |
|---|---|---|---|
| 1 | `ridge_rel` MAE below `MarcelBaseline` MAE on the fold | 0.0263 vs 0.0285 | run once |
| 2 | Paired bootstrap of that difference, 95% CI excluding zero | -0.0023 (-0.0036 to -0.0008) | run once |
| 3 | 80% `resid_q_pa` interval coverage inside 0.75-0.85 | 0.815 | run once |

**Pass** = all three on fold 3, with criteria 1 and 2 also holding on fold 2 (they do).

**Fail** = any one of them missing. A fail is reported as a fail, in the README results table,
in `PROJECT_PLAN.md`, and in the model card, with the numbers that produced it. It is not
retried, re-tuned, or re-described as a partial success. The specific failure modes worth naming
in advance, so that nobody gets to invent an excuse afterwards:

- **Beats Marcel but the CI crosses zero.** Criterion 2 fails. Report: the model is not shown to
  beat Marcel on 2026. 362-ish test rows make this entirely possible even if the edge is real;
  that is a known property of the design, not a reason to widen the test.
- **Coverage outside 75-85%.** Criterion 3 fails. Below 0.75 means the intervals are too narrow,
  which is the in-sample-residual weakness already documented on fold 1's narrow window. Report
  it, and say that split conformal was the fallback with the finite-sample guarantee -- without
  swapping to it and re-running.
- **MAE worse than Marcel.** Criterion 1 fails. Report it, and note the most likely cause
  (`lg_woba_hat` as a random walk: 2026's environment shift is unknown at the forecast date) as
  an explanation, not a defence.

**The probabilities are not a pass/fail criterion.** Registered that way on purpose: they are
read off a distribution criteria 1-3 already test, so a fourth criterion on the same quantity
would be double-counting. Fold 3 reports breakout and regression Brier, skill against the
training base rate, and the reliability deciles beside fold 2's numbers, as evidence. A
reliability curve that is flatter or steeper than fold 2's is a finding for the write-up and for
v2, and it does not change the verdict.

## 7. What may still change after the tag, without re-running fold 3

The freeze is on the forecaster, the features, the intervals, the probability definitions, and
the data snapshot. These are untouched by it and need no new tag:

- Comps / nearest neighbours, the undervalued ranking, the Streamlit dashboard, the model card,
  the data dictionary, and the write-up.
- The 2027 production forecast: the same frozen configuration retrained on everything through
  2026, timestamped and published, scored after the 2027 season. That is the prospective test,
  and it is the only honest way to get a second out-of-sample read.
- Anything in the v2 list: park factors, pitchers, minors, contracts, the league-level estimator,
  a calibration map for the breakout probabilities.

If a change to the frozen set turns out to be necessary, it is v2 and a new freeze, and the
fold 3 number from `design-freeze-v1` stands as the v1 result.
