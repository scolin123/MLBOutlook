"""The single entry point for the fold 3 holdout. Run once, after the design freeze.

Written and tested before the freeze, as `docs/DESIGN_FREEZE.md` section 4 requires, so that
nothing about the report is authored with a fold 3 number already in hand. It reports what
section 4 lists, plus the reporting-only readings added before the tag:

- fold 3 on the extended window, once, with the selected configuration;
- `ridge_rel` MAE, RMSE and bias against all three baselines, unweighted and PA-weighted;
- the paired bootstrap of `ridge_rel - MarcelBaseline` MAE;
- `resid_q_pa` 80% coverage, mean width, and coverage per lag-1 PA bucket and age bracket;
- the population the scored rows are a subset of: eligible by prior-season PA, then 0 /
  1-99 / 100+ target-season PA;
- per player, `ridge_rel` minus Marcel and which of the two came closer (summary printed,
  rows written);
- breakout and regression Brier, skill against the training base rate, the reliability
  deciles, and precision at the top 10 and 25;
- the pass/fail verdict from section 6;
- `artifacts/fold3_*.csv`.

No model selection, no threshold sweep (the registered 0.030 only), no second window, no
ablation, and no per-player leaderboard printed. Those are all decided, and the point of the
freeze is that there is nothing left to choose when this runs. Everything added beyond
section 4's list is a reading of a quantity already pre-registered, and none of it feeds the
verdict, which is the three criteria on the unweighted aggregate.

**The gate.** Fold 3 is refused unless all three hold: `fold3.frozen` is `true` in
`configs/folds.yaml`, HEAD is exactly tagged `design-freeze-v1`, and the working tree is
clean. The tag is what stops the holdout from being run on a working tree that merely looks
frozen; the clean tree is what makes the tag mean something, since an uncommitted edit to a
model or a feature is not described by the tag and the run would not be reproducible from it.
`allow_unfrozen` is a test-only escape hatch and is never used from here.

Usage:
    uv run python scripts/run_holdout.py

    uv run python scripts/run_holdout.py --rehearse fold2   # test-only, see --rehearse
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_outlook.config import Fold, FoldsConfig, load_folds_config
from mlb_outlook.evaluation.calibration import PRECISION_KS, run_probability_fold
from mlb_outlook.evaluation.diagnostics import (
    coverage_by_age,
    coverage_by_pa,
    per_player_forecasts,
    population_accounting,
)
from mlb_outlook.evaluation.harness import (
    fold_frames,
    paired_bootstrap_mae_diff,
    run_fold,
)
from mlb_outlook.models.baselines import LastSeason, MarcelBaseline, Weighted3
from mlb_outlook.models.intervals import DEFAULT_LEVEL, ResidualQuantileIntervals
from mlb_outlook.models.probabilities import LABELS, THRESHOLD
from mlb_outlook.models.relative import relative_ridge

DATA = Path("data/processed/hitter_seasons.parquet")
OUT_DIR = Path("artifacts")
CONFIG = Path("configs/folds.yaml")

HOLDOUT_FOLD = "fold3"
FREEZE_TAG = "design-freeze-v1"
# The holdout runs on the extended window only: the selected configuration is `ridge_rel` on
# the extended window, and the base window is not a candidate to be chosen between.
EXTENDED = True

MODEL = "ridge_rel"
INTERVALS = "resid_q_pa"
BASELINE = "marcel"
BASELINES = ("last_season", "weighted3", "marcel")
FACTORIES = {
    "last_season": LastSeason,
    "weighted3": Weighted3,
    "marcel": MarcelBaseline,
    MODEL: relative_ridge,
    INTERVALS: ResidualQuantileIntervals,
}

# Fold 2's criteria 1 and 2, recorded from the selection runs in `docs/PROJECT_PLAN.md`
# section 6 and `docs/DESIGN_FREEZE.md` section 6, not recomputed here: a pass needs them to
# hold on fold 2 as well as on fold 3. Keeping them current is the pre-freeze checklist's job,
# which regenerates the artifacts and matches the docs against them before the tag.
FOLD2_MAE = 0.0263
FOLD2_MAE_MARCEL = 0.0285
FOLD2_BOOTSTRAP = (-0.0023, -0.0036, -0.0008)


class HoldoutRefused(RuntimeError):
    """Raised when the gate in `DESIGN_FREEZE.md` section 4 is not open."""


# ---- the gate ----


def _git(*args: str) -> str | None:
    """Stdout of a git command, or None if it fails or git is not there at all."""
    try:
        done = subprocess.run(
            ["git", *args], capture_output=True, text=True, check=False, timeout=30
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.strip()


def exact_tag() -> str | None:
    """The tag HEAD carries exactly, or None if HEAD is untagged."""
    return _git("describe", "--tags", "--exact-match")


def head_description() -> str:
    """Something human-readable to name HEAD with in a refusal message."""
    return _git("describe", "--tags", "--always", "--dirty") or "an unknown commit"


def dirty_paths() -> list[str]:
    """Modified or untracked paths, or [] if the tree is clean or git is unavailable.

    Git being unavailable cannot be mistaken for a clean tree here: `check_gate` asks for the
    exact tag first, and that question has no answer without git either.
    """
    out = _git("status", "--porcelain")
    return [line for line in (out or "").splitlines() if line.strip()]


def check_gate(fold: Fold) -> None:
    """Refuse the holdout unless the fold is frozen, HEAD is the freeze tag, and the tree is clean.

    All three, because each one alone is easy to satisfy by accident: a flipped flag in an
    uncommitted file is not a freeze, a tag on a commit whose config still says
    ``frozen: false`` is not one either, and a tagged commit with edits on top of it is not
    the thing the tag describes.
    """
    if fold.frozen is not True:
        raise HoldoutRefused(
            f"{fold.name} has frozen: {fold.frozen!r} in {CONFIG}. The holdout runs only "
            "after the freeze commit flips it to true, with nothing else in that file "
            "changed (DESIGN_FREEZE.md section 4)."
        )
    tag = exact_tag()
    if tag != FREEZE_TAG:
        found = f"tagged {tag!r}" if tag else f"at {head_description()}, untagged"
        raise HoldoutRefused(
            f"HEAD is {found}; the holdout runs only on a commit tagged exactly "
            f"{FREEZE_TAG!r}. Create the freeze tag first -- and if the tag exists but HEAD "
            "has moved past it, check the tag out rather than tagging again "
            "(DESIGN_FREEZE.md sections 4 and 5)."
        )
    dirty = dirty_paths()
    if dirty:
        listing = "\n  ".join(dirty[:20])
        more = "" if len(dirty) <= 20 else f"\n  ... and {len(dirty) - 20} more"
        raise HoldoutRefused(
            f"the working tree has {len(dirty)} uncommitted or untracked path(s), so what "
            f"would run is not what {FREEZE_TAG} pins. Section 5 requires "
            "`git status --porcelain` to print nothing before the single look. Commit, stash "
            f"or remove these first:\n  {listing}{more}"
        )


def check_rehearsal(fold: Fold) -> None:
    """Refuse ``--rehearse`` on anything that is a holdout.

    A fold carrying a ``frozen`` key at all is a holdout: that key is what `folds.yaml` uses
    to mark the fold nothing may be fitted to. The rehearsal path exists to exercise this
    script on a selection fold, and it must not become a way around `check_gate`.
    """
    if fold.name == HOLDOUT_FOLD or fold.frozen is not None:
        raise HoldoutRefused(
            f"--rehearse will not run {fold.name}: it carries a frozen flag in {CONFIG}, "
            "which marks it as a holdout. The holdout goes through the freeze gate, not "
            "through the rehearsal path."
        )


def resolve_fold(cfg: FoldsConfig, name: str) -> Fold:
    by_name = {f.name: f for f in cfg.folds}
    fold = by_name.get(name)
    if fold is None:
        raise HoldoutRefused(f"{name!r} is not in {CONFIG}: {sorted(by_name)}")
    return fold


def snapshot_fingerprint(path: Path) -> tuple[str, int]:
    """SHA-256 and row count of the data snapshot, the two numbers the freeze commit pins.

    Savant revises historical lines, so a re-pull that moves either is a different experiment
    and the fold 3 result from this run does not describe it.
    """
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return digest, len(pd.read_parquet(path, columns=["season"]))


# ---- the report ----





def verdict_table(
    table: pd.DataFrame, bootstrap: tuple[float, float, float], cfg: FoldsConfig
) -> pd.DataFrame:
    """The three pre-registered criteria from `DESIGN_FREEZE.md` section 6, scored.

    Each row carries the number that decided it, so a fail is reportable from this frame
    alone. Criterion 2 is "the CI excludes zero" exactly as written; criterion 1 is what
    carries the direction, so passing both means the model beat Marcel and the interval
    agrees it did.
    """
    point = table.set_index("model")
    mae = float(point.loc[MODEL, "mae"])
    mae_base = float(point.loc[BASELINE, "mae"])
    coverage = float(point.loc[INTERVALS, "coverage"])
    diff, lo, hi = bootstrap
    cov_lo, cov_hi = cfg.success_criteria.interval_80_coverage_range
    rows = [
        {
            "criterion": 1,
            "description": f"{MODEL} MAE below {BASELINE} MAE on the fold",
            "value": f"{mae:.4f} vs {mae_base:.4f}",
            "passed": mae < mae_base,
        },
        {
            "criterion": 2,
            "description": "paired bootstrap of that difference, 95% CI excluding zero",
            "value": f"{diff:+.4f} ({lo:+.4f} to {hi:+.4f})",
            "passed": hi < 0 or lo > 0,
        },
        {
            "criterion": 3,
            "description": (
                f"{DEFAULT_LEVEL:.0%} {INTERVALS} coverage inside {cov_lo:.2f}-{cov_hi:.2f}"
            ),
            "value": f"{coverage:.3f}",
            "passed": cov_lo <= coverage <= cov_hi,
        },
    ]
    return pd.DataFrame(rows, columns=["criterion", "description", "value", "passed"])


def run_holdout(
    df: pd.DataFrame, fold: Fold, cfg: FoldsConfig, *, out_dir: Path = OUT_DIR
) -> dict[str, pd.DataFrame]:
    """Compute the whole report for one fold, write the CSVs, and return the frames.

    The gate is the caller's job and `main` is what opens it: called directly, this runs
    whatever fold it is handed, which is what the tests do.
    """
    table = run_fold(df, fold, cfg, FACTORIES, extended=EXTENDED)

    # Refitted here for the two things the results table does not carry: the paired bootstrap
    # needs per-row errors, and the per-bucket coverage needs the bounds.
    train, test = fold_frames(df, fold, cfg, extended=EXTENDED)
    truth = test["y_woba"].to_numpy(dtype=float)
    intervals = ResidualQuantileIntervals().fit(train, train["y_woba"])
    model_err = np.abs(intervals.predict(test).to_numpy(dtype=float) - truth)
    marcel = FACTORIES[BASELINE]().fit(train, train["y_woba"])
    base_pred = marcel.predict(test)
    base_err = np.abs(base_pred.to_numpy(dtype=float) - truth)
    bootstrap = paired_bootstrap_mae_diff(model_err, base_err)
    bounds = intervals.predict_interval(test)
    # Two slices of the same bounds: the PA buckets the intervals are built on, and age
    # brackets they are not, so a bracket outside the band is a v2 finding about the width.
    coverage = pd.concat(
        [coverage_by_pa(bounds, test), coverage_by_age(bounds, test)], ignore_index=True
    )

    # The registered threshold only. A sweep here would be a choice made after the fact.
    scores, reliability = run_probability_fold(
        df, fold, cfg, extended=EXTENDED, thresholds=(THRESHOLD,)
    )

    # The real snapshot carries display names; the synthetic frames the tests use do not.
    names = (
        df.drop_duplicates("player_id").set_index("player_id")["name"]
        if "name" in df.columns
        else None
    )
    players = per_player_forecasts(
        test,
        intervals.predict(test),
        base_pred,
        names=names,
        bounds=bounds,
        model_name=MODEL,
        baseline_name=BASELINE,
    )

    frames = {
        "point": table,
        "interval_coverage": coverage,
        "population": population_accounting(df, fold, cfg),
        "players": players,
        "probabilities": scores,
        "reliability": reliability,
        "verdict": verdict_table(table, bootstrap, cfg),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(out_dir / f"{fold.name}_{name}.csv", index=False)
    print_report(fold, cfg, frames, bootstrap)
    written = ", ".join(f"{fold.name}_{name}.csv" for name in frames)
    print(f"\nwrote {out_dir}: {written}")
    return frames


# ---- printing ----


def print_report(
    fold: Fold,
    cfg: FoldsConfig,
    frames: dict[str, pd.DataFrame],
    bootstrap: tuple[float, float, float],
) -> None:
    point = frames["point"].set_index("model")
    n_train = int(point.loc[MODEL, "n_train"])
    n_test = int(point.loc[MODEL, "n_test"])
    print(
        f"\n{fold.name}: train targets {cfg.extended_train_targets(fold)} -> test "
        f"{fold.test_target}, extended window, {n_train} train rows, {n_test} test rows"
    )

    print(f"\npoint forecast ({MODEL}) against the three baselines:")
    print(
        f"  {'':12s}  {'unweighted (the criterion)':^28s}   {'PA-weighted (reporting only)':^28s}"
    )
    print(
        f"  model          MAE     RMSE      bias      MAE     RMSE      bias"
        f"     MAE vs {MODEL}"
    )
    mae = float(point.loc[MODEL, "mae"])
    for name in (*BASELINES, MODEL):
        row = point.loc[name]
        delta = "" if name == MODEL else f"{mae - float(row.mae):+.4f}"
        print(
            f"  {name:12s}  {row.mae:.4f}  {row.rmse:.4f}  {row.bias:+.4f}"
            f"   {row.mae_pa:.4f}  {row.rmse_pa:.4f}  {row.bias_pa:+.4f}   {delta:>10s}"
        )
    diff, lo, hi = bootstrap
    clear = "clear of zero" if (hi < 0 or lo > 0) else "spans zero"
    print(
        f"\npaired bootstrap, {MODEL} - {BASELINE} MAE = {diff:+.4f} "
        f"(95% CI {lo:+.4f} to {hi:+.4f}, {clear})"
    )

    cov_lo, cov_hi = cfg.success_criteria.interval_80_coverage_range
    row = point.loc[INTERVALS]
    print(
        f"\n{DEFAULT_LEVEL:.0%} intervals ({INTERVALS}): coverage {row.coverage:.3f} "
        f"(target {cov_lo:.2f}-{cov_hi:.2f}), mean width {row.width:.4f}"
    )
    print("       slice                      n   coverage   width")
    for grouping, part in frames["interval_coverage"].groupby("grouping", sort=False):
        print(f"  by {grouping}:")
        for b in part.itertuples():
            flag = "    " if cov_lo <= b.coverage <= cov_hi else " OUT"
            print(f"   {flag} {b.range:22s}  {b.n:>4}      {b.coverage:.3f}  {b.width:.4f}")

    print_population(frames["population"])
    print_players(frames["players"])
    print_probabilities(frames["probabilities"], frames["reliability"])
    print_verdict(frames["verdict"])


def print_population(accounting: pd.DataFrame) -> None:
    """What the scored rows are a subset of: everything here is conditional on playing."""
    eligible = int(accounting["eligible"].iloc[0])
    print(f"\npopulation: {eligible} players eligible by prior-season PA")
    for row in accounting.itertuples():
        print(f"  {row.bucket:28s} {row.n:>4}   {row.share:6.1%}")
    print("  every forecast, interval and probability below is conditional on the 100+ row.")


def print_players(players: pd.DataFrame) -> None:
    """Summary of the per-player file: is the edge broad or concentrated?

    Deliberately a summary and not a leaderboard. The per-player rows are in
    `<fold>_players.csv` for the write-up to quote on purpose; printing the ten biggest
    disagreements here would turn the single look into the per-player browse that
    DESIGN_FREEZE.md section 5 rules out.
    """
    diff = players[f"{MODEL}_minus_{BASELINE}"]
    advantage = players["advantage"]
    closer = float((advantage > 0).mean())
    print(f"\nper player, {MODEL} against {BASELINE} ({len(players)} rows):")
    print(
        f"  forecast gap      mean {diff.mean():+.4f}   median {diff.median():+.4f}   "
        f"sd {diff.std():.4f}   range {diff.min():+.4f} to {diff.max():+.4f}"
    )
    print(
        f"  higher than {BASELINE} on {int((diff > 0).sum())} of {len(players)} players, "
        f"lower on {int((diff < 0).sum())}"
    )
    print(
        f"  {MODEL} closer on {closer:.1%} of players; mean error advantage "
        f"{advantage.mean():+.4f} (this is the MAE difference, per player)"
    )
    print("  full rows, including the interval and whether it covered: <fold>_players.csv")


def print_probabilities(scores: pd.DataFrame, reliability: pd.DataFrame) -> None:
    print(
        f"\nbreakout / regression probabilities, registered threshold +/-{THRESHOLD:.3f} "
        "(evidence, not a criterion):"
    )
    for label in LABELS:
        r = scores.loc[scores["label"] == label].iloc[0]
        print(f"\n{label}:")
        print(
            f"  base rate   train {r.base_rate_train:.3f}   test {r.base_rate_test:.3f}   "
            f"mean predicted {r.mean_predicted:.3f}"
        )
        print(
            f"  Brier       {r.brier:.4f}   base rate {r.brier_base:.4f}   "
            f"skill {r.skill:+.4f}"
        )
        print(
            f"  model - base rate Brier {r.brier_diff:+.4f} "
            f"(95% CI {r.ci_low:+.4f} to {r.ci_high:+.4f})   "
            f"{'clear of zero' if r.ci_high < 0 or r.ci_low > 0 else 'spans zero'}"
        )
        print(f"  reliability gap (count-weighted mean |observed - predicted|) {r.ece:.4f}")
        cells = "   ".join(
            f"precision@{k} {getattr(r, f'precision_at_{k}'):.3f}" for k in PRECISION_KS
        )
        print(f"  {cells}   (chance {r.base_rate_test:.3f})")
        part = reliability.loc[reliability["label"] == label]
        print("    decile    n   predicted   observed       gap")
        for d in part.itertuples():
            print(
                f"    {d.decile:>6}  {d.n:>3}      {d.mean_predicted:.3f}      "
                f"{d.observed:.3f}   {d.gap:+.3f}"
            )


def print_verdict(verdict: pd.DataFrame) -> None:
    """The section 6 verdict: three criteria on this fold, plus fold 2's recorded two."""
    print("\n" + "=" * 78)
    print("PASS / FAIL (pre-registered in configs/folds.yaml, DESIGN_FREEZE.md section 6)")
    print("=" * 78)
    for row in verdict.itertuples():
        print(f"  {'PASS' if row.passed else 'FAIL'}  {row.criterion}. {row.description}")
        print(f"        {row.value}")
    diff, lo, hi = FOLD2_BOOTSTRAP
    print(
        f"\n  fold 2, recorded: MAE {FOLD2_MAE:.4f} against {FOLD2_MAE_MARCEL:.4f} for "
        f"{BASELINE}, bootstrap {diff:+.4f} ({lo:+.4f} to {hi:+.4f}) -- criteria 1 and 2 "
        "hold there, which a pass also requires."
    )
    passed = bool(verdict["passed"].all())
    print(f"\n  VERDICT: {'PASS' if passed else 'FAIL'}")
    if passed:
        print("  All three criteria met on this fold. Report the numbers as they stand.")
    else:
        missed = ", ".join(str(c) for c in verdict.loc[~verdict["passed"], "criterion"])
        print(
            f"  Criterion {missed} missed. This is reported as a fail -- in the README "
            "results table, in PROJECT_PLAN.md and in the model card, with the numbers "
            "above. It is not retried, re-tuned, or re-described as a partial success "
            "(DESIGN_FREEZE.md section 6)."
        )


# ---- entry point ----


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the fold 3 holdout once, after the design freeze."
    )
    parser.add_argument(
        "--rehearse",
        metavar="FOLD",
        default=None,
        help=(
            "TEST-ONLY: run the whole report on a selection fold (e.g. fold2) to exercise "
            "this script before the freeze. Refused for any fold carrying a frozen flag, so "
            "it cannot be used to reach the holdout, and it writes <fold>_*.csv rather than "
            "fold3_*.csv."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_folds_config(CONFIG)
    rehearsal = args.rehearse is not None
    try:
        fold = resolve_fold(cfg, args.rehearse if rehearsal else HOLDOUT_FOLD)
        if rehearsal:
            check_rehearsal(fold)
        else:
            check_gate(fold)
    except HoldoutRefused as refused:
        print(f"refused: {refused}")
        return 1

    if rehearsal:
        print(
            f"REHEARSAL on {fold.name}. This is the test path: it is not the holdout, the "
            "numbers below come from a fold already used for selection, and the verdict is a "
            "dry run of the print-out rather than a result."
        )
    else:
        print(
            f"HOLDOUT RUN. {fold.name}, once, at tag {FREEZE_TAG}, on a clean tree."
        )

    digest, rows = snapshot_fingerprint(DATA)
    print(f"\ndata snapshot {DATA}: sha256 {digest}, {rows} rows")
    print("  this must match the hash and row count in the freeze commit message.")

    df = pd.read_parquet(DATA)
    frames = run_holdout(df, fold, cfg, out_dir=OUT_DIR)
    return 0 if bool(frames["verdict"]["passed"].all()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
