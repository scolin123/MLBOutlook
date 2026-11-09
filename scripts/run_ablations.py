"""Feature-group ablations for `ridge_rel` on the selection fold only.

Removes one feature group at a time from the selected configuration (`ridge_rel`, extended
training window) and reports MAE, the paired bootstrap CI against the full model, and the
80% interval coverage of `resid_q_pa` wrapped around the ablated forecaster.

Fold 2 is the only fold run. Fold 3 is the final holdout: it is looked up by name and never
touched, and `run_ablations` keeps the same `frozen: false` guard as `run_fold` anyway.

Usage:
    uv run python scripts/run_ablations.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from mlb_outlook.config import FoldsConfig, load_folds_config
from mlb_outlook.evaluation.ablation import FULL, GROUPS, run_ablations

DATA = Path("data/processed/hitter_seasons.parquet")
OUT = Path("artifacts/ablation_results.csv")
FOLD = "fold2"
EXTENDED = True


def selection_fold(cfg: FoldsConfig):
    """The fold ablations are allowed to run on; anything unfrozen is refused."""
    by_name = {f.name: f for f in cfg.folds}
    fold = by_name.get(FOLD)
    if fold is None:
        raise SystemExit(f"{FOLD!r} is not in configs/folds.yaml: {sorted(by_name)}")
    if fold.frozen is False:
        raise SystemExit(f"{FOLD!r} is marked frozen: false; ablations do not run on a holdout")
    return fold


def main() -> None:
    df = pd.read_parquet(DATA)
    cfg = load_folds_config("configs/folds.yaml")
    fold = selection_fold(cfg)
    table = run_ablations(df, fold, cfg, extended=EXTENDED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT, index=False)
    print(table.drop(columns=["config", "fold"]).round(4).to_string(index=False))
    print_report(table, cfg)
    print(f"\nwrote {OUT}")


def print_report(table: pd.DataFrame, cfg: FoldsConfig) -> None:
    """Per-group lines, then whatever needs flagging for a human."""
    notes = {g.name: g.note for g in GROUPS}
    full = table.loc[table["group"] == FULL].iloc[0]
    lo, hi = cfg.success_criteria.interval_80_coverage_range
    print(f"\nfull model: MAE {full.mae:.4f}, {full.n_features} features, "
          f"80% coverage {full.coverage:.3f}")
    print(f"\ndropping one group at a time ({full.fold}, {full.config} window):")
    for row in table.loc[table["group"] != FULL].itertuples():
        print(
            f"  {row.group:16s} -{row.n_dropped:2d} cols  MAE {row.mae:.4f}  "
            f"vs full {row.mae_vs_full:+.4f} (95% CI {row.ci_low:+.4f} to {row.ci_high:+.4f})  "
            f"{row.verdict:7s}  coverage {row.coverage:.3f}  width {row.width:.4f}"
            f"   [{notes.get(row.group, '')}]"
        )

    flagged = table.loc[table["verdict"] == "hurts"]
    print()
    if flagged.empty:
        print("No group clearly hurts: every CI for (ablated - full) MAE touches zero or "
              "sits above it. Selected configuration unchanged.")
    else:
        print("FLAG: these groups improve MAE when dropped, CI clear of zero. "
              "Not acted on -- the selected configuration is unchanged:")
        for row in flagged.itertuples():
            print(
                f"  {row.group}: {row.mae_vs_full:+.4f} "
                f"(CI {row.ci_low:+.4f} to {row.ci_high:+.4f})"
            )

    out_of_range = table.loc[(table["coverage"] < lo) | (table["coverage"] > hi)]
    if not out_of_range.empty:
        print(f"\n80% coverage outside the pre-registered {lo:.0%}-{hi:.0%} range:")
        for row in out_of_range.itertuples():
            print(f"  {row.group}: {row.coverage:.3f}")


if __name__ == "__main__":
    main()
