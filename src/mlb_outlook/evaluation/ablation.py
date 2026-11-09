"""Feature-group ablations for the selected forecaster (`ridge_rel`).

One group of feature columns is removed at a time from `RELATIVE_FEATURES` and the fold is
re-run from scratch, so each row answers "what does this group buy over the rest of the
model". The point forecaster, its alpha, and the interval approach are the selected ones;
only the feature list moves.

Every ablation is scored against the full model on the same test rows with
`paired_bootstrap_mae_diff`, which is what the `verdict` column reads:

- ``helps``   -- the CI for (ablated - full) MAE sits above zero: dropping the group costs
  accuracy, so it is earning its place.
- ``hurts``   -- the CI sits below zero: dropping the group *improves* MAE.
- ``unclear`` -- the CI spans zero, the usual outcome for a small group on ~360 test rows.

A note on the league group: `RELATIVE_FEATURES` carries no ``lg_woba_lag{k}`` column at all
(the rate lags are already league-relative, so the absolute league level drops out), so
there is nothing there to delete -- see `LEAGUE_LEVEL_COLUMNS`. The league ablation that
does mean something is `league_rel`, which swaps each ``<name>_rel_lag{k}`` for its
absolute ``<name>_lag{k}`` counterpart: the model keeps its relative target and re-levelling
but loses league normalisation on the feature side.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from mlb_outlook.config import Fold, FoldsConfig
from mlb_outlook.evaluation.harness import (
    FoldNotFrozenError,
    fold_frames,
    interval_metrics,
    paired_bootstrap_mae_diff,
)
from mlb_outlook.models.intervals import ResidualQuantileIntervals
from mlb_outlook.models.relative import LeagueRelative
from mlb_outlook.models.ridge import RELATIVE_FEATURES, RidgeForecaster

LAGS = (1, 2, 3)
# Named for the docstring's sake: the absolute league level the relative model does not use.
LEAGUE_LEVEL_COLUMNS = tuple(f"lg_woba_lag{k}" for k in LAGS)
FULL = "full"
RATE_SOURCES = ("woba", "xwoba", "avg_ev", "barrel_pct", "hard_hit_pct")


def _lagged(*names: str) -> tuple[str, ...]:
    return tuple(f"{name}_lag{k}" for name in names for k in LAGS)


@dataclass(frozen=True)
class Group:
    """A feature group to remove, and anything that replaces it.

    `drop` columns must all be present in the base feature list; `add` columns are appended
    after the survivors, which is how a parameterisation swap is expressed.
    """

    name: str
    drop: tuple[str, ...] = ()
    add: tuple[str, ...] = ()
    note: str = ""


GROUPS: tuple[Group, ...] = (
    Group(
        "contact_quality",
        _lagged("xwoba_rel", "avg_ev_rel", "barrel_pct_rel", "hard_hit_pct_rel"),
        note="Statcast quality of contact, league-relative",
    ),
    Group("k_bb", _lagged("k_pct", "bb_pct"), note="strikeout and walk rates"),
    Group("age", ("age_target",), note="age in the target season"),
    Group(
        "pa_flags",
        _lagged("pa", "miss", "short"),
        note="playing time, missing-season and 2020 flags",
    ),
    Group(
        "league_rel",
        _lagged(*(f"{name}_rel" for name in RATE_SOURCES)),
        add=_lagged(*RATE_SOURCES),
        note="league normalisation on the feature side, swapped for absolute rates",
    ),
)

ABLATION_COLUMNS = [
    "group",
    "config",
    "fold",
    "n_features",
    "n_dropped",
    "mae",
    "bias",
    "mae_vs_full",
    "ci_low",
    "ci_high",
    "verdict",
    "coverage",
    "width",
]


def ablated_features(group: Group, base: Sequence[str] = RELATIVE_FEATURES) -> tuple[str, ...]:
    """The base feature list with `group.drop` removed and `group.add` appended."""
    base = tuple(base)
    unknown = [c for c in group.drop if c not in base]
    if unknown:
        raise ValueError(f"group {group.name!r} drops columns absent from the base set: {unknown}")
    kept = tuple(c for c in base if c not in group.drop)
    extra = tuple(dict.fromkeys(c for c in group.add if c not in kept))
    if not kept and not extra:
        raise ValueError(f"group {group.name!r} would leave the model with no features")
    return kept + extra


def ablated_ridge(group: Group, *, alpha: float = 10.0) -> Callable[[], LeagueRelative]:
    """A `ridge_rel` factory whose inner ridge sees the ablated feature list."""
    cols = ablated_features(group)

    def factory() -> LeagueRelative:
        return LeagueRelative(RidgeForecaster(alpha=alpha, feature_cols=cols))

    return factory


def verdict(ci_low: float, ci_high: float) -> str:
    """Read the paired-bootstrap CI for (ablated - full) MAE."""
    if ci_low > 0:
        return "helps"
    if ci_high < 0:
        return "hurts"
    return "unclear"


def run_ablations(
    df: pd.DataFrame,
    fold: Fold,
    cfg: FoldsConfig,
    *,
    extended: bool = True,
    groups: Sequence[Group] = GROUPS,
    alpha: float = 10.0,
    allow_unfrozen: bool = False,
    n_boot: int = 1000,
    seed: int = 0,
) -> pd.DataFrame:
    """Score the full model and one row per ablated group on a single fold.

    The forecaster is wrapped in `ResidualQuantileIntervals`, the selected interval
    approach, which fits its point model on the whole training frame -- so the `mae` column
    is the bare forecaster's MAE and the coverage comes from the same fit. Its PA buckets
    read ``pa_lag1`` off the frame rather than the feature list, so they survive the
    `pa_flags` ablation.

    The holdout guard from `run_fold` applies here too: a fold marked ``frozen: false``
    raises unless `allow_unfrozen` is set.
    """
    if fold.frozen is False and not allow_unfrozen:
        raise FoldNotFrozenError(
            f"fold {fold.name!r} is not frozen; pass allow_unfrozen=True after design freeze"
        )
    train, test = fold_frames(df, fold, cfg, extended=extended)
    y = test["y_woba"].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    errors: dict[str, np.ndarray] = {}
    for group in (Group(FULL, note="selected configuration"), *groups):
        cols = ablated_features(group)
        model = ResidualQuantileIntervals(factory=ablated_ridge(group, alpha=alpha))
        model.fit(train, train["y_woba"])
        err = model.predict(test).to_numpy(dtype=float) - y
        errors[group.name] = np.abs(err)
        bounds = model.predict_interval(test)
        coverage, width = interval_metrics(bounds["lo"], bounds["hi"], y)
        diff, lo, hi = paired_bootstrap_mae_diff(
            errors[group.name], errors[FULL], n_boot=n_boot, seed=seed
        )
        rows.append(
            {
                "group": group.name,
                "config": "extended" if extended else "base",
                "fold": fold.name,
                "n_features": len(cols),
                "n_dropped": len(group.drop),
                "mae": float(np.mean(errors[group.name])),
                "bias": float(np.mean(err)),
                "mae_vs_full": diff,
                "ci_low": lo,
                "ci_high": hi,
                "verdict": FULL if group.name == FULL else verdict(lo, hi),
                "coverage": coverage,
                "width": width,
            }
        )
    return pd.DataFrame(rows, columns=ABLATION_COLUMNS)
