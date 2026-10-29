"""Load and validate ``configs/folds.yaml``."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _check_increasing(values: list[int], label: str) -> None:
    if not values:
        raise ValueError(f"{label} must be non-empty")
    if len(set(values)) != len(values):
        raise ValueError(f"{label} contains duplicates: {values}")
    if any(b <= a for a, b in zip(values, values[1:], strict=False)):
        raise ValueError(f"{label} must be strictly increasing: {values}")


class Fold(_Strict):
    name: str
    train_targets: list[int]
    test_target: int
    purpose: str | None = None
    frozen: bool | None = None

    @model_validator(mode="after")
    def _check(self) -> Fold:
        _check_increasing(self.train_targets, f"fold {self.name!r} train_targets")
        if self.test_target in self.train_targets:
            raise ValueError(
                f"fold {self.name!r}: test_target {self.test_target} overlaps train_targets"
            )
        if self.test_target <= max(self.train_targets):
            raise ValueError(
                f"fold {self.name!r}: test_target {self.test_target} must be greater than "
                f"max(train_targets) {max(self.train_targets)}"
            )
        return self


class Production(_Strict):
    name: str
    train_targets: list[int]
    forecast_target: int

    @model_validator(mode="after")
    def _check(self) -> Production:
        _check_increasing(self.train_targets, "production train_targets")
        if self.forecast_target <= max(self.train_targets):
            raise ValueError(
                f"production forecast_target {self.forecast_target} must be greater than "
                f"max(train_targets) {max(self.train_targets)}"
            )
        return self


class Extended(_Strict):
    extra_train_targets: list[int]
    flag_seasons: list[int]


class MinPA(_Strict):
    feature_season: int
    target_season: int


class SuccessCriteria(_Strict):
    beats_marcel_mae_on: list[str]
    bootstrap_ci_excludes_zero: bool
    interval_80_coverage_range: tuple[float, float]

    @model_validator(mode="after")
    def _check(self) -> SuccessCriteria:
        if not self.beats_marcel_mae_on:
            raise ValueError("beats_marcel_mae_on must be a non-empty list of fold names")
        lo, hi = self.interval_80_coverage_range
        if not 0 <= lo < hi <= 1:
            raise ValueError(
                f"interval_80_coverage_range must satisfy 0 <= lo < hi <= 1, got [{lo}, {hi}]"
            )
        return self


class FoldsConfig(_Strict):
    forecast_date_rule: str
    folds: list[Fold]
    production: Production
    extended: Extended
    min_pa: MinPA
    success_criteria: SuccessCriteria

    @model_validator(mode="after")
    def _check(self) -> FoldsConfig:
        names = [f.name for f in self.folds]
        if len(set(names)) != len(names):
            raise ValueError(f"fold names must be unique: {names}")
        targets = [f.test_target for f in self.folds]
        if any(b <= a for a, b in zip(targets, targets[1:], strict=False)):
            raise ValueError(f"folds must be in strictly increasing test_target order: {targets}")
        unknown = [n for n in self.success_criteria.beats_marcel_mae_on if n not in names]
        if unknown:
            raise ValueError(f"success_criteria.beats_marcel_mae_on has unknown folds: {unknown}")
        return self


def load_folds_config(path: str | Path) -> FoldsConfig:
    """Read a folds YAML file, validate it, and return the model."""
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return FoldsConfig.model_validate(data)
