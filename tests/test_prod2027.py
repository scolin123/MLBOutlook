"""Tests for `scripts/export_prod2027_forecasts.py` and the committed prod2027 artifacts.

Two halves. The first runs the script's pieces on synthetic seasons and a shrunk config, so
the structural guarantees -- no outcome column reaches the forecast frame, the probabilities
and the interval come off one fit, the gate refuses a changed configuration -- are tested
without touching the real snapshot. The second reads the committed
`artifacts/prod2027_forecasts.csv` and checks it against its own manifest: those rows are the
prospective test and must not drift, so this is the tripwire that says if they ever do.

The script is not an importable module, so it is loaded from its path.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mlb_outlook.config import load_folds_config
from mlb_outlook.db.versions import model_version
from mlb_outlook.models.intervals import DEFAULT_LEVEL
from mlb_outlook.models.probabilities import THRESHOLD

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "export_prod2027_forecasts.py"
ARTIFACTS = ROOT / "artifacts"


def _load_script():
    spec = importlib.util.spec_from_file_location("export_prod2027_forecasts", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


prod = _load_script()

CFG = load_folds_config(ROOT / "configs" / "folds.yaml")
# `synthetic.make_seasons` runs 2018-2026, so the real window's 2016 and 2017 would be empty
# frames. The shape of the thing under test is the same: an extended window, every train
# target with an outcome, and a forecast target one past the last of them.
SYNTH_CFG = CFG.model_copy(
    update={
        "production": CFG.production.model_copy(
            update={"train_targets": [2023, 2024, 2025], "forecast_target": 2026}
        ),
        "extended": CFG.extended.model_copy(update={"extra_train_targets": [2019, 2021, 2022]}),
    }
)


# ---- the configuration this run claims to use ----


def test_the_run_uses_the_prod_block_and_the_extended_window(seasons_df):
    train, forecast, targets = prod.production_frames(seasons_df, SYNTH_CFG)
    assert targets == [2019, 2021, 2022, 2023, 2024, 2025]
    assert set(forecast["target_season"]) == {SYNTH_CFG.production.forecast_target}
    assert set(train["target_season"]) == set(targets)
    assert len(train) > len(forecast)


def test_the_real_config_trains_through_2026_and_forecasts_2027():
    """The window the committed run used, read off `configs/folds.yaml` rather than written down."""
    extras = {s for s in CFG.extended.extra_train_targets if s < CFG.production.forecast_target}
    targets = sorted(set(CFG.production.train_targets) | extras)
    assert CFG.production.name == "prod2027"
    assert CFG.production.forecast_target == 2027
    assert max(targets) == 2026
    assert targets == [2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025, 2026]


def test_the_forecast_frame_carries_no_outcome_column(seasons_df):
    """`include_target=False` is what makes 2027 leak-proof structurally, not by promise."""
    _, forecast, _ = prod.production_frames(seasons_df, SYNTH_CFG)
    assert [c for c in forecast.columns if c.startswith("y_")] == []


def test_the_registered_settings_are_not_arguments():
    """Nothing about the frozen configuration is tunable from the command line."""
    args = prod.parse_args([])
    assert vars(args) == {"overwrite": False, "no_comps": False}
    assert prod.MODEL == "ridge_rel"
    assert prod.BASELINE == "marcel"
    assert prod.COMPS_K == 10


# ---- the gate ----


def test_the_gate_refuses_a_snapshot_that_does_not_match_the_freeze(tmp_path, monkeypatch):
    other = tmp_path / "other.parquet"
    pd.DataFrame({"season": [2026], "player_id": [1]}).to_parquet(other)
    monkeypatch.setattr(prod, "frozen_code_commit", lambda *a, **k: prod.tag_commit())
    with pytest.raises(prod.ProductionRefused, match="hashes to"):
        prod.check_frozen(other)


def test_the_gate_refuses_when_the_frozen_paths_have_moved(monkeypatch):
    monkeypatch.setattr(prod, "frozen_code_commit", lambda *a, **k: "f" * 40)
    with pytest.raises(prod.ProductionRefused, match="frozen paths were last touched"):
        prod.check_frozen()


def test_the_gate_refuses_when_there_is_no_freeze_tag(monkeypatch):
    monkeypatch.setattr(prod, "tag_commit", lambda *a, **k: None)
    with pytest.raises(prod.ProductionRefused, match="no 'design-freeze-v1' tag"):
        prod.check_frozen()


def test_the_gate_passes_on_this_repository():
    """The committed forecasts came from the tagged configuration, and it has not moved."""
    freeze_commit, digest = prod.check_frozen()
    assert freeze_commit == prod.tag_commit()
    assert digest == prod.FROZEN_DATA_SHA256


def test_existing_forecasts_are_never_overwritten_silently(tmp_path):
    path = tmp_path / "prod2027_forecasts.csv"
    path.write_text("already here\n")
    with pytest.raises(prod.ProductionRefused, match="prospective test"):
        prod.check_writable([path], overwrite=False)
    prod.check_writable([path], overwrite=True)  # explicit, and allowed
    prod.check_writable([tmp_path / "absent.csv"], overwrite=False)


# ---- the forecasts, on synthetic seasons ----


@pytest.fixture(scope="module")
def synthetic_out():
    from synthetic import make_seasons

    return prod.build(make_seasons(), SYNTH_CFG, comps=False)


def test_build_writes_the_promised_columns_in_order(synthetic_out):
    assert list(synthetic_out.columns) == prod.COLUMNS


def test_every_forecast_has_a_point_an_interval_and_a_baseline(synthetic_out):
    out = synthetic_out
    assert len(out) > 0
    assert out["ridge_rel"].notna().all()
    assert out["marcel"].notna().all()
    assert (out["lo"] <= out["ridge_rel"]).all()
    assert (out["ridge_rel"] <= out["hi"]).all()
    assert np.allclose(out["width"], out["hi"] - out["lo"])
    assert np.allclose(out["ridge_rel_minus_marcel"], out["ridge_rel"] - out["marcel"])


def test_the_three_label_probabilities_sum_to_one(synthetic_out):
    total = (
        synthetic_out["p_breakout"] + synthetic_out["p_stable"] + synthetic_out["p_regression"]
    )
    assert (total - 1).abs().max() < 1e-9
    assert set(synthetic_out["threshold"]) == {THRESHOLD}


def test_rows_are_ordered_by_the_disagreement_with_marcel(synthetic_out):
    gap = synthetic_out["ridge_rel_minus_marcel"]
    assert gap.is_monotonic_decreasing


def test_the_interval_and_the_probabilities_come_off_one_fit(seasons_df, monkeypatch):
    """The consistency check is load-bearing: break the point fit and the build must refuse."""

    class Divergent(prod.BucketedResidualDistribution):
        def predict(self, X):
            return super().predict(X) + 0.01

    monkeypatch.setattr(prod, "BucketedResidualDistribution", Divergent)
    with pytest.raises(prod.ProductionRefused, match="point fits differ"):
        prod.build(seasons_df, SYNTH_CFG, comps=False)


def test_comps_are_taken_as_of_the_last_feature_season(seasons_df):
    """Comp seasons stop before 2026 and their outcomes run through it -- the forecast's own
    information set, nothing from the season being forecast."""
    ids = seasons_df.loc[seasons_df["season"] == 2025, "player_id"].unique()[:4]
    frame = prod.comps_frame(seasons_df, ids, 2025)
    assert list(frame.columns) == [
        "player_id", "comps_n", "comps_played_next",
        "comps_median_next_woba_rel", "comps_median_delta_woba_rel",
    ]
    assert len(frame) == len(ids)
    assert (frame["comps_n"] <= prod.COMPS_K).all()
    assert (frame["comps_played_next"] <= frame["comps_n"]).all()


def test_a_player_without_comps_gets_nan_not_an_invented_neighbour():
    """One player, no history to draw a comp pool from: the row is empty, not imputed."""
    frame = prod.comps_frame(pd.DataFrame(columns=["player_id", "season"]), [1], 2025)
    assert int(frame["comps_n"].iloc[0]) == 0
    assert pd.isna(frame["comps_median_next_woba_rel"].iloc[0])


# ---- the committed artifacts ----

FORECASTS = ARTIFACTS / "prod2027_forecasts.csv"
MANIFEST = ARTIFACTS / "prod2027_manifest.json"

needs_prod = pytest.mark.skipif(
    not (FORECASTS.exists() and MANIFEST.exists()),
    reason="prod2027 artifacts not generated; see scripts/export_prod2027_forecasts.py",
)


@pytest.fixture(scope="module")
def committed():
    if not (FORECASTS.exists() and MANIFEST.exists()):
        pytest.skip("prod2027 artifacts not generated")
    return pd.read_csv(FORECASTS), json.loads(MANIFEST.read_text())


@needs_prod
def test_the_committed_forecasts_are_for_2027_only(committed):
    out, manifest = committed
    assert set(out["target_season"]) == {2027}
    assert manifest["target_season"] == 2027
    assert manifest["features_through"] == 2026
    assert manifest["run_kind"] == "production"


@needs_prod
def test_the_committed_forecasts_carry_no_outcome(committed):
    """2027 has not been played. A column claiming otherwise would make these worthless."""
    out, _ = committed
    assert list(out.columns) == prod.COLUMNS
    assert not [c for c in out.columns if c.startswith("y_") or c in ("covered", "advantage")]


@needs_prod
def test_the_committed_forecasts_match_their_manifest(committed):
    out, manifest = committed
    assert len(out) == manifest["n_forecast"]
    assert manifest["data_sha256"] == prod.FROZEN_DATA_SHA256
    assert manifest["data_rows"] == prod.FROZEN_DATA_ROWS
    assert manifest["threshold"] == THRESHOLD
    assert manifest["interval_level"] == DEFAULT_LEVEL
    assert manifest["comps_k"] == prod.COMPS_K
    assert set(out["threshold"]) == {manifest["threshold"]}


@needs_prod
def test_the_committed_forecasts_name_the_frozen_model_and_tag(committed):
    _, manifest = committed
    assert manifest["freeze_tag"] == "design-freeze-v1"
    assert manifest["freeze_commit"] == prod.tag_commit()
    assert manifest["model_version"] == model_version()
    assert manifest["model_version"].endswith(manifest["freeze_commit"][:12])
    assert manifest["generated_at"] and manifest["generated_from_commit"]


@needs_prod
def test_the_committed_rows_are_internally_consistent(committed):
    out, _ = committed
    assert out.notna().all().all()
    assert out["player_id"].is_unique
    assert (out["lo"] <= out["ridge_rel"]).all() and (out["ridge_rel"] <= out["hi"]).all()
    assert np.allclose(out["ridge_rel_minus_marcel"], out["ridge_rel"] - out["marcel"])
    total = out["p_breakout"] + out["p_stable"] + out["p_regression"]
    assert (total - 1).abs().max() < 1e-9
    assert out["pa_lag1"].min() >= CFG.min_pa.feature_season
    assert out["ridge_rel_minus_marcel"].is_monotonic_decreasing


@needs_prod
def test_the_committed_rows_cover_the_2026_regulars(committed):
    """One forecast per player with a qualifying 2026 season, and no one else."""
    out, _ = committed
    df = pd.read_parquet(ROOT / "data" / "processed" / "hitter_seasons.parquet")
    qualified = df.loc[
        (df["season"] == 2026) & (df["pa"] >= CFG.min_pa.feature_season), "player_id"
    ]
    assert set(out["player_id"]) == set(qualified)
