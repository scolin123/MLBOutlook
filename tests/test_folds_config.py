import copy

import pytest
import yaml
from pydantic import ValidationError

from mlb_outlook.config import FoldsConfig, load_folds_config

BASE = {
    "forecast_date_rule": "end of regular season",
    "folds": [
        {"name": "f1", "train_targets": [2023], "test_target": 2024},
        {"name": "f2", "train_targets": [2023, 2024], "test_target": 2025},
        {"name": "f3", "train_targets": [2023, 2024, 2025], "test_target": 2026},
    ],
    "production": {
        "name": "prod",
        "train_targets": [2023, 2024, 2025, 2026],
        "forecast_target": 2027,
    },
    "extended": {"extra_train_targets": [2019], "flag_seasons": [2020]},
    "min_pa": {"feature_season": 100, "target_season": 100},
    "success_criteria": {
        "beats_marcel_mae_on": ["f2", "f3"],
        "bootstrap_ci_excludes_zero": True,
        "interval_80_coverage_range": [0.75, 0.85],
    },
}


def make():
    return copy.deepcopy(BASE)


def test_valid_config_passes():
    assert FoldsConfig.model_validate(make()).folds[0].name == "f1"


def test_real_config_loads():
    cfg = load_folds_config("configs/folds.yaml")
    assert [f.name for f in cfg.folds] == ["fold1", "fold2", "fold3"]
    assert cfg.production.forecast_target == 2027


def test_load_from_tmp_yaml(tmp_path):
    p = tmp_path / "folds.yaml"
    p.write_text(yaml.safe_dump(make()))
    assert load_folds_config(p).production.name == "prod"


def test_extra_key_forbidden():
    cfg = make()
    cfg["bogus"] = 1
    with pytest.raises(ValidationError):
        FoldsConfig.model_validate(cfg)


def test_duplicate_fold_names_fail():
    cfg = make()
    cfg["folds"][1]["name"] = "f1"
    cfg["success_criteria"]["beats_marcel_mae_on"] = ["f1"]
    with pytest.raises(ValidationError, match="unique"):
        FoldsConfig.model_validate(cfg)


def test_shuffled_fold_order_fails():
    cfg = make()
    cfg["folds"][0], cfg["folds"][1] = cfg["folds"][1], cfg["folds"][0]
    with pytest.raises(ValidationError, match="increasing test_target"):
        FoldsConfig.model_validate(cfg)


def test_equal_test_target_across_folds_fails():
    cfg = make()
    cfg["folds"][1].update(test_target=2024, train_targets=[2023])
    with pytest.raises(ValidationError, match="increasing test_target"):
        FoldsConfig.model_validate(cfg)


def test_earlier_test_target_across_folds_fails():
    cfg = make()
    cfg["folds"][2].update(test_target=2024, train_targets=[2023])
    with pytest.raises(ValidationError, match="increasing test_target"):
        FoldsConfig.model_validate(cfg)


def test_empty_train_targets_fails():
    cfg = make()
    cfg["folds"][0]["train_targets"] = []
    with pytest.raises(ValidationError, match="non-empty"):
        FoldsConfig.model_validate(cfg)


def test_duplicate_train_targets_fail():
    cfg = make()
    cfg["folds"][1]["train_targets"] = [2023, 2023]
    with pytest.raises(ValidationError, match="duplicates"):
        FoldsConfig.model_validate(cfg)


def test_unsorted_train_targets_fail():
    cfg = make()
    cfg["folds"][1]["train_targets"] = [2024, 2023]
    with pytest.raises(ValidationError, match="strictly increasing"):
        FoldsConfig.model_validate(cfg)


def test_train_test_overlap_fails():
    cfg = make()
    cfg["folds"][0].update(train_targets=[2023, 2024], test_target=2024)
    with pytest.raises(ValidationError, match="overlaps"):
        FoldsConfig.model_validate(cfg)


def test_test_target_not_after_max_train_fails():
    cfg = make()
    cfg["folds"][0].update(train_targets=[2023, 2025], test_target=2024)
    with pytest.raises(ValidationError, match="greater than"):
        FoldsConfig.model_validate(cfg)


def test_forecast_target_equal_to_max_train_fails():
    cfg = make()
    cfg["production"]["forecast_target"] = 2026
    with pytest.raises(ValidationError, match="forecast_target"):
        FoldsConfig.model_validate(cfg)


def test_forecast_target_earlier_than_max_train_fails():
    cfg = make()
    cfg["production"]["forecast_target"] = 2020
    with pytest.raises(ValidationError, match="forecast_target"):
        FoldsConfig.model_validate(cfg)


def test_production_train_targets_must_be_increasing():
    cfg = make()
    cfg["production"]["train_targets"] = [2024, 2023]
    with pytest.raises(ValidationError, match="strictly increasing"):
        FoldsConfig.model_validate(cfg)


def test_unknown_fold_in_success_criteria_fails():
    cfg = make()
    cfg["success_criteria"]["beats_marcel_mae_on"] = ["f2", "nope"]
    with pytest.raises(ValidationError, match="unknown folds"):
        FoldsConfig.model_validate(cfg)


def test_empty_beats_marcel_list_fails():
    cfg = make()
    cfg["success_criteria"]["beats_marcel_mae_on"] = []
    with pytest.raises(ValidationError, match="non-empty"):
        FoldsConfig.model_validate(cfg)


def test_ci_flag_must_be_bool():
    cfg = make()
    cfg["success_criteria"]["bootstrap_ci_excludes_zero"] = "maybe"
    with pytest.raises(ValidationError):
        FoldsConfig.model_validate(cfg)


@pytest.mark.parametrize("rng", [[0.9, 0.8], [0.8, 0.8], [-0.1, 0.5], [0.5, 1.1]])
def test_bad_coverage_range_fails(rng):
    cfg = make()
    cfg["success_criteria"]["interval_80_coverage_range"] = rng
    with pytest.raises(ValidationError, match="interval_80_coverage_range"):
        FoldsConfig.model_validate(cfg)


def test_coverage_range_wrong_length_fails():
    cfg = make()
    cfg["success_criteria"]["interval_80_coverage_range"] = [0.8]
    with pytest.raises(ValidationError):
        FoldsConfig.model_validate(cfg)


def test_missing_success_criteria_fails():
    cfg = make()
    del cfg["success_criteria"]
    with pytest.raises(ValidationError, match="success_criteria"):
        FoldsConfig.model_validate(cfg)
