from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from ml.src.model_pipeline import (
    DEFAULT_MANIFEST_PATH,
    PlattCalibrator,
    build_model,
    build_preprocessor,
    feature_columns,
    load_manifest,
    predict_probabilities,
    split_dataset,
    validate_dataset,
)


DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "processed"
    / "logging_policy_dataset.csv"
)


@pytest.fixture(scope="module")
def frozen_data() -> tuple[pd.DataFrame, dict]:
    dataframe = pd.read_csv(DATASET_PATH)
    manifest = load_manifest(DEFAULT_MANIFEST_PATH)
    validate_dataset(dataframe, manifest, source_path=DATASET_PATH)
    return dataframe, manifest


@pytest.fixture(scope="module")
def fitted_components(frozen_data):
    dataframe, manifest = frozen_data
    train_features, train_target = split_dataset(dataframe, manifest)["train"]
    sample_features = train_features.iloc[:600]
    sample_target = train_target.iloc[:600]
    preprocessor = build_preprocessor(manifest)
    matrix = preprocessor.fit_transform(sample_features)
    model = build_model(seed=42)
    model.set_params(n_estimators=25)
    model.fit(matrix, sample_target)
    return preprocessor, model, sample_features


def test_frozen_temporal_split_is_preserved(frozen_data) -> None:
    dataframe, _ = frozen_data
    timestamps = pd.to_datetime(dataframe["prediction_time"])
    train = timestamps.loc[dataframe["split"].eq("train")]
    validation = timestamps.loc[dataframe["split"].eq("validation")]
    test = timestamps.loc[dataframe["split"].eq("test")]

    assert train.max() < validation.min()
    assert validation.max() < test.min()
    assert dataframe["split"].value_counts().to_dict() == {
        "train": 8663,
        "test": 1857,
        "validation": 1856,
    }


def test_target_and_policy_metadata_are_not_model_features(frozen_data) -> None:
    _, manifest = frozen_data
    columns = set(feature_columns(manifest))
    assert "recovered" not in columns
    assert "policy_probability" not in columns
    assert "policy_type" not in columns
    assert "base_policy_intervention" not in columns
    assert "fraud_flag" not in columns


def test_no_simulator_or_counterfactual_feature(frozen_data) -> None:
    _, manifest = frozen_data
    lowered = [column.lower() for column in feature_columns(manifest)]
    assert not any("simulated_recovery_probability" in value for value in lowered)
    assert not any("counterfactual" in value for value in lowered)
    assert not any(value.endswith("_outcome") for value in lowered)


def test_changed_frozen_split_membership_is_rejected(frozen_data) -> None:
    dataframe, manifest = frozen_data
    changed = dataframe.copy()
    train_index = changed.index[changed["split"].eq("train")][0]
    test_index = changed.index[changed["split"].eq("test")][0]
    changed.loc[train_index, "split"] = "test"
    changed.loc[test_index, "split"] = "train"

    with pytest.raises(ValueError, match="membership changed"):
        validate_dataset(changed, manifest)


def test_manifest_cannot_include_target_as_feature(frozen_data) -> None:
    dataframe, manifest = frozen_data
    invalid = deepcopy(manifest)
    invalid["numerical_features"].append("recovered")

    with pytest.raises(ValueError, match="Target leaked"):
        validate_dataset(dataframe, invalid)


def test_preprocessor_fits_only_train_and_ignores_unseen_category(
    frozen_data,
) -> None:
    dataframe, manifest = frozen_data
    splits = split_dataset(dataframe, manifest)
    train_features, _ = splits["train"]
    validation_features, _ = splits["validation"]
    preprocessor = build_preprocessor(manifest)
    preprocessor.fit(train_features)

    unseen = validation_features.iloc[[0]].copy()
    unseen["transaction_type"] = "UNSEEN_TRANSACTION_TYPE"
    transformed = preprocessor.transform(unseen)
    encoder = preprocessor.named_transformers_["categorical"]
    transaction_type_index = manifest["categorical_features"].index(
        "transaction_type"
    )
    assert "UNSEEN_TRANSACTION_TYPE" not in encoder.categories_[
        transaction_type_index
    ]
    assert transformed.shape[0] == 1


def test_model_outputs_deterministic_probabilities_in_unit_interval(
    fitted_components,
) -> None:
    preprocessor, model, sample_features = fitted_components
    first = predict_probabilities(preprocessor, model, sample_features)
    second = predict_probabilities(preprocessor, model, sample_features)

    np.testing.assert_array_equal(first, second)
    assert np.logical_and(first >= 0, first <= 1).all()


def test_calibrator_outputs_probabilities(fitted_components) -> None:
    preprocessor, model, sample_features = fitted_components
    raw = predict_probabilities(preprocessor, model, sample_features)
    target = np.tile([0, 1], len(raw) // 2)
    calibrator = PlattCalibrator().fit(raw, target)
    calibrated = calibrator.predict(raw)

    assert np.logical_and(calibrated >= 0, calibrated <= 1).all()


def test_saved_model_preprocessor_and_calibrator_reload(
    fitted_components,
    tmp_path,
) -> None:
    preprocessor, model, sample_features = fitted_components
    raw = predict_probabilities(preprocessor, model, sample_features)
    target = np.tile([0, 1], len(raw) // 2)
    calibrator = PlattCalibrator().fit(raw, target)
    expected = predict_probabilities(
        preprocessor,
        model,
        sample_features,
        calibrator,
    )

    model_path = tmp_path / "model.json"
    preprocessing_path = tmp_path / "preprocessing.joblib"
    calibration_path = tmp_path / "calibration.joblib"
    model.save_model(model_path)
    joblib.dump(preprocessor, preprocessing_path)
    joblib.dump(calibrator, calibration_path)

    loaded_model = XGBClassifier()
    loaded_model.load_model(model_path)
    actual = predict_probabilities(
        joblib.load(preprocessing_path),
        loaded_model,
        sample_features,
        joblib.load(calibration_path),
    )
    np.testing.assert_array_equal(expected, actual)


def test_transformed_feature_count_matches_model(fitted_components) -> None:
    preprocessor, model, _ = fitted_components
    assert len(preprocessor.get_feature_names_out()) == model.n_features_in_
