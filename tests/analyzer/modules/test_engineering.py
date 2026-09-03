import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from jarvais.analyzer.modules.engineering import (
    BinningSpec,
    FeatureEngineeringModule,
    InteractionSpec,
    OncologyFeatureSpec,
    RatioSpec,
    TransformationSpec,
)


def test_interaction_spec_sets_default_feature_types():
    spec = InteractionSpec(name="age_plus_bmi", features=["age", "bmi"], operation="add")
    assert spec.feature_types == ["numerical", "numerical"]


def test_interaction_spec_subtract_requires_two_features():
    with pytest.raises(ValidationError):
        InteractionSpec(name="bad_subtract", features=["a", "b", "c"], operation="subtract")


def test_binning_spec_custom_requires_bins():
    with pytest.raises(ValidationError):
        BinningSpec(method="custom")


def test_module_disabled_returns_original_dataframe():
    df = pd.DataFrame({"age": [30, 40], "bmi": [22.0, 28.0]})
    module = FeatureEngineeringModule(
        enabled=False,
        feature_interactions=[InteractionSpec(name="age_x_bmi", features=["age", "bmi"], operation="multiply")],
    )

    out = module(df)

    assert out.equals(df)
    assert "age_x_bmi" not in out.columns


def test_module_creates_interaction_transform_ratio_and_binning():
    df = pd.DataFrame(
        {
            "age": [20.0, 30.0, 40.0, 50.0],
            "bmi": [22.0, 25.0, 28.0, 31.0],
        }
    )
    module = FeatureEngineeringModule(
        feature_interactions=[
            InteractionSpec(name="age_x_bmi", features=["age", "bmi"], operation="multiply"),
        ],
        transformations=[
            TransformationSpec(features=["bmi"], transform_type="log1p", prefix="log"),
        ],
        binning={
            "age": BinningSpec(method="uniform", n_bins=2, labels=["low", "high"]),
        },
        ratio_features=[
            RatioSpec(name="bmi_to_age", numerator="bmi", denominator="age", denominator_power=1),
        ],
    )

    out = module(df)

    assert "age_x_bmi" in out.columns
    assert "log_bmi" in out.columns
    assert "age_binned" in out.columns
    assert "bmi_to_age" in out.columns

    np.testing.assert_allclose(out["age_x_bmi"].to_numpy(), (df["age"] * df["bmi"]).to_numpy())
    np.testing.assert_allclose(out["log_bmi"].to_numpy(), np.log1p(df["bmi"]).to_numpy())
    np.testing.assert_allclose(out["bmi_to_age"].to_numpy(), (df["bmi"] / (df["age"] + 1e-8)).to_numpy())
    assert set(out["age_binned"].unique()) <= {"low", "high"}


def test_module_skips_missing_columns_without_crashing():
    df = pd.DataFrame({"age": [20.0, 30.0], "bmi": [21.0, 24.0]})
    module = FeatureEngineeringModule(
        transformations=[TransformationSpec(features=["missing_col"], transform_type="sqrt", prefix="sqrt")],
        ratio_features=[RatioSpec(name="bad_ratio", numerator="bmi", denominator="missing_col")],
    )

    out = module(df)

    assert out.equals(df)
    assert module.created_features == []


def test_module_creates_oncology_features():
    df = pd.DataFrame(
        {
            "age": [55, 68],
            "stage": ["II", "IV"],
            "sex": ["M", "F"],
            "smoking_status": ["current", "never"],
            "weight_kg": [70.0, 80.0],
            "height_m": [1.75, 1.60],
            "psa": [2.0, 5.0],
        }
    )
    module = FeatureEngineeringModule(
        domain="oncology",
        oncology_features=OncologyFeatureSpec(
            calculate_bmi=True,
            age_stage_risk=True,
            smoking_sex_interaction=True,
            log_biomarkers=["psa"],
        ),
    )

    out = module(df)

    assert "bmi" in out.columns
    assert "age_stage_risk" in out.columns
    assert "smoking_sex_risk" in out.columns
    assert "log_psa" in out.columns


def test_report_tracks_created_features():
    df = pd.DataFrame({"age": [20.0, 40.0], "bmi": [22.0, 30.0]})
    module = FeatureEngineeringModule(
        feature_interactions=[InteractionSpec(name="age_plus_bmi", features=["age", "bmi"], operation="add")]
    )

    out = module(df)

    assert "age_plus_bmi" in out.columns
    assert module.report["n_features_created"] == 1
    assert "age_plus_bmi" in module.report["created_features"]
    assert module.report["operations_applied"]["interactions"] == 1
