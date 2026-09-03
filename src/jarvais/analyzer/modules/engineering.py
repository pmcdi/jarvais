from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator
from sklearn.preprocessing import PolynomialFeatures  # type: ignore

from jarvais.loggers import logger
from jarvais.analyzer.modules.base import AnalyzerModule


class InteractionSpec(BaseModel):
    name: str = Field(description="Name of the output interaction feature.")
    features: list[str] = Field(
        min_length=2,
        description="Input features used to build the interaction feature.",
    )
    operation: Literal["multiply", "add", "subtract", "concat"] = Field(
        default="multiply",
        description="Interaction operation.",
    )
    feature_types: list[Literal["numerical", "categorical"]] | None = Field(
        default=None,
        description="Optional feature type list aligned with features.",
    )

    @model_validator(mode="after")
    def validate_spec(self) -> "InteractionSpec":
        if self.operation == "subtract" and len(self.features) != 2:
            raise ValueError("subtract interaction requires exactly two features.")
        if self.feature_types is not None and len(self.feature_types) != len(self.features):
            raise ValueError("feature_types must have same length as features.")
        if self.feature_types is None:
            default_type = "categorical" if self.operation == "concat" else "numerical"
            self.feature_types = [default_type] * len(self.features)
        return self


class TransformationSpec(BaseModel):
    features: list[str] = Field(
        min_length=1,
        description="Features to transform.",
    )
    transform_type: Literal["log", "log1p", "sqrt", "square", "cube"] = Field(
        default="log1p",
        description="Mathematical transform to apply.",
    )
    prefix: str | None = Field(
        default=None,
        description="Output column prefix. Defaults to transform_type.",
    )

    @model_validator(mode="after")
    def set_default_prefix(self) -> "TransformationSpec":
        if self.prefix is None:
            self.prefix = self.transform_type
        return self


class PolynomialSpec(BaseModel):
    features: list[str] = Field(
        min_length=1,
        description="Features used for polynomial expansion.",
    )
    degree: int = Field(
        default=2,
        ge=2,
        description="Polynomial degree.",
    )
    interaction_only: bool = Field(
        default=True,
        description="Create only interaction terms when True.",
    )
    include_bias: bool = Field(
        default=False,
        description="Include polynomial bias term when True.",
    )


class BinningSpec(BaseModel):
    method: Literal["quantile", "uniform", "custom"] = Field(
        default="quantile",
        description="Binning method.",
    )
    n_bins: int = Field(
        default=5,
        ge=2,
        description="Number of bins for quantile or uniform method.",
    )
    labels: list[str] | None = Field(
        default=None,
        description="Optional bin labels.",
    )
    bins: list[float] | None = Field(
        default=None,
        description="Custom bin edges for custom method.",
    )

    @model_validator(mode="after")
    def validate_binning(self) -> "BinningSpec":
        if self.method == "custom":
            if self.bins is None or len(self.bins) < 2:
                raise ValueError("custom binning requires at least two bin edges in bins.")
            if self.labels is not None and len(self.labels) != len(self.bins) - 1:
                raise ValueError("labels length must be len(bins)-1 for custom binning.")
        elif self.method == "uniform" and self.labels is not None and len(self.labels) != self.n_bins:
            raise ValueError("labels length must equal n_bins for uniform binning.")
        return self


class RatioSpec(BaseModel):
    name: str = Field(description="Name of the output ratio feature.")
    numerator: str = Field(description="Numerator feature name.")
    denominator: str = Field(description="Denominator feature name.")
    denominator_power: float = Field(
        default=1.0,
        gt=0,
        description="Exponent applied to denominator before division.",
    )


class OncologyFeatureSpec(BaseModel):
    calculate_bmi: bool = Field(
        default=False,
        description="Create BMI feature from weight/height columns if available.",
    )
    age_stage_risk: bool = Field(
        default=False,
        description="Create age-stage risk interaction when age and stage are available.",
    )
    smoking_sex_interaction: bool = Field(
        default=False,
        description="Create smoking-sex risk feature when columns are available.",
    )
    log_biomarkers: list[str] = Field(
        default_factory=list,
        description="Biomarker columns to log-transform with log1p.",
    )

    @field_validator("log_biomarkers")
    @classmethod
    def unique_biomarkers(cls, biomarkers: list[str]) -> list[str]:
        return list(dict.fromkeys(biomarkers))


class FeatureEngineeringModule(AnalyzerModule):
    """
    Structured feature engineering module.

    Operations are defined declaratively in typed configuration objects rather
    than arbitrary code to keep feature creation serializable and reproducible.
    """

    feature_interactions: list[InteractionSpec] = Field(
        default_factory=list,
        description="Interaction feature specifications.",
        examples=[
            [
                {
                    "name": "age_x_bmi",
                    "features": ["age", "bmi"],
                    "operation": "multiply",
                    "feature_types": ["numerical", "numerical"],
                },
                {
                    "name": "sex_treatment",
                    "features": ["sex", "treatment"],
                    "operation": "concat",
                    "feature_types": ["categorical", "categorical"],
                },
            ]
        ],
    )
    transformations: list[TransformationSpec] = Field(
        default_factory=list,
        description="Transformation feature specifications.",
        examples=[
            [
                {"features": ["tumor_size"], "transform_type": "log1p", "prefix": "log"},
                {"features": ["age"], "transform_type": "square", "prefix": "sq"},
            ]
        ],
    )
    polynomial_features: PolynomialSpec | None = Field(
        default=None,
        description="Polynomial feature specification.",
        examples=[
            {
                "features": ["age", "tumor_size"],
                "degree": 2,
                "interaction_only": True,
                "include_bias": False,
            }
        ],
    )
    binning: dict[str, BinningSpec] = Field(
        default_factory=dict,
        description="Feature to binning specification mapping.",
        examples=[
            {
                "age": {"method": "quantile", "n_bins": 4, "labels": ["Q1", "Q2", "Q3", "Q4"]},
                "tumor_size": {"method": "custom", "bins": [0, 2, 5, 10], "labels": ["small", "medium", "large"]},
            }
        ],
    )
    ratio_features: list[RatioSpec] = Field(
        default_factory=list,
        description="Ratio feature specifications.",
        examples=[
            [
                {
                    "name": "neutrophil_to_lymphocyte_ratio",
                    "numerator": "neutrophil_count",
                    "denominator": "lymphocyte_count",
                    "denominator_power": 1,
                }
            ]
        ],
    )
    domain: Literal["general", "oncology"] = Field(
        default="general",
        description="Domain context for optional domain-specific features.",
        examples=["general", "oncology"],
    )
    oncology_features: OncologyFeatureSpec = Field(
        default_factory=OncologyFeatureSpec,
        description="Optional oncology-specific feature specification.",
        examples=[
            {
                "calculate_bmi": True,
                "age_stage_risk": True,
                "smoking_sex_interaction": False,
                "log_biomarkers": ["psa", "cea"],
            }
        ],
    )

    _created_features: list[str] = PrivateAttr(default_factory=list)
    _feature_mapping: dict[str, Any] = PrivateAttr(default_factory=dict)

    @classmethod
    def build(
        cls,
        feature_interactions: list[InteractionSpec] | None = None,
        transformations: list[TransformationSpec] | None = None,
        polynomial_features: PolynomialSpec | None = None,
        binning: dict[str, BinningSpec] | None = None,
        ratio_features: list[RatioSpec] | None = None,
        domain: Literal["general", "oncology"] = "general",
        oncology_features: OncologyFeatureSpec | None = None,
    ) -> "FeatureEngineeringModule":
        return cls(
            feature_interactions=feature_interactions or [],
            transformations=transformations or [],
            polynomial_features=polynomial_features,
            binning=binning or {},
            ratio_features=ratio_features or [],
            domain=domain,
            oncology_features=oncology_features or OncologyFeatureSpec(),
        )

    @property
    def created_features(self) -> list[str]:
        return list(self._created_features)

    @property
    def report(self) -> dict[str, Any]:
        return {
            "n_features_created": len(self._created_features),
            "created_features": list(self._created_features),
            "operations_applied": {
                "interactions": len(self.feature_interactions),
                "transformations": len(self.transformations),
                "polynomial": 1 if self.polynomial_features else 0,
                "binning": len(self.binning),
                "ratios": len(self.ratio_features),
                "domain_specific": 1 if self.domain == "oncology" and self._has_oncology_config() else 0,
            },
            "feature_mapping": dict(self._feature_mapping),
        }

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.enabled:
            return df

        logger.info("Applying feature engineering module...")
        df_new = df.copy()
        self._created_features = []
        self._feature_mapping = {}

        for interaction in self.feature_interactions:
            df_new = self._create_interaction(df_new, interaction)

        for transform in self.transformations:
            df_new = self._apply_transformation(df_new, transform)

        if self.polynomial_features:
            df_new = self._create_polynomial(df_new, self.polynomial_features)

        for feature, config in self.binning.items():
            df_new = self._create_bins(df_new, feature, config)

        for ratio in self.ratio_features:
            df_new = self._create_ratio(df_new, ratio)

        if self.domain == "oncology" and self._has_oncology_config():
            df_new = self._create_oncology_features(df_new, self.oncology_features)

        return df_new

    def _has_oncology_config(self) -> bool:
        return (
            self.oncology_features.calculate_bmi
            or self.oncology_features.age_stage_risk
            or self.oncology_features.smoking_sex_interaction
            or len(self.oncology_features.log_biomarkers) > 0
        )

    def _create_interaction(self, df: pd.DataFrame, interaction: InteractionSpec) -> pd.DataFrame:
        missing = [f for f in interaction.features if f not in df.columns]
        if missing:
            logger.warning(f"Skipping interaction '{interaction.name}'; missing features: {missing}")
            return df

        try:
            if all(ft == "numerical" for ft in interaction.feature_types or []):
                if interaction.operation == "multiply":
                    logger.info(f"Creating interaction '{interaction.name}' by multiplying features: {interaction.features}")
                    out = df[interaction.features[0]]
                    for feature_name in interaction.features[1:]:
                        out = out * df[feature_name]
                elif interaction.operation == "add":
                    logger.info(f"Creating interaction '{interaction.name}' by adding features: {interaction.features}")
                    out = df[interaction.features[0]]
                    for feature_name in interaction.features[1:]:
                        out = out + df[feature_name]
                elif interaction.operation == "subtract":
                    logger.info(f"Creating interaction '{interaction.name}' by subtracting features: {interaction.features}")
                    out = df[interaction.features[0]] - df[interaction.features[1]]
                else:
                    logger.warning(
                        f"Skipping interaction '{interaction.name}'; "
                        f"unknown numerical operation '{interaction.operation}'."
                    )
                    return df
                df[interaction.name] = out
            elif "categorical" in (interaction.feature_types or []) and "numerical" in (interaction.feature_types or []):
                logger.info(f"Creating interaction '{interaction.name}' by grouping categorical feature: {interaction.features}")
                cat_feature = interaction.features[(interaction.feature_types or []).index("categorical")]
                num_feature = interaction.features[(interaction.feature_types or []).index("numerical")]
                df[interaction.name] = df.groupby(cat_feature)[num_feature].transform("mean")
            else:
                logger.info(f"Creating interaction '{interaction.name}' by concatenating features: {interaction.features}")
                out = df[interaction.features[0]].astype(str)
                for feature_name in interaction.features[1:]:
                    out = out + "_" + df[feature_name].astype(str)
                df[interaction.name] = out

            self._created_features.append(interaction.name)
            self._feature_mapping[interaction.name] = interaction.model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to create interaction '{interaction.name}': {exc}")

        return df

    def _apply_transformation(self, df: pd.DataFrame, transform: TransformationSpec) -> pd.DataFrame:
        for feature_name in transform.features:
            if feature_name not in df.columns:
                logger.warning(f"Skipping transform for missing feature '{feature_name}'.")
                continue

            new_name = f"{transform.prefix}_{feature_name}"
            logger.info(f"Applying transformation '{transform.transform_type}' to feature '{feature_name}' and saving to '{new_name}'")

            try:
                if transform.transform_type == "log":
                    df[new_name] = np.log(df[feature_name].clip(lower=1e-10))
                elif transform.transform_type == "log1p":
                    df[new_name] = np.log1p(df[feature_name].clip(lower=0))
                elif transform.transform_type == "sqrt":
                    df[new_name] = np.sqrt(df[feature_name].clip(lower=0))
                elif transform.transform_type == "square":
                    df[new_name] = df[feature_name] ** 2
                elif transform.transform_type == "cube":
                    df[new_name] = df[feature_name] ** 3

                self._created_features.append(new_name)
                self._feature_mapping[new_name] = transform.model_dump(mode="json")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"Failed transforming '{feature_name}' with '{transform.transform_type}': {exc}"
                )

        return df

    def _create_polynomial(self, df: pd.DataFrame, poly_config: PolynomialSpec) -> pd.DataFrame:
        missing = [f for f in poly_config.features if f not in df.columns]
        if missing:
            logger.warning(f"Skipping polynomial features; missing features: {missing}")
            return df

        try:
            poly = PolynomialFeatures(
                degree=poly_config.degree,
                interaction_only=poly_config.interaction_only,
                include_bias=poly_config.include_bias,
            )
            x_subset = df[poly_config.features].values
            x_poly = poly.fit_transform(x_subset)
            poly_names = poly.get_feature_names_out(poly_config.features)

            for index, feature_name in enumerate(poly_names):
                if feature_name not in poly_config.features:
                    new_name = f"poly_{feature_name}"
                    logger.info(f"Creating polynomial feature '{feature_name}' and saving to '{new_name}'")
                    df[new_name] = x_poly[:, index]
                    self._created_features.append(new_name)

            self._feature_mapping["polynomial_features"] = poly_config.model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed to create polynomial features: {exc}")

        return df

    def _create_bins(self, df: pd.DataFrame, feature: str, config: BinningSpec) -> pd.DataFrame:
        if feature not in df.columns:
            logger.warning(f"Skipping binning for missing feature '{feature}'.")
            return df

        new_name = f"{feature}_binned"
        logger.info(f"Creating bins for feature '{feature}' and saving to '{new_name}'")

        try:
            if config.method == "quantile":
                binned = pd.qcut(df[feature], q=config.n_bins, labels=config.labels, duplicates="drop")
            elif config.method == "uniform":
                binned = pd.cut(df[feature], bins=config.n_bins, labels=config.labels)
            elif config.method == "custom":
                binned = pd.cut(df[feature], bins=config.bins, labels=config.labels)
            else:
                logger.warning(f"Skipping binning for '{feature}'; invalid method '{config.method}'.")
                return df

            df[new_name] = binned.astype(str)
            self._created_features.append(new_name)
            self._feature_mapping[new_name] = config.model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed binning '{feature}': {exc}")

        return df

    def _create_ratio(self, df: pd.DataFrame, ratio: RatioSpec) -> pd.DataFrame:
        if ratio.numerator not in df.columns or ratio.denominator not in df.columns:
            logger.warning(f"Skipping ratio '{ratio.name}'; numerator or denominator missing.")
            return df

        try:
            logger.info(f"Creating ratio '{ratio.name}' by dividing numerator '{ratio.numerator}' by denominator '{ratio.denominator}'")
            denom_values = df[ratio.denominator] ** ratio.denominator_power
            df[ratio.name] = df[ratio.numerator] / (denom_values + 1e-8)
            self._created_features.append(ratio.name)
            self._feature_mapping[ratio.name] = ratio.model_dump(mode="json")
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Failed creating ratio '{ratio.name}': {exc}")

        return df

    def _create_oncology_features(self, df: pd.DataFrame, oncology_config: OncologyFeatureSpec) -> pd.DataFrame:
        # BMI
        if oncology_config.calculate_bmi:
            if "weight_kg" in df.columns and "height_m" in df.columns:
                logger.info(f"Creating BMI feature from weight_kg and height_m")
                df["bmi"] = df["weight_kg"] / (df["height_m"] ** 2 + 1e-8)
                self._created_features.append("bmi")
            elif "weight" in df.columns and "height" in df.columns:
                logger.info(f"Creating BMI feature from weight and height")
                df["bmi"] = df["weight"] / (df["height"] ** 2 + 1e-8)
                self._created_features.append("bmi")

        # Age-stage risk
        if oncology_config.age_stage_risk:
            stage_col = next((col for col in df.columns if "stage" in col.lower()), None)
            if "age" in df.columns and stage_col:
                logger.info(f"Creating age-stage risk interaction from age and stage")
                stage_map = {"I": 1, "II": 2, "III": 3, "IV": 4, "1": 1, "2": 2, "3": 3, "4": 4}
                df["age_stage_risk"] = df["age"] * df[stage_col].astype(str).map(stage_map).fillna(2)
                self._created_features.append("age_stage_risk")

        # Smoking-sex interaction
        if oncology_config.smoking_sex_interaction:
            smoking_col = next((col for col in df.columns if "smok" in col.lower()), None)
            if smoking_col and "sex" in df.columns:
                logger.info(f"Creating smoking-sex interaction from smoking and sex")
                sex_map = {"M": 1, "Male": 1, "F": 0.7, "Female": 0.7}
                smoking_map = {
                    "current": 2.0,
                    "Current": 2.0,
                    "former": 1.2,
                    "Former": 1.2,
                    "never": 0.1,
                    "Never": 0.1,
                }
                sex_risk = df["sex"].astype(str).map(sex_map).fillna(1.0)
                smoking_risk = df[smoking_col].astype(str).map(smoking_map).fillna(1.0)
                df["smoking_sex_risk"] = sex_risk * smoking_risk
                self._created_features.append("smoking_sex_risk")

        # Biomarker logs
        for biomarker in oncology_config.log_biomarkers:
            if biomarker in df.columns:
                logger.info(f"Creating log biomarker '{biomarker}'")
                df[f"log_{biomarker}"] = np.log1p(df[biomarker].clip(lower=0))
                self._created_features.append(f"log_{biomarker}")

        return df


if __name__ == "__main__":
    # Example DataFrame
    example_df = pd.DataFrame({

        "age": [55, 68, 74],
        "stage": ["II", "III", "IV"],
        "sex": ["M", "F", "M"],
        "smoking_status": ["current", "never", "former"],
        "biomarker1": [2.3, 4.1, 1.1],
        "bmi": [20.0, 25.0, 30.0],
    })

    module = FeatureEngineeringModule(
        feature_interactions=[
            InteractionSpec(name="age_x_bmi", features=["age", "bmi"], operation="multiply"),
        ],
        transformations=[
            TransformationSpec(features=["bmi"], transform_type="log1p", prefix="log"),
        ],
        polynomial_features=PolynomialSpec(features=["age", "bmi"], degree=2),
        binning={
            "age": BinningSpec(method="quantile", n_bins=4, labels=["Q1", "Q2", "Q3", "Q4"]),
        },
        ratio_features=[RatioSpec(name="bmi_to_age_ratio", numerator="bmi", denominator="age", denominator_power=1)],
        domain="oncology",
        oncology_features=OncologyFeatureSpec(age_stage_risk=True, smoking_sex_interaction=True, log_biomarkers=["biomarker1"]),
    )

    from rich import print
    modified_df = module(example_df)
    print(modified_df)