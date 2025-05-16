from pydantic import BaseModel

from jarvais.analyzer.modules import (
    OutlierModule, 
    VisualizationModule, 
    MissingnessModule,
    OneHotEncodingModule
)

from pydantic import BaseModel, Field
from pathlib import Path


class AnalyzerSettings(BaseModel):
    output_dir: Path = Field(
        description="Output directory.",
        title="Output Directory",
        examples=["output"],
    )
    categorical_columns: list[str] = Field(
        description="List of categorical columns.",
        title="Categorical Columns",
        examples=["gender", "treatment_type", "tumor_stage"],
    )
    continuous_columns: list[str] = Field(
        description="List of continuous columns.",
        title="Continuous Columns",
        examples=["age", "tumor_size", "survival_rate"],
    )
    date_columns: list[str] = Field(
        description="List of date columns.",
        title="Date Columns",
        examples=["date_of_treatment"],
    )
    task: str | None = Field(
        description="Task to perform.",
        title="Task",
        examples=["classification", "regression", "survival"],
    )
    target_variable: str | None = Field(
        description="Target variable.",
        title="Target Variable",
        examples=["death"],
    )
    generate_report: bool = Field(
        default=True,
        description="Whether to generate a pdf report."
    )
    settings_path: Path | None = Field(
        default=None,
        description="Path to settings file.",
    )
    settings_schema_path: Path | None = Field(
        default=None,
        description="Path to settings schema file.",
    )

    missingness: MissingnessModule
    outlier: OutlierModule
    encoding: OneHotEncodingModule
    visualization: VisualizationModule

    def model_post_init(self, context):
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_task(cls, task):
        if task not in ['classification', 'regression', 'survival', None]:
            raise ValueError("Invalid task parameter. Choose one of: 'classification', 'regression', 'survival'.")
        return task
