from pathlib import Path

from pydantic import BaseModel, Field   

from jarvais.trainer.modules import (
    FeatureReductionModule, 
    SurvivalTrainerModule, 
    AutogluonTabularWrapper,
    EvaluationModule
)


class TrainerSettings(BaseModel):
    
    output_dir: Path = Field(
        description="Output directory.",
        title="Output Directory",
        examples=["output"]
    )
    target_variable: str | list[str] = Field(
        description="Target variable. Can be a list only for survival analysis.",
        title="Target Variable",
        examples=["tumor_stage", ["time", "event"]]
    )
    task: str = Field(
        description="Task to perform.",
        title="Task",
        examples=["classification", "regression", "survival"]
    )
    stratify_on: str | None = Field(
        description="Variable to stratify on.",
        title="Stratify On",
        examples=["gender"]
    )
    test_size: float = Field(
        default=0.2,
        description="Test size.",
        title="Test Size"
    )
    random_state: int = Field(
        default=42,
        description="Random state.",
        title="Random State"
    )
    explain: bool = Field(
        default=False,
        description="Whether to generate explainability reports for the model.",
        title="Generate Explainability Model"
    )

    reduction_module: FeatureReductionModule
    trainer_module: SurvivalTrainerModule | AutogluonTabularWrapper
    evaluation_module: EvaluationModule

    def model_post_init(self, context):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_task(cls, task):
        if task not in ['classification', 'regression', 'survival', None]:
            raise ValueError("Invalid task parameter. Choose one of: 'classification', 'regression', 'survival'.")
        return task

