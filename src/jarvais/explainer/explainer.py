from __future__ import annotations

from pathlib import Path
import json
import rich.repr
from typing import TYPE_CHECKING

from jarvais.explainer.modules import ImportanceModule, VisualizationModule, BiasAuditModule
from jarvais.explainer.settings import ExplainerSettings
from jarvais.utils.pdf import generate_explainer_report_pdf

if TYPE_CHECKING:
    from jarvais.trainer import TrainerSupervised


class Explainer():

    def __init__(
        self,
        output_dir: str | Path,
        sensitive_features: list | None = None,
    ) -> None:
    
        output_dir = Path(output_dir)
        figures_dir = output_dir / 'figures'
        bias_dir = output_dir / 'bias'

        self.visualization_module = VisualizationModule(output_dir=figures_dir)
        self.importance_module = ImportanceModule(output_dir=figures_dir)
        self.bias_audit_module = BiasAuditModule(output_dir=bias_dir, sensitive_features=sensitive_features)

        self.settings = ExplainerSettings(
            output_dir=output_dir,
            visualization=self.visualization_module,
            importance=self.importance_module,
            bias_audit=self.bias_audit_module
        )

    @classmethod
    def from_settings(
            cls, 
            settings_dict: dict
        ) -> "Explainer":
        """
        Initialize an Explainer instance with a given settings dictionary. Settings are validated by pydantic.

        Args:
            settings_dict (dict): A dictionary containing the explainer settings.

        Returns:
            Explainer: An explainer instance with the given settings.
        """
        try:
            settings = ExplainerSettings.model_validate(settings_dict)
        except Exception as e:
            raise ValueError("Invalid explainer settings") from e

        explainer = cls(
            output_dir=settings.output_dir,
        )

        explainer.visualization_module = settings.visualization
        explainer.importance_module = settings.importance
        explainer.bias_audit_module = settings.bias_audit

        explainer.settings = settings

        return explainer

    def run(self, trainer: "TrainerSupervised") -> None:
        """Generate diagnostic plots and reports for the trained model."""

        # Run Modules
        self.bias_audit_module(trainer)
        self.visualization_module(trainer)
        self.importance_module(trainer)

        # Generate Report
        generate_explainer_report_pdf(trainer.settings.task, self.settings.output_dir)

        # Save Settings
        self.settings.settings_schema_path = self.settings.output_dir / 'explainer_settings.schema.json'
        with self.settings.settings_schema_path.open("w") as f:
            json.dump(self.settings.model_json_schema(), f, indent=2)

        self.settings.settings_path = self.settings.output_dir / 'explainer_settings.json'
        with self.settings.settings_path.open('w') as f:
            json.dump({
                "$schema": str(self.settings.settings_schema_path.relative_to(self.settings.output_dir)),
                **self.settings.model_dump(mode="json") 
            }, f, indent=2)

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.settings

    def __repr__(self) -> str:
        return f"Analyzer(settings={self.settings.model_dump_json(indent=2)})"

if __name__ == "__main__":
    from jarvais.trainer import TrainerSupervised

    trainer = TrainerSupervised.load_trainer("temp_output/trainer_test_rad")
    explainer = Explainer(output_dir="temp_output/explainer_test_rad")
    explainer.run(trainer)