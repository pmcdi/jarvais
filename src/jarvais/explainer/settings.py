from pydantic import BaseModel, Field
from pathlib import Path
from typing import Any

from jarvais.explainer.modules import (
    VisualizationModule, 
    ImportanceModule, 
    BiasAuditModule
)

class ExplainerSettings(BaseModel):
    output_dir: Path = Field(
        description="Output directory.",
        title="Output Directory",
    )
    settings_path: Path | None = Field(
        default=None,
        description="Path to settings file.",
    )
    settings_schema_path: Path | None = Field(
        default=None,
        description="Path to settings schema file.",
    )
    
    visualization: VisualizationModule
    importance: ImportanceModule
    bias_audit: BiasAuditModule

    def model_post_init(self, context: Any) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)