import pandas as pd
from pathlib import Path

from jarvais.analyzer.modules import VisualizationModule

def test_visualization_module_init(tmp_path):

    vis_module = VisualizationModule.build(
        output_dir=tmp_path,
        continuous_columns=["age"],
        categorical_columns=["sex"],
        target_variable="target",
        task="classification"
    )

    assert "kaplan_meier" not in vis_module.plots
