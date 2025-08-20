from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from pydantic import Field, PrivateAttr

from jarvais.loggers import logger
from jarvais.utils.statistical_ranking import find_top_multiplots
from jarvais.utils.plot import plot_dashboard
from .base import AnalyzerModule


class DashboardModule(AnalyzerModule):

    output_dir: str | Path = Field(
        description="Output directory.",
        title="Output Directory",
        examples=["output"],
        repr=False
    )
    continuous_columns: list[str] = Field(
        description="List of continuous columns.",
        title="Continuous Columns",
        examples=["age", "tumor_size", "survival_rate"],
        repr=False
    )
    categorical_columns: list[str] = Field(
        description="List of categorical columns.",
        title="Categorical Columns",
        examples=["gender", "treatment_type", "tumor_stage"],
        repr=False
    )
    n_top: int = Field(
        default=10,
        description="Number of top significant results to consider for the dashboard.",
        title="Top N"
    )
    significance_threshold: float = Field(
        default=0.05,
        description="P-value threshold to consider a result significant.",
        title="Significance Threshold"
    )

    _significant_results: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _dashboard_plot_path: Path | None = PrivateAttr(default=None)

    @classmethod
    def build(
        cls,
        output_dir: str | Path,
        continuous_columns: list[str],
        categorical_columns: list[str],
        n_top: int = 10,
        significance_threshold: float = 0.05,
    ) -> "DashboardModule":
        return cls(
            output_dir=output_dir,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
            n_top=n_top,
            significance_threshold=significance_threshold,
        )

    @property
    def significant_results(self) -> List[Dict[str, Any]]:
        return self._significant_results

    @property
    def dashboard_plot_path(self) -> Path | None:
        return self._dashboard_plot_path

    def __call__(self, df: pd.DataFrame, original_data: pd.DataFrame = None) -> pd.DataFrame:
        if not self.enabled:
            logger.warning("Dashboard is disabled.")
            return df

        logger.info("Computing statistical ranking for dashboard...")

        # Use original data to compute significance. Requires multiplots already generated in figures/multiplots
        # If original_data is not provided, fall back to df (for backwards compatibility)
        data_for_analysis = original_data if original_data is not None else df
        
        results = find_top_multiplots(
            data=data_for_analysis,
            categorical_columns=self.categorical_columns,
            continuous_columns=self.continuous_columns,
            output_dir=self.output_dir,
            n_top=self.n_top,
            significance_threshold=self.significance_threshold,
        )

        self._significant_results = results

        if len(results) == 0:
            logger.warning("No significant results found for dashboard plot. Skipping dashboard image generation.")
            return df

        logger.info("Generating dashboard plot of significant multiplots...")
        try:
            figures_dir = Path(self.output_dir) / "figures"
            figures_dir.mkdir(exist_ok=True, parents=True)
            self._dashboard_plot_path = plot_dashboard(results, data_for_analysis, figures_dir)
        except Exception as e:
            logger.warning(f"Failed to generate dashboard plot: {e}")

        return df