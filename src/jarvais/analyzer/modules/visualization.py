
import pandas as pd
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field, PrivateAttr
from joblib import Parallel, delayed

from jarvais.utils.plot import (
    plot_corr,
    plot_frequency_table,
    plot_kaplan_meier_by_category,
    plot_one_multiplot,
    plot_pairplot,
    plot_umap
)
from jarvais.loggers import logger


class VisualizationModule(BaseModel):

    plots: list[str] = Field(
        description="List of plots to generate.",
        title="Plots",
        examples=["corr", "pairplot", "frequency_table", "multiplot", "umap"]
    )
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
    task: str | None = Field(
        description="Task to perform.",
        title="Task",
        examples=["classification", "regression", "survival"],
        repr=False
    )
    target_variable: str | None = Field(
        description="Target variable.",
        title="Target Variable",
        examples=["death"],
        repr=False
    )
    enabled: bool = Field(
        default=True,
        description="Whether to perform visualization."
    )

    _figures_dir: str | Path | None = PrivateAttr(default=None)
    _multiplots: list[str] = PrivateAttr(default=list)
    _umap_data: np.ndarray | None = PrivateAttr(default=None)

    def model_post_init(self, context):
        
        self._figures_dir = Path(self.output_dir) / "figures"
        self._figures_dir.mkdir(exist_ok=True)

        plot_order = ["corr", "pairplot", "umap", "frequency_table", "multiplot", "kaplan_meier"]
        self.plots = [p for p in plot_order if p in self.plots] # Need UMAP before frequency table

    @classmethod
    def validate_plots(cls, plots):
        plot_registry = ["corr", "pairplot", "frequency_table", "multiplot", "umap", "kaplan_meier"]
        invalid = [p for p in plots if p not in plot_registry]
        if invalid:
            raise ValueError(f"Invalid plots: {invalid}. Available: {plot_registry}")
        return plots
    
    @classmethod
    def build(
            cls,
            output_dir: str | Path,
            continuous_columns: list[str],
            categorical_columns: list[str],
            task: str | None,
            target_variable: str | None
        ) -> "VisualizationModule":
        plots = ["corr", "pairplot", "frequency_table", "multiplot", "umap"]

        if task == "survival":
            plots.append("kaplan_meier")

        return cls(plots=plots, 
                   output_dir=output_dir,
                   continuous_columns=continuous_columns,
                   categorical_columns=categorical_columns,
                   task=task,
                   target_variable=target_variable
                )   

    def __call__(self, data: pd.DataFrame) -> None:
        if not self.enabled:
            logger.warning("Visualization is disabled.")
            return

        for plot in self.plots:
            try:
                match plot:
                    case "corr":
                        logger.info("Plotting Correlation Matrix...")
                        self._plot_correlation(data)
                    case "pairplot":
                        logger.info("Plotting Pairplot...")
                        self._plot_pairplot(data)
                    case "frequency_table":
                        logger.info("Plotting Frequency Table...")
                        plot_frequency_table(data, self.categorical_columns, self._figures_dir)
                    case "umap":
                        logger.info("Plotting UMAP...")
                        self._umap_data = plot_umap(data, self.continuous_columns, self._figures_dir)
                    case "kaplan_meier":
                        logger.info("Plotting Kaplan Meier Curves...")
                        self._plot_kaplan_meier(data)
            except Exception as e:
                logger.info(f"Skipping {plot} due to error: {e}")
        
        if 'multiplot' in self.plots:
            if self._umap_data is None:
                raise ValueError("Cannot plot multiplot without UMAP data.")
            
            logger.info("Plotting Multiplot...")
            self._plot_multiplot(data)

    def _plot_correlation(self, data: pd.DataFrame):
        p_corr = data[self.continuous_columns].corr(method="pearson")
        s_corr = data[self.continuous_columns].corr(method="spearman")
        size = 1 + len(self.continuous_columns)*1.2
        plot_corr(p_corr, size, file_name='pearson_correlation.png', output_dir=self._figures_dir, title="Pearson Correlation")
        plot_corr(s_corr, size, file_name='spearman_correlation.png', output_dir=self._figures_dir, title="Spearman Correlation")

    def _plot_pairplot(self, data: pd.DataFrame):
        if self.target_variable in self.categorical_columns:
            plot_pairplot(data, self.continuous_columns, output_dir=self._figures_dir, target_variable=self.target_variable)
        else:
            plot_pairplot(data, self.continuous_columns, output_dir=self._figures_dir)

    def _plot_multiplot(self, data: pd.DataFrame):
        (self._figures_dir / 'multiplots').mkdir(parents=True, exist_ok=True)
        self._multiplots = Parallel(n_jobs=-1)(
            delayed(plot_one_multiplot)(
                data,
                self._umap_data,
                var,
                self.continuous_columns,
                self._figures_dir
            ) for var in self.categorical_columns
        )

    def _plot_kaplan_meier(self, data: pd.DataFrame):
        data_x = data.drop(columns=['time', 'event'])
        data_y = data[['time', 'event']]
        categorical_columns = [cat for cat in self.categorical_columns if cat != 'event']
        plot_kaplan_meier_by_category(data_x, data_y, categorical_columns, self._figures_dir / 'kaplan_meier')
