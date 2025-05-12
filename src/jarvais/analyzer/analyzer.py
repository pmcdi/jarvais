
import json
from pathlib import Path

import pandas as pd
import rich.repr
from tableone import TableOne # type: ignore

from jarvais.analyzer._utils import infer_types
from jarvais.analyzer.modules import (
    MissingnessModule,
    OneHotEncodingModule,
    OutlierModule,
    VisualizationModule,
)
from jarvais.analyzer.settings import AnalyzerSettings, BaseAnalyzerSettings
from jarvais.loggers import logger
from jarvais.utils.pdf import generate_analysis_report_pdf


class Analyzer():

    def __init__(
            self, 
            data: pd.DataFrame,
            output_dir: str | Path,
            categorical_columns: list[str] | None = None, 
            continuous_columns: list[str] | None = None,
            date_columns: list[str] | None = None,
            target_variable: str | None = None,
            task: str | None = None,
            generate_report: bool = True
        ) -> None:
        
        self.data = data

        # Infer all types if none provided
        if not categorical_columns and not continuous_columns and not date_columns:
            categorical_columns, continuous_columns, date_columns = infer_types(self.data)
        else:
            categorical_columns = categorical_columns or []
            continuous_columns = continuous_columns or []
            date_columns = date_columns or []

            specified_cols = set(categorical_columns + continuous_columns + date_columns)
            remaining_cols = set(self.data.columns) - specified_cols

            if not categorical_columns:
                logger.warning("Categorical columns not specified. Inferring from remaining columns.")
                categorical_columns = list(remaining_cols)

            elif not continuous_columns:
                logger.warning("Continuous columns not specified. Inferring from remaining columns.")
                continuous_columns = list(remaining_cols)

            elif not date_columns:
                logger.warning("Date columns not specified. Inferring from remaining columns.")
                date_columns = list(remaining_cols)        
        
        self.base_settings = BaseAnalyzerSettings(
            output_dir=Path(output_dir),
            categorical_columns=categorical_columns,
            continuous_columns=continuous_columns,
            date_columns=date_columns,
            target_variable=target_variable,
            task=task,
            generate_report=generate_report
        )

        self.missingness_module = MissingnessModule.build(
            categorical_columns=categorical_columns, 
            continuous_columns=continuous_columns
        )
        self.outlier_module = OutlierModule.build(
            categorical_columns=categorical_columns, 
            continuous_columns=continuous_columns
        )
        self.encoding_module = OneHotEncodingModule.build(
            categorical_columns=categorical_columns, 
            target_variable=target_variable
        )
        self.visualization_module = VisualizationModule.build(
            output_dir=output_dir,
            continuous_columns=continuous_columns,
            categorical_columns=categorical_columns,
            task=task,
            target_variable=target_variable
        )

        self.settings = AnalyzerSettings(
            base=self.base_settings,
            missingness=self.missingness_module,
            outlier=self.outlier_module,
            visualization=self.visualization_module,
            encoding=self.encoding_module
        )

    @classmethod
    def from_settings(
            cls, 
            data: pd.DataFrame, 
            settings_dict: dict
        ) -> "Analyzer":

        try:
            settings = AnalyzerSettings.model_validate(settings_dict)
        except Exception as e:
            raise ValueError("Invalid analyzer settings") from e

        analyzer = cls(
            data=data,
            output_dir=settings.base.output_dir,
        )

        analyzer.base_settings = settings.base
        analyzer.missingness_module = settings.missingness
        analyzer.outlier_module = settings.outlier
        analyzer.visualization_module = settings.visualization

        analyzer.settings = settings

        return analyzer

    def run(self) -> None:
        
        # Create Table One
        self.mytable = TableOne(
            self.data[self.base_settings.continuous_columns + self.base_settings.categorical_columns], 
            categorical=self.base_settings.categorical_columns, 
            continuous=self.base_settings.continuous_columns,
            pval=False
        )
        print(self.mytable.tabulate(tablefmt = "grid"))
        self.mytable.to_csv(self.base_settings.output_dir / 'tableone.csv')

        # Run Data Cleaning
        self.input_data = self.data.copy()
        self.data = (
            self.data
            .pipe(self.missingness_module)
            .pipe(self.outlier_module)
        )

        # Run Visualization
        figures_dir = self.base_settings.output_dir / 'figures'
        figures_dir.mkdir(exist_ok=True, parents=True)
        self.visualization_module(self.data)

        # Run Encoding
        self.data = self.encoding_module(self.data)

        # Save Data
        self.data.to_csv(self.base_settings.output_dir / 'updated_data.csv', index=False)

        # Generate Report
        if self.base_settings.generate_report:
            multiplots = (
                [f for f in (figures_dir / 'multiplots').iterdir() if f.suffix == '.png']
                if (figures_dir / 'multiplots').exists()
                else []
            )
            generate_analysis_report_pdf(
                outlier_analysis=self.outlier_module.report,
                multiplots=multiplots,
                categorical_columns=self.base_settings.categorical_columns,
                continuous_columns=self.base_settings.continuous_columns,
                output_dir=self.base_settings.output_dir
            )
        else:
            logger.warning("Skipping report generation.")

        # Save Settings
        schema_path = self.base_settings.output_dir / 'analyzer_settings.schema.json'
        with schema_path.open("w") as f:
            json.dump(self.settings.model_json_schema(), f, indent=2)

        settings_path = self.base_settings.output_dir / 'analyzer_settings.json'
        with settings_path.open('w') as f:
            json.dump({
                "$schema": str(schema_path.relative_to(self.base_settings.output_dir)),
                **self.settings.model_dump(mode="json") 
            }, f, indent=2)

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.settings


