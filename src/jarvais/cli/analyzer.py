import click
import json
import pandas as pd
from pathlib import Path

from jarvais.loggers import logger


@click.command(no_args_is_help=True)
@click.argument(
    "csv_path",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
)
@click.argument(
    "output_dir",
    type=click.Path(
        exists=False,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    )
)
@click.option(
    "--config",
    "-c",
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    default=None,
)
@click.option(
    "--categorical-columns",
    "-cc",
    type=str,
    multiple=True,
    help="List of categorical columns in the dataset."
)
@click.option(
    "--continuous-columns",
    "-ctc",
    type=str,
    multiple=True,
    help="List of continuous columns in the dataset."
)
@click.option(
    "--date-columns",
    "-dc",
    type=str,
    multiple=True,
    help="List of date columns in the dataset."
)
@click.option(
    "--task",
    "-t",
    type=click.Choice(["classification", "regression", "survival"], case_sensitive=False),
    help="The type of analysis task to perform."
)
@click.option(
    "--target-variable",
    "-tv",
    type=str,
    help="The target variable for the analysis."
)
@click.option(
    "--generate-report",
    "-gr",
    type=bool,
    default=True,
    help="Whether to generate a PDF report of the analysis."
)
@click.help_option(
    "-h",
    "--help",
)

def analyzer(
    csv_path: str,
    output_dir: str,
    config: str | None,
    categorical_columns: list[str],
    continuous_columns: list[str],
    date_columns: list[str],
    task: str,
    target_variable: str,
    generate_report: bool,
):
    """
    Perform automated data analysis using the JARVAIS Analyzer.

    This CLI command allows users to run exploratory data analysis, 
    missingness detection, outlier detection, one-hot encoding, 
    and visualization on a tabular dataset. Results can optionally 
    be exported as a PDF report.

    \b
    CSV_PATH: Path to the CSV file containing the dataset.
    OUTPUT_DIR: Directory to save the analysis results.

    """
    print(categorical_columns, date_columns)

    data = pd.read_csv(csv_path)
    
    from jarvais.analyzer.analyzer import Analyzer
    if config:
        logger.info(f"Loading analyzer settings from {config}")
        with Path(config).open() as f:
            settings_dict = json.load(f)

        settings_dict["output_dir"] = output_dir
        settings_dict["visualization"]["output_dir"] = output_dir

        analyzer = Analyzer.from_settings(data, settings_dict)
    else:
        categorical_columns = list(categorical_columns) if categorical_columns else None
        continuous_columns = list(continuous_columns) if continuous_columns else None
        date_columns = list(date_columns) if date_columns else None

        analyzer = Analyzer(
            data=data,
            output_dir=output_dir,
            categorical_columns=categorical_columns,
            continuous_columns=continuous_columns,
            date_columns=date_columns,
            task=task,
            target_variable=target_variable,
            generate_report=generate_report
        )

    analyzer.run()



