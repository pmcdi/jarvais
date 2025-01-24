from pathlib import Path

import pandas as pd
from fpdf import FPDF
from fpdf.enums import Align

# UTILS

def _add_multiplots(pdf: FPDF, multiplots: list, categorical_columns: list) -> FPDF:
    for plot, cat in zip(multiplots, categorical_columns):
        pdf.add_page()

        pdf.set_font('inter', '', 12)
        pdf.write(5, f"{cat.title()} Multiplots\n")

        current_y = pdf.get_y()

        img_width = pdf.epw - 20
        img_height = pdf.eph - current_y - 20

        pdf.image(plot, x=10, y=current_y + 5, w=img_width, h=img_height, keep_aspect_ratio=True)

    return pdf

def _add_table(pdf: FPDF, csv_df: pd.DataFrame) -> FPDF:
    headers = csv_df.columns.tolist()
    # Keep empty header entries
    headers = ['' if 'Unnamed:' in header else header for header in headers]
    data = [headers, *csv_df.values.tolist()]

    pdf.add_page()
    pdf.set_font('inter', '', 10)
    with pdf.table() as table:
        for data_row in data:
            row = table.row()
            for datum in data_row:
                row.cell(datum)

    return pdf

# Reports

def generate_analysis_report_pdf(
        outlier_analysis: str,
        multiplots: list,
        categorical_columns: list,
        output_dir: str | Path
    ) -> None:
    """
    Generate a PDF report for the analysis, including plots, tables, and outlier analysis.

    Args:
        outlier_analysis (str): Text summary of outlier analysis to include in the report.
        multiplots (list): A list of paths to plots to include in the multiplots section.
        categorical_columns (list): A list of categorical columns to use for multiplots.
        output_dir (str | Path): The directory where the generated PDF report will be saved.

    Returns:
        None: The function saves the generated PDF to the specified output directory.
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / 'figures'

    # Instantiate PDF
    pdf = FPDF()
    pdf.add_page()
    script_dir = Path(__file__).resolve().parent

    # Adding unicode fonts
    font_path = (script_dir / 'fonts/Inter_28pt-Regular.ttf')
    pdf.add_font("inter", style="", fname=font_path)
    font_path = (script_dir / 'fonts/Inter_28pt-Bold.ttf')
    pdf.add_font("inter", style="b", fname=font_path)
    pdf.set_font('inter', '', 24)

    # Title
    pdf.write(5, "Analysis Report\n\n")

    # Add outlier analysis
    if outlier_analysis != '':
        pdf.set_font('inter', '', 12)
        pdf.write(5, "Outlier Analysis:\n")
        pdf.set_font('inter', '', 10)
        pdf.write(5, outlier_analysis)

    # Add page-wide pairplots
    pdf.image((figures_dir / 'pairplot.png'), Align.C, w=pdf.epw-20)
    pdf.add_page()

    # Add correlation plots
    pdf.image((figures_dir / 'pearson_correlation.png'), Align.C, h=pdf.eph/2)
    pdf.image((figures_dir / 'spearman_correlation.png'), Align.C, h=pdf.eph/2)

    # Add multiplots
    if multiplots and categorical_columns:
        pdf = _add_multiplots(pdf, multiplots, categorical_columns)

    # Add demographic breakdown "table one"
    path_tableone = output_dir / 'tableone.csv'
    if path_tableone.exists():
        csv_df = pd.read_csv(path_tableone, na_filter=False).astype(str)
        pdf = _add_table(pdf, csv_df)

    # Save PDF
    pdf.output(output_dir / 'analysis_report.pdf')

def generate_explainer_report_pdf(
        problem_type: str,
        output_dir: str | Path
    ) -> None:
    """
    Generate a PDF report for the explainer with visualizations and metrics.

    This function creates a PDF report that includes plots and metrics 
    relevant to the specified problem type. The report is saved in the 
    specified output directory.

    Args:
        problem_type (str): The type of machine learning problem. 
            Supported values are 'binary', 'multiclass', 'regression', 
            and 'time_to_event'.
        output_dir (str | Path): The directory where the generated PDF 
            report will be saved.

    Returns:
        None: The function saves the generated PDF to the specified output directory.
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / 'figures'

    # Instantiate PDF
    pdf = FPDF()
    pdf.add_page()
    script_dir = Path(__file__).resolve().parent

    # Adding unicode fonts
    font_path = (script_dir / 'fonts/Inter_28pt-Regular.ttf')
    pdf.add_font("inter", style="", fname=font_path)
    font_path = (script_dir / 'fonts/Inter_28pt-Bold.ttf')
    pdf.add_font("inter", style="b", fname=font_path)
    pdf.set_font('inter', '', 24)

    # Title
    pdf.write(5, "Explainer Report\n\n")

    if problem_type != 'time_to_event':
        pdf.image((figures_dir / 'test_metrics_bootstrap.png'), Align.C, h=pdf.eph//3.5, w=pdf.epw-20)
        pdf.image((figures_dir / 'validation_metrics_bootstrap.png'), Align.C, h=pdf.eph//3.5, w=pdf.epw-20)
        pdf.image((figures_dir /  'train_metrics_bootstrap.png'), Align.C, h=pdf.eph//3.5, w=pdf.epw-20)
        pdf.add_page()

    pdf.image((figures_dir / 'feature_importance.png'), Align.C, w=pdf.epw-20)
    pdf.add_page()

    if problem_type in ['binary', 'multiclass']:
        pdf.image((figures_dir / 'model_evaluation.png'), Align.C, w=pdf.epw-20)
        pdf.image((figures_dir / 'confusion_matrix.png'), Align.C, h=pdf.eph/2, w=pdf.epw-20)
        pdf.add_page()

        pdf.image((figures_dir / 'shap_barplot.png'), Align.C, h=pdf.eph/2, w=pdf.epw-20)
        pdf.image((output_dir /  'figures' / 'shap_heatmap.png'), Align.C, h=pdf.eph/2, w=pdf.epw-20)
    elif problem_type == 'regression':
        pdf.image((figures_dir / 'residual_plot.png'), Align.C, h=pdf.eph/2, w=pdf.epw-20)
        pdf.image((output_dir /  'figures' / 'true_vs_predicted.png'), Align.C, h=pdf.eph/2, w=pdf.epw-20)

    # Save PDF
    pdf.output((output_dir / 'explainer_report.pdf'))
