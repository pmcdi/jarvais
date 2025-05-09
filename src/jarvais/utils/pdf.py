from pathlib import Path
import pandas as pd

from fpdf.enums import Align, YPos

from jarvais.loggers import logger
from ._design import PDFRounded as FPDF
from ._elements_analyzer import _add_multiplots, _add_outlier_analysis, _add_tableone

# Reports
def generate_analysis_report_pdf(
        outlier_analysis: str,
        multiplots: list,
        categorical_columns: list,
        continuous_columns: list,
        output_dir: str | Path
    ) -> None:
    """
    Generate a PDF report for the analysis, including plots, tables, and outlier analysis.

    Args:
        outlier_analysis (str): Text summary of outlier analysis to include in the report.
        multiplots (list): A list of paths to plots to include in the multiplots section.
        categorical_columns (list): A list of categorical columns to use for multiplots.
        continuous_columns (list): A list of continuous columns to use for multiplots.
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
    
    # Title
    pdf.set_font('inter', 'B', 36)
    pdf.cell(text="jarvAIs Analyzer Report\n\n", new_y=YPos.NEXT) 

    # Add outlier analysis
    if outlier_analysis != '':
        pdf.set_y(pdf.get_y() + 10)
        pdf.set_font('inter', 'B', 24)
        pdf.cell(text="Outlier Analysis", new_y=YPos.NEXT)
        pdf = _add_outlier_analysis(pdf, outlier_analysis)

    # Add page-wide pairplots
    if (figures_dir / 'pairplot.png').exists():
        pdf.add_page()
        pdf.set_font('inter', 'B', 24)
        pdf.cell(h=pdf.t_margin, text='Pair Plot of Continuous Variables', new_y=YPos.NEXT)
        img = pdf.image((figures_dir / 'pairplot.png'), Align.C, h=pdf.eph*.5)
        pdf.set_y(pdf.t_margin + img.rendered_height + 25)

    # Add correlation plots
    if (figures_dir / 'pearson_correlation.png').exists() and (figures_dir / 'spearman_correlation.png').exists():
        pdf.set_font('inter', 'B', 24)
        pdf.cell(h=pdf.t_margin, text='Pearson and Spearman Correlation Plots', new_y=YPos.NEXT)

        corr_y = pdf.get_y() + 5

        pdf.image((figures_dir / 'pearson_correlation.png'), Align.L, corr_y, w=pdf.epw*.475)
        pdf.image((figures_dir / 'spearman_correlation.png'), Align.R, corr_y, w=pdf.epw*.475)

    # Add multiplots
    if multiplots and categorical_columns:
        pdf = _add_multiplots(pdf, multiplots, categorical_columns, continuous_columns)

    # Add demographic breakdown "table one"
    path_tableone = output_dir / 'tableone.csv'
    if path_tableone.exists():
        try:
            csv_df = pd.read_csv(path_tableone, na_filter=False).astype(str)
            pdf = _add_tableone(pdf, csv_df)
        except Exception as e:
            logger.warning(f"Unable to add table one to analysis report: {e}")

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
            and 'survival'.
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
