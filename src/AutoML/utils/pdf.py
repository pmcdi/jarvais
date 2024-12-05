import os
from pathlib import Path
import pandas as pd
from fpdf import FPDF
from fpdf.enums import Align

# UTILS

def add_outlier_analysis(pdf, outlier_analysis):
    if outlier_analysis != '':
        pdf.set_font('dejavu-sans', '', 12)  
        pdf.write(5, f"Outlier Analysis:\n")
        pdf.set_font('dejavu-sans', '', 10) 
        pdf.write(5, outlier_analysis)
    
    return pdf

def add_multiplots(pdf, multiplots, categorical_columns):
    for plot, cat in zip(multiplots, categorical_columns):
        pdf.add_page()
        
        pdf.set_font('dejavu-sans', '', 12)
        pdf.write(5, f"{cat.title()} Multiplots\n")
        
        current_y = pdf.get_y()
        
        img_width = pdf.epw - 20  
        img_height = pdf.eph - current_y - 20
        
        pdf.image(plot, x=10, y=current_y + 5, w=img_width, h=img_height, keep_aspect_ratio=True)
        
    return pdf

def add_table(pdf, csv_df):
    headers = csv_df.columns.tolist()
    headers = [f'' if 'Unnamed:' in header else header for header in headers] # Keep empty header entries
    data = [headers] + csv_df.values.tolist()

    pdf.add_page()
    pdf.set_font('dejavu-sans', '', 10)  
    with pdf.table() as table:
        for data_row in data:
            row = table.row()
            for datum in data_row:
                row.cell(datum)

    return pdf

# Reports

def generate_analysis_report_pdf(
        outlier_analysis=None,
        multiplots=None,
        categorical_columns=None,
        output_dir: str = "./"):
    """
    Generate a PDF report of the analysis with plots and tables.

    This method creates a PDF report containing various analysis results, 
    including outlier analysis, correlation plots, multiplots, and a summary table.

    The report is structured as follows:
        - Cover page with the report title.
        - Outlier analysis text.
        - Pair plot, Pearson correlation, and Spearman correlation plots.
        - Multiplots for categorical vs. continuous variable relationships.
        - A tabular summary of the analysis results.

    The PDF uses custom fonts and is saved in the specified output directory.
    """

    # Instantiate PDF
    pdf = FPDF()
    pdf.add_page()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Adding unicode fonts
    font_path = os.path.join(script_dir, 'fonts/DejaVuSans.ttf')
    pdf.add_font("dejavu-sans", style="", fname=font_path)
    font_path = os.path.join(script_dir, 'fonts/DejaVuSans-Bold.ttf')
    pdf.add_font("dejavu-sans", style="b", fname=font_path)
    pdf.set_font('dejavu-sans', '', 24)  

    # Title
    pdf.write(5, "Analysis Report\n\n")

    # Add outlier analysis
    if outlier_analysis:
        pdf = add_outlier_analysis(pdf, outlier_analysis)
    
    # Add page-wide pairplots
    pdf.image(os.path.join(output_dir, 'pairplot.png'), Align.C, w=pdf.epw-20)
    pdf.add_page()

    # Add correlation plots
    pdf.image(os.path.join(output_dir, 'pearson_correlation.png'), Align.C, h=pdf.eph/2)
    pdf.image(os.path.join(output_dir, 'spearman_correlation.png'), Align.C, h=pdf.eph/2)

    # Add multiplots
    if multiplots and categorical_columns:
        pdf = add_multiplots(pdf, multiplots, categorical_columns)

    # Add demographic breakdown "table one"
    path_tableone = os.path.join(output_dir, 'tableone.csv')
    if os.path.exists(path_tableone):
        csv_df = pd.read_csv(path_tableone, na_filter=False).astype(str)
        pdf = add_table(pdf, csv_df)

    # Save PDF
    pdf.output(os.path.join(output_dir, 'analysis_report.pdf'))

def generate_explainer_report_pdf(
        problem_type: str,
        output_dir: str | Path = "./"):
    """
    Generate a PDF report of the explainer with plots.
    """
    output_dir = Path(output_dir)

    # Instantiate PDF
    pdf = FPDF()
    pdf.add_page()
    script_dir = Path(__file__).resolve().parent
    
    # Adding unicode fonts
    font_path = (script_dir / 'fonts/DejaVuSans.ttf')
    pdf.add_font("dejavu-sans", style="", fname=font_path)
    font_path = (script_dir / 'fonts/DejaVuSans-Bold.ttf')
    pdf.add_font("dejavu-sans", style="b", fname=font_path)
    pdf.set_font('dejavu-sans', '', 24)  

    # Title
    pdf.write(5, "Explainer Report\n\n")

    pdf.image((output_dir / 'figures' / 'test_metrics_bootstrap.png'), Align.C, h=pdf.eph//3.5, w=pdf.epw-20)
    pdf.image((output_dir / 'figures' / 'validation_metrics_bootstrap.png'), Align.C, h=pdf.eph//3.5, w=pdf.epw-20)
    pdf.image((output_dir / 'figures' /  'train_metrics_bootstrap.png'), Align.C, h=pdf.eph//3.5, w=pdf.epw-20)
    pdf.add_page()

    pdf.image((output_dir / 'figures' / 'feature_importance.png'), Align.C, w=pdf.epw-20)
    pdf.add_page()

    if problem_type in ['binary', 'multiclass']:
        pdf.image((output_dir / 'figures' / 'model_evaluation.png'), Align.C, w=pdf.epw-20)
        pdf.image((output_dir / 'figures' / 'confusion_matrix.png'), Align.C, h=pdf.eph/2, w=pdf.epw-20)
        pdf.add_page()

        pdf.image((output_dir / 'figures' / 'shap_barplot.png'), Align.C, h=pdf.eph/2, w=pdf.epw-20)
        pdf.image((output_dir /  'figures' / 'shap_heatmap.png'), Align.C, h=pdf.eph/2, w=pdf.epw-20)
    elif problem_type == 'regression':
        pdf.image((output_dir / 'figures' / 'residual_plot.png'), Align.C, h=pdf.eph/2, w=pdf.epw-20)
        pdf.image((output_dir /  'figures' / 'true_vs_predicted.png'), Align.C, h=pdf.eph/2, w=pdf.epw-20)

    # Save PDF
    pdf.output((output_dir / 'explainer_report.pdf'))