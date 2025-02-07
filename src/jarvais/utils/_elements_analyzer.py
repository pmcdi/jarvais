from fpdf import FPDF
from fpdf.enums import YPos
import pandas as pd

# UTILS
def _add_multiplots(pdf: FPDF, multiplots: list, categorical_columns: list, continuous_columns: list) -> FPDF:
    n_continuous = len(continuous_columns) + 2
    if n_continuous < 11:
        n_rows = 19 // n_continuous
    else:
        n_rows = 1

    # start first page of multiplots
    current_y = pdf.t_margin
    for n, (plot, cat) in enumerate(zip(multiplots, categorical_columns)):
        if n % n_rows == 0:
            pdf.add_page()
            current_y = pdf.get_y()
        
        pdf.set_font('inter', 'B', 24)
        pdf.set_y(current_y)
        pdf.cell(text=f"{cat.title()} Multiplots", new_y=YPos.NEXT)

        current_y = pdf.get_y() + 2

        img_width = pdf.epw
        img = pdf.image(
            plot, 
            x=pdf.l_margin, 
            y=current_y, 
            w=img_width, 
            keep_aspect_ratio=True
        )
        current_y += img.rendered_height + 6
        pdf.set_y(current_y)    

    return pdf

def _add_tableone(pdf: FPDF, data: pd.DataFrame) -> FPDF:
    # Keep empty header entries
    data.columns = [f"col_{n}" if 'Unnamed:' in header else header for n, header in enumerate(data.columns)]
    
    # save 
    n = data.iloc[0]['Overall']
    data = data.iloc[1:]
    
    continuous_columns = [col.replace(', mean (SD)', '') for col in data['col_0'].unique() if 'mean (SD)' in col]
    categorical_columns = [col.replace(', n (%)', '') for col in data['col_0'].unique() if 'n (%)' in col]

    # new page for continuous variables + title 
    pdf.add_page()
    pdf.set_font('inter', 'B', 24)
    pdf.cell(h=pdf.t_margin, text='Table 1: Continuous Variables', new_y=YPos.NEXT)

    # table starts here
    pdf.set_font('inter', '', 10)
    with pdf.table(col_widths=(1.5, 1, 1, 1), text_align=['LEFT', 'RIGHT', 'RIGHT', 'RIGHT']) as table:
        # header row
        row = table.row()
        row.cell('Feature Name')
        row.cell(f'Missing (n={n})')
        row.cell('Mean')
        row.cell('SD')

        # iterate through each continuous variable and render its data on a table
        for col in continuous_columns:
            data_row = data[data["col_0"].str.startswith(f"{col},")].iloc[0]

            # separate mean/sd columns
            data_row['Mean'] = data_row['Overall'].split(" (")[0]
            data_row['SD'] = data_row['Overall'].split(" (")[1].split(")")[0]

            # one row of continuous variable
            row = table.row()
            row.cell(col)
            row.cell(data_row['Missing'])
            row.cell(data_row['Mean'])
            row.cell(data_row['SD'])
    
    # start next table 10mm after
    pdf.set_y(pdf.get_y() + 10)
    pdf.set_font('inter', 'B', 24)
    pdf.cell(h=pdf.t_margin, text='Table 2: Categorical Variables', new_y=YPos.NEXT)

    # table starts here
    pdf.set_font('inter', '', 10)
    with pdf.table(col_widths=(1.5, 1, 1, 1), text_align=['LEFT', 'LEFT', 'RIGHT', 'RIGHT']) as table:
        # header row
        row = table.row()
        row.cell('Feature Name')
        row.cell('Value')
        row.cell(f'Count (n={n})')
        row.cell('%')

        # iterate through each categorical variable and render its data on a table
        for col in categorical_columns:
            rows = data[data["col_0"].str.startswith(f"{col},")].reset_index(drop=True)
            
            # separate mean/sd columns
            rows['n'] = rows['Overall'].apply(lambda x: x.split(" (")[0])
            rows['%'] = rows['Overall'].apply(lambda x: x.split(" (")[1].split(")")[0])
    
            # iterate through each value of the categorical variable
            for n, data_row in rows.iterrows():
                row = table.row()
                if n == 0:
                    row.cell(col, rowspan=len(rows))
                row.cell(data_row['col_1'])
                row.cell(data_row['n'])
                row.cell(data_row['%'])

    return pdf

def _add_outlier_analysis(pdf: FPDF, outlier_analysis: str) -> FPDF:
    from ast import literal_eval
    pdf.set_font('inter', '', 10)

    outliers = {}
    for line in outlier_analysis.splitlines():
        var = line.split("found in")[1].split(": [")[0].strip()
        if "No Outliers" in line:
            outliers[var] = "✓ No outliers found"
        else:
            outliers[var] = literal_eval(line.split(f"{var}: ")[1])
    
    # table starts here
    pdf.set_font('inter', '', 10)
    with pdf.table(col_widths=(1, 2), text_align=['LEFT', 'LEFT']) as table:
        # header row
        row = table.row()
        row.cell('Feature Name')
        row.cell('Outlier')

        # iterate through each categorical variable and render its data on a table
        for var in outliers:
            var_outs = outliers[var]
            if isinstance(var_outs, str):
                row = table.row()
                row.cell(var)
                row.cell(var_outs)
            else:
                for n, val in enumerate(var_outs): 
                    row = table.row()
                    if n == 0:
                        row.cell(var, rowspan=len(var_outs))
                    row.cell(val)

    return pdf
