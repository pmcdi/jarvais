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

