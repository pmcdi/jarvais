import json
import yaml

from jarvais import Analyzer


def test_analyzer_radcure(
        radcure_clinical, 
        tmp_path
    ):
    radcure_clinical.rename(columns={'survival_time': 'time', 'death':'event'}, inplace=True)

    analyzer = Analyzer(
        radcure_clinical, 
        output_dir=tmp_path, 
        task='survival',
        target_variable='event',
        continuous_columns=['time', 'age at dx', 'Dose'],
        categorical_columns=[
            'Smoking Status', 'Sex', 
            'T Stage', 'N Stage', 
            'Disease Site', 'Stage', 
            'Chemotherapy', 'HPV Combined', 
            'event'
        ]
    )

    analyzer.run()

    assert (tmp_path / 'analyzer_settings.json').exists()
    assert (tmp_path / 'analyzer_settings.schema.json').exists()
    assert (tmp_path / 'tableone.csv').exists()
    assert (tmp_path / 'updated_data.csv').exists()
    
    assert (tmp_path  / 'figures' / 'pearson_correlation.png').exists()
    assert (tmp_path  / 'figures' / 'spearman_correlation.png').exists()
    assert (tmp_path  / 'figures' / 'frequency_tables').exists()
    assert (tmp_path  / 'figures' / 'kaplan_meier').exists()
    assert (tmp_path  / 'figures' / 'multiplots').exists()

    assert len(analyzer.visualization_module._multiplots) == len(analyzer.settings.categorical_columns) # Should be same as number of categorical columns

    settings_path = tmp_path / 'analyzer_settings.json'
    with settings_path.open() as f:
        settings_dict = json.load(f)
    
    analyzer = Analyzer.from_settings(radcure_clinical, settings_dict)


# def test_analyzer_breast_cancer(
#         breast_cancer, 
#         tmp_path
#     ):
    
#     analyzer = Analyzer(
#         breast_cancer,
#         output_dir=tmp_path,
#         target_variable='Status',
#         continuous_columns=['Survival Months', 'Tumor Size', 'Age', 'Regional Node Examined', 'Reginol Node Positive'],
#         categorical_columns=[
#             'Progesterone Status', '6th Stage', 
#             'T Stage ', 'Race', 
#             'differentiate', 'Estrogen Status', 
#             'Marital Status', 'Grade', 
#             'A Stage', 'Status', 'N Stage'
#         ]
#     )

#     analyzer.run()

#     assert (tmp_path / 'analyzer_settings.json').exists()
#     assert (tmp_path / 'analyzer_settings.schema.json').exists()
#     assert (tmp_path / 'tableone.csv').exists()
#     assert (tmp_path / 'updated_data.csv').exists()
    
#     assert (tmp_path  / 'figures' / 'pearson_correlation.png').exists()
#     assert (tmp_path  / 'figures' / 'spearman_correlation.png').exists()
#     assert (tmp_path  / 'figures' / 'frequency_tables').exists()
#     assert (tmp_path  / 'figures' / 'multiplots').exists()

#     assert len(analyzer.visualization_module._multiplots) == len(analyzer.settings.categorical_columns) # Should be same as number of categorical columns
    
#     settings_path = tmp_path / 'analyzer_settings.json'
#     with settings_path.open() as f:
#         settings_dict = json.load(f)
    
#     analyzer = Analyzer.from_settings(breast_cancer, settings_dict)




