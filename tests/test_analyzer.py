import pytest
import yaml

from jarvais import Analyzer


def test_analyzer_radcure(
        radcure_clinical, 
        tmp_path
    ):
    radcure_clinical.rename(columns={'survival_time': 'time', 'death':'event'}, inplace=True)

    config = Analyzer.dry_run(radcure_clinical)
    config['columns']['categorical'].remove('Dose')
    config['columns']['continuous'].append('Dose') 

    with open(tmp_path / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    analyzer = Analyzer(
        radcure_clinical, 
        output_dir=tmp_path, 
        task='survival',
        target_variable='event', 
        config=tmp_path / 'config.yaml'
    )
    analyzer.run()

    expected_continuous_columns = {'time', 'age at dx', 'Dose'}
    assert set(analyzer.continuous_columns) == expected_continuous_columns

    expected_categorical_columns = {
        'Smoking Status', 'Sex', 
        'T Stage', 'N Stage', 
        'Disease Site', 'Stage', 
        'Chemotherapy', 'HPV Combined', 
        'event'
    }
    assert set(analyzer.categorical_columns) == expected_categorical_columns

    assert len(analyzer.multiplots) == len(analyzer.categorical_columns) # Should be 1 for each categorical column

    assert (tmp_path / 'config.yaml').exists()
    assert (tmp_path / 'tableone.csv').exists()
    assert (tmp_path / 'updated_data.csv').exists()

    assert (analyzer.output_dir / 'figures' / 'pearson_correlation.png').exists()
    assert (analyzer.output_dir / 'figures' / 'spearman_correlation.png').exists()
    assert (analyzer.output_dir / 'figures' / 'multiplots').exists()
    assert (analyzer.output_dir / 'figures' / 'frequency_tables').exists()
    assert (analyzer.output_dir / 'figures' / 'kaplan_meier').exists()


def test_analyzer_breast_cancer(
        breast_cancer, 
        tmp_path
    ):
    config = Analyzer.dry_run(breast_cancer)
    config['columns']['categorical'].remove('Age')
    config['columns']['continuous'].append('Age')
    config['columns']['categorical'].remove('Regional Node Examined')
    config['columns']['continuous'].append('Regional Node Examined')
    config['columns']['categorical'].remove('Reginol Node Positive')
    config['columns']['continuous'].append('Reginol Node Positive')

    with open(tmp_path / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    analyzer = Analyzer(breast_cancer, output_dir=tmp_path, target_variable='Status', config=tmp_path / 'config.yaml')
    analyzer.run()

    expected_continuous_columns = {'Survival Months', 'Tumor Size', 'Age', 'Regional Node Examined', 'Reginol Node Positive'}
    assert set(analyzer.continuous_columns) == expected_continuous_columns

    expected_categorical_columns = {
        'Progesterone Status', '6th Stage', 
        'T Stage ', 'Race', 
        'differentiate', 'Estrogen Status', 
        'Marital Status', 'Grade', 
        'A Stage', 'Status', 'N Stage'
    }
    assert set(analyzer.categorical_columns) == expected_categorical_columns

    assert len(analyzer.multiplots) == len(analyzer.categorical_columns) # Should be 1 for each categorical column

    assert (tmp_path / 'config.yaml').exists()
    assert (tmp_path / 'tableone.csv').exists()
    assert (tmp_path / 'updated_data.csv').exists()

    assert (analyzer.output_dir / 'figures' / 'pearson_correlation.png').exists()
    assert (analyzer.output_dir / 'figures' / 'spearman_correlation.png').exists()
    assert (analyzer.output_dir / 'figures' / 'multiplots').exists()
    assert (analyzer.output_dir / 'figures' / 'frequency_tables').exists()





