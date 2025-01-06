from AutoML.analyzer import Analyzer
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
import shutil

@pytest.fixture
def sample_data():
    data = {
        'A': [1, 2, 3, 4, 5],
        'B': [5, 4, 3, 2, 1],
        'C': ['a', 'b', 'a', 'a', 'b'],
        'D': [None, 2, None, 4, 5]
    }
    df= pd.concat([pd.DataFrame(data), pd.DataFrame(data), pd.DataFrame(data)], axis=0)
    return df.reset_index()

@pytest.fixture
def tmpdir():
    temp_path = Path("./tests/tmp")
    temp_path.mkdir(parents=True, exist_ok=True)

    for file in temp_path.iterdir():
        file_path = temp_path / file
        if file_path.is_file() or file_path.is_symlink():
            file_path.unlink() 
        elif file_path.is_dir():
            shutil.rmtree(file_path) 
                    
    yield temp_path

@pytest.fixture
def analyzer(sample_data, tmpdir):
    config_file = tmpdir / 'config.yaml'
    output_dir = tmpdir
    if "data.csv" not in output_dir.iterdir():
        sample_data.to_csv(output_dir / 'data.csv', index=False)
    return Analyzer(data=sample_data, output_dir=output_dir)

def test_analyzer_initialization(analyzer, sample_data):
    assert analyzer.data.equals(sample_data)
    assert analyzer.target_variable is None
    assert analyzer.output_dir is not None
    assert analyzer.config is None

def test_replace_missing(analyzer):
    analyzer._create_config()
    analyzer.config['missingness_strategy']['continuous']['D'] = 'mean'
    analyzer._apply_config() # replace missing happens in here
    
    assert analyzer.data['D'].isna().sum() == 0
    assert np.isclose(analyzer.data['D'].iloc[0], analyzer.data['D'].mean(), rtol=1e-4)

def test_create_config(analyzer):
    # _infer_types is always run inside _create_config
    analyzer._create_config()
    assert 'A' in analyzer.continuous_columns
    assert 'B' in analyzer.continuous_columns
    assert 'C' in analyzer.categorical_columns

def test_create_multiplots(analyzer):
    analyzer.categorical_columns = ['C']
    analyzer.continuous_columns = ['A', 'B']
    analyzer.umap_data = pd.DataFrame.from_dict({'UMAP1': [i for i in range(1, 16)], 'UMAP2': [j for j in range(15, 0, -1)]}).to_numpy()
    analyzer._create_multiplots(figures_dir=analyzer.output_dir / 'figures')
    assert len(analyzer.multiplots) > 0

def test_run(analyzer):
    analyzer.run()
    assert (analyzer.output_dir / 'tableone.csv').exists()
    assert (analyzer.output_dir / 'updated_data.csv').exists()
    assert (analyzer.output_dir / 'figures' / 'pearson_correlation.png').exists()
    assert (analyzer.output_dir / 'figures' / 'spearman_correlation.png').exists()
    assert (analyzer.output_dir / 'figures' / 'multiplots').exists()
    assert (analyzer.output_dir / 'figures' / 'frequency_tables').exists()
