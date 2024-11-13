from AutoML.analyzer import Analyzer
import pytest
import numpy as np
import pandas as pd
import os
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
    temp_path = "./tests/tmp"
    if not os.path.exists(temp_path):
        os.makedirs(temp_path, exist_ok=True)
    else:
        for file in os.listdir(temp_path):
            file_path = os.path.join(temp_path, file)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
            
    yield temp_path
    
    # Cleanup after tests
    # shutil.rmtree(temp_path, ignore_errors=True)

@pytest.fixture
def analyzer(sample_data, tmpdir):
    config_file = os.path.join(tmpdir, 'config.yaml')
    output_dir = tmpdir
    if "data.csv" not in os.listdir(output_dir):
        sample_data.to_csv(os.path.join(output_dir, 'data.csv'), index=False)
    return Analyzer(data=sample_data, output_dir=output_dir)

def test_analyzer_initialization(analyzer, sample_data):
    assert analyzer.data.equals(sample_data)
    assert analyzer.target_variable is None
    assert analyzer.output_dir is not None
    assert analyzer.config is None

def test_replace_missing(analyzer):
    analyzer.config = {
        'missingness_strategy': {
            'continuous': {'D': 'mean'},
            'categorical': {}
        }
    }
    analyzer.continuous_columns = ['D']
    analyzer.categorical_columns = []
    analyzer._replace_missing()
    assert analyzer.data['D'].isna().sum() == 0
    assert np.isclose(analyzer.data['D'].iloc[0], analyzer.data['D'].mean(), rtol=1e-4)

def test_run_janitor(analyzer):
    # _infer_types is always run inside _run_janitor
    analyzer._run_janitor()
    assert 'A' in analyzer.continuous_columns
    assert 'B' in analyzer.continuous_columns
    assert 'C' in analyzer.categorical_columns

def test_create_multiplots(analyzer):
    analyzer.categorical_columns = ['C']
    analyzer.continuous_columns = ['A', 'B']
    analyzer.umap_data = pd.DataFrame.from_dict({'UMAP1': [i for i in range(1, 16)], 'UMAP2': [j for j in range(15, 0, -1)]}).to_numpy()
    analyzer._create_multiplots()
    assert len(analyzer.multiplots) > 0

def test_run(analyzer):
    analyzer.run()
    assert os.path.exists(os.path.join(analyzer.output_dir, 'tableone.csv'))
    assert os.path.exists(os.path.join(analyzer.output_dir, 'updated_data.csv'))
    assert os.path.exists(os.path.join(analyzer.output_dir, 'pearson_correlation.png'))
    assert os.path.exists(os.path.join(analyzer.output_dir, 'spearman_correlation.png'))
    assert os.path.exists(os.path.join(analyzer.output_dir, 'multiplots'))
