import pandas as pd
import pytest

from jarvais.analyzer.modules.encoding import OneHotEncodingModule


def test_one_hot_encoding_module():
    df = pd.DataFrame({
        "sex": ["M", "F", "F"],
        "target": [1, 0, 1],
    })
    module = OneHotEncodingModule.build(categorical_columns=["sex"], target_variable="target", prefix_sep="_")
    encoded = module(df)

    assert "sex_M" in encoded.columns
    assert "sex_F" in encoded.columns
    assert "sex" not in encoded.columns, "Original column should be replaced"
    assert isinstance(encoded, pd.DataFrame)