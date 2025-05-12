import pandas as pd
import pytest

from jarvais.analyzer.modules.missingness import MissingnessModule


def test_missingness_module_removes_nans():
    df = pd.DataFrame({
        "age": [25, None, 35],
        "sex": ["M", "F", None],
    })
    module = MissingnessModule.build(categorical_columns=["sex"], continuous_columns=["age"])
    out = module(df)

    assert not out.isnull().any().any(), "Missing values should be handled"
    assert isinstance(out, pd.DataFrame)