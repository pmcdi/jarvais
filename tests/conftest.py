import pytest
import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter

@pytest.fixture(scope="session")
def radcure_clinical():
    """
    Read in the processed RADCURE clinical data

    DOI: 10.7937/J47W-NM11
    """
    df = pd.read_csv('https://raw.githubusercontent.com/pmcdi/jarvais/main/data/RADCURE_processed_clinical.csv', index_col=0)
    df.drop(columns=["Study ID"], inplace=True)

    return df

# @pytest.fixture(scope="session")
# def breast_cancer():
#     """
#     Download and return breast cancer dataset.

#     DOI: 10.21227/a9qy-ph35
#     """

#     df = kagglehub.load_dataset(
#         KaggleDatasetAdapter.PANDAS,
#         "reihanenamdari/breast-cancer",
#         'Breast_Cancer.csv',
#     )

#     return df

