import sys

import os

# Get the absolute path of the project root directory
project_root = os.path.abspath(os.path.join(os.getcwd()))

# Add the project root directory to the Python path
sys.path.append(project_root)


import pandas as pd

# Define the path to the data directory
data_dir = os.path.join(project_root, 'data')

# Example: Access a specific data file in the data directory
data_file_path = os.path.join(data_dir, 'RADCURE_challenge_clinical.csv')
df = pd.read_csv(data_file_path, index_col=0)
df['binary_death'] = df.apply(lambda x: 1 if x['death'] == 'Yes' and x['survival_time'] < 2 else 0, axis=1)

df.drop(columns=["split", "survival_time", "death"], inplace=True)

from jarvais.analyzer import Analyzer

analyzer = Analyzer(df, target_variable='binary_death', output_dir='./tutorials/outputs')
analyzer.run()