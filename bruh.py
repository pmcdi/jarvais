from jarvais.explainer import SubgroupAnalysis
import pandas as pd

df = pd.read_csv('../auto-seg-bias/data/radcure_nnunet_with_clinical.csv', index_col=0)

sensitive_feats = ['Sex', 'Age', 'Ds Site', 'Stage', 'HPV', 'N', 'T', 'Chemo? ', 'Smoking Status']

bins = [0, 40, 60, 80, float('inf')]
labels = ["≤40", "41-60", "61-80", "80+"]
df['Age'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

for oar in df["OAR"].unique():
    _df = df[df["OAR"] == oar]

    analyzer = SubgroupAnalysis(_df[sensitive_feats], _df['APL'], '../auto-seg-bias/figures/radcure_bias')
    analyzer.intersectional_analysis(show_figure=False, tag=oar)