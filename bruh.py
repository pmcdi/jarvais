from jarvais.explainer.subgroup_analysis import *
import pandas as pd

# df = pd.read_csv('../auto-seg-bias/data/radcure_nnunet_with_clinical.csv', index_col=0)

# sensitive_feats = ['Sex', 'Age', 'Ds Site', 'Stage', 'HPV', 'N', 'T', 'Chemo? ', 'Smoking Status']

# bins = [0, 40, 60, 80, float('inf')]
# labels = ["≤40", "41-60", "61-80", "80+"]
# df['Age'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

df = pd.read_csv('../auto-seg-bias/data/nnunet700+_hnscc_metrics.csv', index_col=0)

sensitive_feats = ['Gender', 'Age at Diag', 'Smoking status', 'HPV Status', 'T-category', 'N-category', 'AJCC Stage (7th edition)']

bins = [0, 40, 60, 80, float('inf')]
labels = ["≤40", "41-60", "61-80", "80+"]
df['Age at Diag'] = pd.cut(df['Age at Diag'], bins=bins, labels=labels, right=True)

# df = pd.read_csv('../auto-seg-bias/data/nnunet700+_quebec_metrics_with_clinical.csv', index_col=0)

# sensitive_feats = ['Sex', 'Age', 'Primary Site', 'T-stage', 'N-stage', 'M-stage', 'TNM group stage', 'HPV status']

# bins = [0, 40, 60, 80, float('inf')]
# labels = ["≤40", "41-60", "61-80", "80+"]
# df['Age'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

_df = df[df["OAR"] == 'Larynx']

for metric in ['APL', 'VolDice', '95HD']:
        print(metric)
        intersectional_analysis(_df[sensitive_feats], _df[metric], 'bruh', show_figure=False, tag=f'{'Larynx'}+{metric}')
        
        generate_violin(_df[sensitive_feats], _df[metric], 'bruh', show_figure=False, tag=f'{'Larynx'}+{metric}')

# for oar, count in df["OAR"].value_counts().items():
#     _df = df[df["OAR"] == oar]

#     print(oar)
#     for metric in ['APL', 'VolDice', '95HD']:
#         print(metric)
#         analyzer = SubgroupAnalysis(_df[sensitive_feats], _df[metric], '../auto-seg-bias/figures/hnscc_bias/pval_heatmaps')
#         analyzer.intersectional_analysis(show_figure=False, tag=f'{oar}(n={count})+{metric}')

#         analyzer = SubgroupAnalysis(_df[sensitive_feats], _df[metric], '../auto-seg-bias/figures/hnscc_bias/violin_plots')
#         analyzer.generate_violin(show_figure=False, tag=f'{oar}(n={count})')

