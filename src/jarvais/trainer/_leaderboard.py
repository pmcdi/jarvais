import pandas as pd

def format_leaderboard(leaderboard: pd.DataFrame, extra_metrics: list, score_col_name: str) -> pd.DataFrame:

    if score_col_name == 'score_val' and 'score_val' in leaderboard.columns:
        leaderboard = leaderboard.drop(score_col_name, axis=1)
    leaderboard = leaderboard.rename(columns={'score_test': score_col_name})

    def format_scores(row, score_col, extra_metrics):
        """Format scores as a string with AUROC, F1, and AUPRC. Or with R2 and RMSE for regression."""
        if 'f1' in extra_metrics:
            return f"AUROC {row[score_col]}\nF1: {row['f1']}\nAUPRC: {row['auprc']}"
        else:
            return f"R2 {row[score_col]}\nRMSE: {row['root_mean_squared_error']}"

    leaderboard[score_col_name] = leaderboard.apply(
        lambda row: format_scores(row, score_col_name, extra_metrics),
        axis=1
    )
    return leaderboard[['model', score_col_name]]

def aggregate_folds(consolidated_leaderboard:list, extra_metrics: list) -> pd.DataFrame:
    extra_metrics = [str(item) for item in extra_metrics]

    to_agg = {k: ['mean', 'min', 'max'] for k in ['score_test', *extra_metrics]}

    aggregated_leaderboard = consolidated_leaderboard.groupby('model').agg(to_agg).reset_index()

    final_leaderboard = pd.DataFrame({'model': aggregated_leaderboard['model']})

    for col in to_agg.keys():
        final_leaderboard[col] = [
            f'{round(row[0], 2)} [{round(row[1], 2)}, {round(row[2], 2)}]'
            for row in aggregated_leaderboard[col].values
        ]

    return final_leaderboard
