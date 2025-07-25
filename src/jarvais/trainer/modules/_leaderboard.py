import pandas as pd

def format_leaderboard(leaderboard: pd.DataFrame, eval_metric: str, extra_metrics: list, score_col_name: str) -> pd.DataFrame:

    if score_col_name == 'score_val' and 'score_val' in leaderboard.columns:
        leaderboard = leaderboard.drop(score_col_name, axis=1)
    leaderboard = leaderboard.rename(columns={'score_test': score_col_name})

    def format_scores(row: pd.Series, score_col: str, metrics: list[str]) -> str:
        """Format scores as a string with all specified metrics."""
        output = f"{eval_metric.upper()}: {row[score_col]}"
        
        for metric in metrics:
            metric = str(metric) # Doing this for the custom auprc scorer
            
            if metric in row:
                output += f"\n{metric.upper()}: {row[metric]:>10}"
        
        return output

    leaderboard[score_col_name] = leaderboard.apply(
        lambda row: format_scores(row, score_col_name, extra_metrics),
        axis=1
    )
    return leaderboard[['model', score_col_name]]

def aggregate_folds(consolidated_leaderboard: pd.DataFrame, extra_metrics: list) -> pd.DataFrame:
    extra_metrics = [str(item) for item in extra_metrics]

    metrics_to_aggregate = {k: ['mean', 'min', 'max'] for k in ['score_test', *extra_metrics]}

    aggregated_leaderboard = consolidated_leaderboard.groupby('model').agg(metrics_to_aggregate).reset_index() # type: ignore

    final_leaderboard = pd.DataFrame({'model': aggregated_leaderboard['model']})

    for metric in metrics_to_aggregate:
        final_leaderboard[metric] = [
            f'{round(row[0], 2)} [{round(row[1], 2)}, {round(row[2], 2)}]'
            for row in aggregated_leaderboard[metric].values
        ]

    return final_leaderboard