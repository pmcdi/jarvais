import json
import pandas as pd
from pathlib import Path
import rich.repr
from sklearn.model_selection import train_test_split

from jarvais.explainer import Explainer
from jarvais.loggers import logger
from jarvais.trainer.modules import (
    FeatureReductionModule, 
    SurvivalTrainerModule, 
    AutogluonTabularWrapper,
)
from jarvais.trainer.settings import TrainerSettings


class TrainerSupervised:
    def __init__(
        self,
        data: pd.DataFrame,
        output_dir: str | Path,
        target_variable: str | list[str],
        task: str,
        stratify_on: str | None = None,
        test_size: float = 0.2,
        k_folds: int = 5,
        reduction_method: str | None = None,
        keep_k: int = 2,
        random_state: int = 42,
        explain: bool = False
    ) -> None:
        
        self.data = data    

        self.reduction_module = FeatureReductionModule.build(
            method=reduction_method,
            task=task,
            keep_k=keep_k
        )

        if task == "survival":
            if set(target_variable) != {'time', 'event'}: 
                raise ValueError("Target variable must be a list of ['time', 'event'] for survival analysis.")

            self.trainer_module = SurvivalTrainerModule.build(
                output_dir=output_dir 
            )
        else:
            self.trainer_module = AutogluonTabularWrapper.build(
                output_dir=output_dir,
                target_variable=target_variable,
                task=task,
                k_folds=k_folds
            )

        self.settings = TrainerSettings(
            output_dir=Path(output_dir),
            target_variable=target_variable,
            task=task,
            stratify_on=stratify_on,
            test_size=test_size,
            random_state=random_state,
            explain=explain,
            reduction_module=self.reduction_module,
            trainer_module=self.trainer_module,
        )

    def run(self):

        # Preprocess
        X = self.data.drop(self.settings.target_variable, axis=1)
        y = self.data[self.settings.target_variable]

        X, y = self.reduction_module(X, y)     

        if self.settings.task in {'binary', 'multiclass'}:
            stratify_col = (
                y.astype(str) + '_' + self.data[self.settings.stratify_on].astype(str)
                if self.settings.stratify_on is not None
                else y
            )
        else:
            stratify_col = None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, 
            y, 
            test_size=self.settings.test_size, 
            stratify=stratify_col, 
            random_state=self.settings.random_state
        )

        # Train
        self.predictor, self.X_val, self.y_val = self.trainer_module.fit(
            X_train=self.X_train, 
            y_train=self.y_train, 
            X_test=self.X_test, 
            y_test=self.y_test
        )

        self.X_train = self.X_train.drop(self.X_val.index)
        self.y_train = self.y_train.drop(self.y_val.index)

        data_dir = self.settings.output_dir / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        self.X_train.to_csv((data_dir / 'X_train.csv'), index=False)
        self.X_test.to_csv((data_dir / 'X_test.csv'), index=False)
        self.X_val.to_csv((data_dir / 'X_val.csv'), index=False)
        self.y_train.to_csv((data_dir / 'y_train.csv'), index=False)
        self.y_test.to_csv((data_dir / 'y_test.csv'), index=False)
        self.y_val.to_csv((data_dir / 'y_val.csv'), index=False)

        if self.settings.explain:
            explainer = Explainer.from_trainer(self)
            explainer.run()

        # Save Settings
        schema_path = self.settings.output_dir / 'trainer_settings.schema.json'
        with schema_path.open("w") as f:
            json.dump(self.settings.model_json_schema(), f, indent=2)

        settings_path = self.settings.output_dir / 'trainer_settings.json'
        with settings_path.open('w') as f:
            json.dump({
                "$schema": str(schema_path.relative_to(self.settings.output_dir)),
                **self.settings.model_dump(mode="json") 
            }, f, indent=2)

    def model_names(self) -> list[str]:
        """
        Returns all trainer model names.

        This method retrieves the names of all models associated with the 
        current predictor. It requires that the predictor has been trained.

        Returns:
            list: A list of model names available in the predictor.
        """

        return self.predictor.model_names()

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.settings


if __name__ == "__main__":
    from rich import print
    import numpy as np

    # np.random.seed(0)
    # data = pd.DataFrame(
    #     {
    #         'time': np.random.randint(1, 100, 50),
    #         'event': np.random.randint(0, 2, 50),
    #         'age': np.random.randint(20, 80, 50),
    #         # 'sex': np.random.choice(["M", "F"], 50),
    #         'tumor_stage': np.random.randint(1, 10, 50)
    #     }
    # )

    # trainer = TrainerSupervised(
    #     data, 
    #     output_dir="temp_output/trainer_test", 
    #     target_variable=["event", "time"], 
    #     task="survival"
    # )

    # print(trainer)

    # trainer.run()

    from jarvais.analyzer import Analyzer

    
    df = pd.read_csv('./data/RADCURE_processed_clinical.csv', index_col=0)
    df.drop(columns=["Study ID"], inplace=True)
    df.rename(columns={'survival_time': 'time', 'death':'event'}, inplace=True)

    analyzer = Analyzer(
        data=df, 
        output_dir='./survival_outputs/analyzer',
        categorical_columns= [
        "Sex",
        "T Stage",
        "N Stage",
        "Stage",
        "Smoking Status",
        "Disease Site",
        "HPV Combined",
        "Chemotherapy"
        ],
        continuous_columns = [
        "time",
        "age at dx",
        "Dose"
        ],
        target_variable='event', 
        task='survival'
    )

    analyzer.visualization_module.enabled = False

    print(analyzer)

    analyzer.run()

    # analyzer.data["event"] = analyzer.data['event'].astype(bool)
    trainer = TrainerSupervised(
        analyzer.data,
        output_dir="temp_output/trainer_test_rad", 
        target_variable=["event", "time"], 
        task="survival",
        k_folds=2
    )

    # trainer.trainer_module.deep_models = []
        
    print(trainer)

    trainer.run()
