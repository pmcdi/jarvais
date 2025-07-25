import json
from pathlib import Path

import numpy as np
import pandas as pd
import rich.repr
from sklearn.model_selection import train_test_split

from jarvais.explainer import Explainer
from jarvais.trainer.modules import (
    AutogluonTabularWrapper,
    FeatureReductionModule,
    SurvivalTrainerModule,
)
from jarvais.trainer.settings import TrainerSettings


class TrainerSupervised:
    """
    TrainerSupervised is a class for automating the process of feature reduction, 
    model training, and evaluation for various machine learning tasks.

    Parameters:
        output_dir (str | Path): The output directory for saving the trained model and data.
        target_variable (str | list[str]): The column name of the target variable, or a list of two column names for survival analysis.
        task (str): The type of task to perform, e.g. 'binary', 'multiclass', 'regression', or 'survival'.
        stratify_on (str | None): The column name of a variable to stratify the train-test split over. If None, no stratification will be performed.
        test_size (float): The proportion of data to use for testing. Default is 0.2.
        k_folds (int): The number of folds to use for cross-validation. Default is 5.
        reduction_method (str | None): The method to use for feature reduction. If None, no feature reduction will be performed.
        keep_k (int): The number of features to keep after reduction. Default is 2.
        random_state (int): The random state for reproducibility. Default is 42.
        explain (bool): Whether to generate explanations for the model. Default is False.
    """
    def __init__(
        self,
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

    @classmethod
    def from_settings(
            cls, 
            settings_dict: dict,
        ) -> "TrainerSupervised":
        """
        Initialize a TrainerSupervised instance with a given settings dictionary.

        Args:
            dict: A dictionary containing the settings for the TrainerSupervised instance.

        Returns:
            TrainerSupervised: An instance of TrainerSupervised with the given settings.
        """
        settings = TrainerSettings.model_validate(settings_dict)
        
        trainer = cls(
            output_dir=settings.output_dir,
            target_variable=settings.target_variable,
            task=settings.task,
        )

        trainer.reduction_module = settings.reduction_module
        trainer.trainer_module = settings.trainer_module

        trainer.settings = settings

        return trainer

    def run(
            self,
            data: pd.DataFrame 
        ) -> None:
        self.data = data

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
    
    def infer(self, data: pd.DataFrame, model: str | None = None) -> np.ndarray:
        """
        Make predictions on new data using the trained predictor.

        Args:
            data (pd.DataFrame): The new data to make predictions on.
            model (str | None): The model to use for prediction. If None, the best model will be used.

        Returns:
            np.ndarray: The predicted values.
        """

        if self.settings.task == 'survival':
            inference = self.predictor.predict(data, model)
        elif self.predictor.can_predict_proba:
            inference = self.predictor.predict_proba(data, model, as_pandas=False)[:, 1]
        else:
            inference = self.predictor.predict(data, model, as_pandas=False)

        return inference
    
    @classmethod
    def load_trainer(
            cls, 
            project_dir: str | Path
        ) -> "TrainerSupervised":

        from autogluon.tabular import TabularPredictor

        from jarvais.trainer.modules.survival_trainer import SurvivalPredictor

        project_dir = Path(project_dir)
        with (project_dir / 'trainer_settings.json').open('r') as f:
            trainer_config = json.load(f)

        trainer = cls.from_settings(trainer_config)

        if trainer.settings.task == 'survival':
            model_dir = (project_dir / 'survival_models')
            trainer.predictor = SurvivalPredictor.load(model_dir)
        else:
            model_dir = (project_dir / 'autogluon_models' / 'autogluon_models_best_fold')
            trainer.predictor = TabularPredictor.load(model_dir, verbosity=1)

        trainer.X_test = pd.read_csv(project_dir / 'data' / 'X_test.csv')
        trainer.X_val = pd.read_csv(project_dir / 'data' / 'X_val.csv')
        trainer.X_train = pd.read_csv(project_dir / 'data' / 'X_train.csv')
        trainer.y_test = pd.read_csv(project_dir / 'data' / 'y_test.csv').squeeze()
        trainer.y_val = pd.read_csv(project_dir / 'data' / 'y_val.csv').squeeze()
        trainer.y_train = pd.read_csv(project_dir / 'data' / 'y_train.csv').squeeze()
  
        return trainer

    def __rich_repr__(self) -> rich.repr.Result:
        yield self.settings

    def __repr__(self) -> str:
        return f"TrainerSupervised(settings={self.settings.model_dump_json(indent=2)})"


if __name__ == "__main__":
    from rich import print  # noqa: A004
    from jarvais.analyzer import Analyzer

    
    df = pd.read_csv('./data/RADCURE_processed_clinical.csv', index_col=0)
    df.drop(columns=["Study ID"], inplace=True)
    df.rename(columns={'survival_time': 'time', 'death':'event'}, inplace=True)

    analyzer = Analyzer(
        df,
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
        output_dir="temp_output/trainer_test_rad", 
        target_variable="Dose", 
        task="regression",
        k_folds=2
    )
        
    print(trainer)

    trainer.run(analyzer.data)
