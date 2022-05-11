from pathlib import Path
from joblib import dump

import click
import mlflow
import mlflow.sklearn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_validate

from .data import get_dataset


@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--search",
    default="NestedCV",
    type=click.Choice(["KFold", "NestedCV"]),
    show_default=True,
)
@click.option(
    "--select-model",
    default="logist_regression",
    type=click.Choice(["logist_regression", "random_forest"]),
    show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
@click.option(
    "--feature-engineering",
    default="standard_scaler",
    type=click.Choice(["standard_scaler", "min_max_scaler", "None"]),
    show_default=True,
)
@click.option(
    "--max-iter",
    default=500,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)
@click.option(
    "--n-estimators",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--max-depth",
    default=None,
    type=int,
    show_default=True,
)
@click.option(
    "--max-features",
    default=1,
    type=click.FloatRange(0, 1, min_open=True, max_open=False),
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    feature_engineering: str,
    max_iter: int,
    logreg_c: float,
    select_model: str,
    n_estimators: int,
    max_depth: int,
    max_features: float,
    search: str,
) -> None:
    features_train, target_train = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )
    with mlflow.start_run():
        if feature_engineering == "standard_scaler":
            features_train = StandardScaler().fit_transform(features_train)
        elif feature_engineering == "min_max_scaler":
            features_train = MinMaxScaler().fit_transform(features_train)
        cv = KFold(n_splits=8, shuffle=True, random_state=random_state)

        if search == "KFold":
            if select_model == "logist_regression":
                model = LogisticRegression(
                    random_state=random_state, max_iter=max_iter, C=logreg_c, n_jobs=-1
                )
            elif select_model == "random_forest":
                model = RandomForestClassifier(
                    random_state=random_state,
                    max_depth=max_depth,
                    max_features=max_features,
                    n_estimators=n_estimators,
                    n_jobs=-1,
                )
            scores = cross_validate(
                model,
                features_train,
                target_train,
                cv=cv,
                scoring=("accuracy", "f1_macro", "roc_auc_ovr"),
            )
            accuracy = scores["test_accuracy"].mean()
            f1_macro = scores["test_f1_macro"].mean()
            roc_auc_ovr = scores["test_roc_auc_ovr"].mean()

            if select_model == "logist_regression":
                mlflow.log_param("max_iter", max_iter)
                mlflow.log_param("logreg_c", logreg_c)
            elif select_model == "random_forest":
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_param("max_features", max_features)
                mlflow.log_param("n_estimators", n_estimators)
        elif search == "NestedCV":
            if select_model == "logist_regression":
                model = LogisticRegression(random_state=random_state, n_jobs=-1)
                space = dict()
                space["max_iter"] = [10, 100, 200, 400, 500, 700]
                space["C"] = [0.001, 0.01, 0.1, 1.0, 10.0]
            elif select_model == "random_forest":
                model = RandomForestClassifier(random_state=random_state, n_jobs=-1)
                space = dict()
                space["max_depth"] = [None, 5, 10, 50, 70]
                space["max_features"] = [0.2, 0.4, 0.7, 1.0]
                space["n_estimators"] = [20, 50, 100, 200]
                # poetry run train --select-model random_forest
                # space['max_depth'] = [None, 5]
                # space['max_features'] = [0.7,1.0]
                # space['n_estimators'] = [20, 100]
            search_q = GridSearchCV(model, space, scoring="accuracy", n_jobs=-1, cv=cv)
            cv_outer = KFold(n_splits=5)
            scores = cross_validate(
                search_q,
                features_train,
                target_train,
                cv=cv_outer,
                scoring=("accuracy", "f1_macro", "roc_auc_ovr"),
            )
            accuracy = scores["test_accuracy"].mean()
            f1_macro = scores["test_f1_macro"].mean()
            roc_auc_ovr = scores["test_roc_auc_ovr"].mean()

            search_q.fit(features_train, target_train)
            print(
                "The parameters combination that would give best accuracy is : ",
                search_q.best_params_,
            )
            if select_model == "logist_regression":
                mlflow.log_param("max_iter", search_q.best_params_["max_iter"])
                mlflow.log_param("logreg_c", search_q.best_params_["C"])
            elif select_model == "random_forest":
                mlflow.log_param("max_depth", search_q.best_params_["max_depth"])
                mlflow.log_param("max_features", search_q.best_params_["max_features"])
                mlflow.log_param("n_estimators", search_q.best_params_["n_estimators"])
        mlflow.log_param("feature_engineering", feature_engineering)
        mlflow.log_param("select_model", select_model)
        mlflow.log_param("search", search)
        mlflow.log_param("random_state", random_state)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("roc_auc_ovr", roc_auc_ovr)
        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"f1_macro: {f1_macro}.")
        click.echo(f"roc_auc_ovr: {roc_auc_ovr}.")
        click.echo(f"select_model: {select_model}.")
        dump(model, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
