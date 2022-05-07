from pathlib import Path
from joblib import dump

import click
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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
    "--select-model",
    default='logist_regression',
    type=str,
    show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--max-iter",
    default=100,
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
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
    select_model:str,
    n_estimators:int,
    max_depth:int,
    max_features:int,
) -> None:
    features_train, target_train = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )
    with mlflow.start_run():
        if use_scaler:       
            features_train = StandardScaler().fit_transform(features_train)
        cv = KFold(n_splits=5)
        if(select_model=='logist_regression'):
            model=LogisticRegression(random_state=random_state, max_iter=max_iter, C=logreg_c,n_jobs=-1)
        elif(select_model=='random_forest'):
            model=RandomForestClassifier(random_state=random_state,max_depth=max_depth,max_features=max_features,n_estimators=n_estimators,n_jobs=-1)
        accuracy = (cross_val_score(model, features_train, target_train, cv = cv, scoring='accuracy')).mean()
        f1_micro = (cross_val_score(model, features_train, target_train, cv = cv, scoring='f1_micro')).mean()
        roc_auc_ovr = (cross_val_score(model, features_train, target_train, cv = cv, scoring='roc_auc_ovr')).mean()
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("logreg_c", logreg_c)
        mlflow.log_param("select_model", select_model)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("max_features", max_features)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_micro", f1_micro)
        mlflow.log_metric("roc_auc_ovr", roc_auc_ovr)
        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"f1_micro: {f1_micro}.")
        click.echo(f"roc_auc_ovr: {roc_auc_ovr}.")
        click.echo(f"select_model: {select_model}.")
        dump(model, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")
