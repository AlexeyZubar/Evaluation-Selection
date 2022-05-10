# Evaluation-Selection it's my homework
## Usage
This package allows you to train model for detecting the presence of heart disease in the patient.
1. Clone this repository to your machine.
2. Download [Heart Disease](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) dataset, save csv locally (default path is *data/heart.csv* in repository's root).
3. Make sure Python 3.9 and [Poetry](https://python-poetry.org/docs/) are installed on your machine (I use Poetry 1.1.11).
4. Install the project dependencies (*run this and following commands in a terminal, from the root of a cloned repository*):
```sh
poetry install --no-dev
```
5. Run train with the following command:
```sh
poetry run train -d <path to csv with data> -s <path to save trained model>
```
You can configure additional options (such as hyperparameters) in the CLI. To get a full list of them, use help:
```sh
poetry run train --help
```
For example, you can select another algorithm (logist_regression is the default) by calling
```sh
poetry run train --select-model random_forest
```
For example, you can select another search method (NestedCV is the default) by calling
```sh
poetry run train --search KFold
```
6. Run MLflow UI to see the information about experiments you conducted:
```sh
poetry run mlflow ui
```
