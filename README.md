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

## Development

The code in this repository must be tested, formatted with black, and pass mypy typechecking before being commited to the repository.

Install all requirements (including dev requirements) to poetry environment:
```
poetry install
```

Task 8
i.
![image](https://user-images.githubusercontent.com/20214519/167739977-adcb2a9d-d227-4231-8804-a732fd655166.png)
ii.
![image](https://user-images.githubusercontent.com/20214519/167740444-4d0c5129-f430-497f-a209-a0039eea3456.png)
iii.
![image](https://user-images.githubusercontent.com/20214519/167740699-9a9d1649-e3db-4da9-8353-d30542b3e54b.png)

