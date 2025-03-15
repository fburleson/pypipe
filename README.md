# pypipe :gear:
A simple, easy to use library for automated machine learning and highly composable pipelines. Made with scikit-learn and pandas.

## Introduction :book:
This project was made to showcase my understanding of the machine learning pipeline/workflow and scikit-learn. It allows for quick development by automating model selection and hyperparameter optimization. Made to solve classification and regression problems whilst being being modular and easy to use.

## Features :sparkles:
- Building highly composable, custom machine learning pipelines, from preprocessing to model evaluation
- Builtin compatibility with numpy, pandas, scikit-learn transformers and scikit-learn estimators.
- Automated model selection and hyperparameter optimization
- Saving and loading pipelines

## Requirements :clipboard:
- Python >= 3.11

## Installation :gear:
### Linux
```bash
git clone --depth 1 git@github.com:fburleson/pypipe.git pypipe
cd pypipe
python3 -m venv .venv
source .venv/bin/activate 
pip install -e .
pytest
```
### Windows (Powershell)
```bash
git clone --depth 1 https://github.com/fburleson/pypipe.git pypipe
cd pypipe
python -m venv .venv
.\.venv\Scripts\activate 
pip install -e .
pytest
```

## Usage :computer:
```bash
python run.py
```
or
```bash
python3 run.py
```

## Pipeline building example
```python
from functools import partial
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from pypipe.deploy import save_pipeline, load_pipeline
from pypipe.compose import Pipeline, Model
from pypipe.segments.util import Subset, Split


def main():
    features: list[str] = [
        "sex",
        "age",
        "fare",
        "embarked",
        "sibsp",
        "pclass",
        "parch",
    ]
    encoded_features: list[str] = features.copy()
    encoded_features.remove("embarked")
    encoded_features.extend(["embarked_C", "embarked_Q", "embarked_S"])
    targets: list[str] = ["survived"]
    preprocess: Pipeline = Pipeline(
        Subset(features + targets),
        ColumnTransformer(
            [
                ("mean", SimpleImputer(strategy="mean"), ["age", "fare"]),
                (
                    "mode",
                    SimpleImputer(strategy="most_frequent"),
                    ["embarked", "survived"],
                ),
                ("onehot", OneHotEncoder(sparse_output=False), ["embarked"]),
                ("label", OrdinalEncoder(), ["sex"]),
            ],
            verbose_feature_names_out=False,
            remainder="passthrough",
        ),
        Split(encoded_features, targets),
        (
            partial(pd.DataFrame.to_numpy, copy=True),
            partial(pd.DataFrame.to_numpy, copy=True),
        ),
        (lambda x: x, np.squeeze),
    )
    save_pipeline(preprocess, "titanic_preprocess_pipeline.pipe")


def train_test():
    raw_openml = fetch_openml("titanic", version=1, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        raw_openml.data, raw_openml.target
    )
    train_data: pd.DataFrame = pd.concat([X_train, y_train], axis=1)
    test_data: pd.DataFrame = pd.concat([X_test, y_test], axis=1)
    preprocess: Pipeline = load_pipeline("titanic_preprocess_pipeline.pipe")
    train: Pipeline = Pipeline(
        *preprocess,
        DecisionTreeClassifier(),
    )
    model: Model = train(train_data)
    test: Pipeline = Pipeline(
        *preprocess,
        (model.forward, lambda x: x),
    )
    y_pred, y_true = test(test_data)
    print(f"{accuracy_score(y_true, y_pred):.2%}")


if __name__ == "__main__":
    main()
    train_test()
```
