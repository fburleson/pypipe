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


def preprocess_train():
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
    raw_openml = fetch_openml("titanic", version=1, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        raw_openml.data, raw_openml.target
    )
    train_data: pd.DataFrame = pd.concat([X_train, y_train], axis=1)
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
    train: Pipeline = Pipeline(
        *preprocess,
        DecisionTreeClassifier(),
    )
    model: Model = train(train_data)
    test: Pipeline = Pipeline(
        *preprocess,
        (model.forward, lambda x: x),
    )
    save_pipeline(test, "model.pipe")


def test():
    raw_openml = fetch_openml("titanic", version=1, as_frame=True)
    X_train, X_test, y_train, y_test = train_test_split(
        raw_openml.data, raw_openml.target
    )
    test_data: pd.DataFrame = pd.concat([X_test, y_test], axis=1)
    model: Pipeline = load_pipeline("model.pipe")
    y_pred, y_true = model(test_data)
    print(f"{accuracy_score(y_true, y_pred):.2%}")


if __name__ == "__main__":
    preprocess_train()
    test()
