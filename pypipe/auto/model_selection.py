import numpy as np
from sklearn.linear_model import (
    LogisticRegression,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from pypipe.models import Model, ScikitModel


def search_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    models: list[Model],
    metric: callable,
    test_size: float = 0.2,
) -> Model:
    costs: np.ndarray = np.empty((0))
    for model in models:
        model.train(X_train, y_train)
        y_pred = model.forward(X_test)
        costs = np.append(costs, [metric(y_test, y_pred)])
    return models[np.argmin(costs)]


def search_classif_model(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    metric: callable,
    test_size: float = 0.2,
) -> Model:
    return search_model(
        X_train,
        X_test,
        y_train,
        y_test,
        metric=metric,
        test_size=test_size,
        models=(
            ScikitModel(LogisticRegression()),
            ScikitModel(SVC()),
            ScikitModel(RandomForestClassifier()),
            ScikitModel(DecisionTreeClassifier()),
        ),
    )
