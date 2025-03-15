import numpy as np
from itertools import product
from sklearn.base import BaseEstimator
from pypipe.compose import Model, ScikitModel


def grid_search_model(
    X_train,
    X_test,
    y_train,
    y_test,
    model: Model | BaseEstimator,
    eval_func: callable,
    minimize: bool,
    **grid,
) -> Model:
    if isinstance(model, BaseEstimator):
        model = ScikitModel(model)
    grid_combis: list = list(product(*grid.values()))
    if minimize:
        best_score = np.inf
    else:
        best_score = -np.inf
    best_params = None
    for params in grid_combis:
        param_dict: dict = dict(zip(grid.keys(), params))
        model.set_params(**param_dict)
        model.train(X_train, y_train)
        y_pred = model.forward(X_test)
        score = eval_func(y_pred, y_test)
        if minimize and score < best_score:
            best_score = score
            best_params = param_dict
        elif score > best_score:
            best_score = score
            best_params = param_dict
    model.set_params(**best_params)
    return model


def auto_select_model(
    X_train,
    X_test,
    y_train,
    y_test,
    eval_func: callable,
    minimize: bool,
    grids: list,
) -> Model:
    if minimize:
        best_score = np.inf
    else:
        best_score = -np.inf
    models: list[Model] = [
        grid_search_model(
            X_train,
            y_train,
            X_test,
            y_test,
            grid[0],
            eval_func,
            minimize=minimize,
            **grid[1],
        )
        for grid in grids
    ]
    for model in models:
        y_pred = model.forward(X_test)
        score = eval_func(y_pred, y_test)
        if minimize and score < best_score:
            best_score = score
            best_model = model
        elif score > best_score:
            best_score = score
            best_model = model
    return best_model
