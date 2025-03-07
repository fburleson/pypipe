import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pypipe.compose import Pipeline
from pypipe.models import ScikitModel
from pypipe.segments.preprocess import TrainTestSplit, MinMaxScale
from pypipe.segments.util import ToPandas, ToNumpy, Concat, Passthrough


def main():
    raw = load_iris()
    model: ScikitModel = ScikitModel(LogisticRegression())
    format_data: Pipeline = Pipeline(
        [
            [ToPandas(raw.feature_names), ToPandas(["species"])],
            Concat(),
        ]
    )
    preprocess: Pipeline = Pipeline(
        [
            TrainTestSplit(
                [
                    "sepal length (cm)",
                    "sepal width (cm)",
                    "petal length (cm)",
                    "petal width (cm)",
                ],
                ["species"],
            ),
            [MinMaxScale(), MinMaxScale(), Passthrough(), Passthrough()],
            [ToNumpy()] * 4,
        ]
    )
    predict: Pipeline = Pipeline(
        [
            [model, Passthrough()],
            [ToPandas()] * 2,
            Concat(),
        ]
    )
    data: pd.DataFrame = format_data((raw.data, raw.target))
    print(data)
    X_train, X_test, y_train, y_test = preprocess(data)
    model.train(X_train, np.squeeze(y_train))
    predictions: np.ndarray = predict((X_test, y_test)).to_numpy().T
    print(f"{accuracy_score(predictions[0], predictions[1]):.2%}")


if __name__ == "__main__":
    main()
