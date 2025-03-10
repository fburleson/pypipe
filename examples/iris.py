import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from pypipe.models import LogisticRegression
from pypipe.compose import Pipeline
from pypipe.models import ScikitModel
from pypipe.segments.preprocess import TrainTestSplit, MinMaxScale
from pypipe.segments.util import ToPandas, ToNumpy, Concat, Passthrough


def main():
    raw = load_iris()
    model: ScikitModel = LogisticRegression()
    preprocess: Pipeline = Pipeline(
        [
            [ToPandas(raw.feature_names), ToPandas(["species"])],
            Concat(),
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
    X_train, X_test, y_train, y_test = preprocess((raw.data, raw.target))
    predict: Pipeline = Pipeline(
        [
            [model.train(X_train, np.squeeze(y_train)), Passthrough()],
            [ToPandas()] * 2,
            Concat(),
        ]
    )
    predictions: np.ndarray = predict((X_test, y_test)).to_numpy().T
    print(f"{accuracy_score(predictions[0], predictions[1]):.2%}")


if __name__ == "__main__":
    main()
