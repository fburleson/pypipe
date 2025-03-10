from typing import Any
from abc import abstractmethod
from sklearn.base import BaseEstimator
from pypipe.compose import Transformer


class Model(Transformer):
    def transform(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def train(self, X, y) -> Any:
        pass

    @abstractmethod
    def forward(self, X) -> Any:
        pass


class ScikitModel(Model):
    def __init__(self, model: BaseEstimator):
        self.model = model

    def train(self, X, y) -> Model:
        self.model.fit(X, y)
        return self

    def forward(self, X):
        return self.model.predict(X)
