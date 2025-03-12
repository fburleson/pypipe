from typing import Any
from typing import Self
from abc import abstractmethod
from sklearn.base import BaseEstimator
from pypipe.compose import Transformer


class Model(Transformer):
    def __init__(self, train: bool = True):
        self.train = train

    def transform(self, *args, **kwargs) -> Any:
        if self.train:
            return self.train(*args, **kwargs)
        return self.forward(*args, **kwargs)

    @abstractmethod
    def train(self, X, y) -> Self:
        pass

    @abstractmethod
    def forward(self, X) -> Any:
        pass


class ScikitModel(Model):
    def __init__(self, model: BaseEstimator):
        self.model = model

    def train(self, X, y) -> Self:
        self.model.fit(X, y)
        return self

    def forward(self, X):
        return self.model.predict(X)

    def get(self) -> BaseEstimator:
        return self.model
