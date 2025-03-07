from typing import Any
from abc import abstractmethod
from sklearn.linear_model import LogisticRegression
from pypipe.compose import Transformer


class Model(Transformer):
    def transform(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def forward(self, X) -> Any:
        pass


class LogisticClassifier(Model):
    def __init__(self):
        self.model = LogisticRegression(solver="lbfgs")

    def train(self, X, y):
        self.model.fit(X, y)

    def forward(self, X):
        return self.model.predict(X)
