from typing import Any
from abc import abstractmethod
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
