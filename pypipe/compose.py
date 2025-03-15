from typing import Any, Self
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator


class Transformer(ABC):
    @abstractmethod
    def transform(self, *args, **kwargs) -> Any:
        pass

    def __call__(self, *args, **kwargs) -> Any:
        return self.transform(*args, **kwargs)


class SequenceTransformer(Transformer):
    def __init__(self, *transformers: tuple[callable]):
        self.transformers: list[Transformer] = []
        for segment in transformers:
            if isinstance(segment, tuple):
                self.transformers.append(Parallel(*segment))
            elif isinstance(segment, TransformerMixin):
                self.transformers.append(ScikitTransformer(segment))
            elif isinstance(segment, BaseEstimator):
                self.transformers.append(ScikitModel(segment).train)
            else:
                self.transformers.append(segment)

    @abstractmethod
    def transform(self, *args, **kwargs) -> Any:
        pass

    def __getitem__(self, idx):
        return self.transformers[idx]


class Model:
    def __init__(self, **params):
        self.params = params

    def set_params(self, **params):
        self.params.update(params)

    def get_param(self, param: str) -> Any:
        return self.params[param]

    @abstractmethod
    def train(self) -> Self:
        pass

    @abstractmethod
    def forward(self, X) -> Any:
        pass


class Parallel(SequenceTransformer):
    def transform(self, data) -> tuple[Any]:
        if len(self.transformers) == 1:
            if isinstance(data, tuple):
                return self.transformers[0](*data)
            else:
                return self.transformers[0](data)
        output: list = []
        for i, data_i in enumerate(data):
            output.append(self.transformers[i](data_i))
        return tuple(output)


class Pipeline(SequenceTransformer):
    def transform(self, *args, **kwargs) -> Any:
        out = self.transformers[0](*args, **kwargs)
        for transformer in self.transformers[1:]:
            out = transformer(out)
        return out


class ScikitTransformer(Transformer):
    def __init__(self, transformer: TransformerMixin):
        self._transformer = transformer

    def transform(self, data, *args, **kwargs):
        if isinstance(data, pd.DataFrame):
            self._transformer.set_output(transform="pandas")
        return self._transformer.fit_transform(data, *args, **kwargs)


class ScikitModel(Model):
    def __init__(self, model: BaseEstimator, train: bool = True):
        super().__init__(train=train, params=model.get_params())
        self._model = model

    def set_params(self, **params):
        super().set_params(**params)
        self._model.set_params(**params)

    def train(self, X, y=None) -> Self:
        if isinstance(X, tuple):
            self._model.fit(*X)
        else:
            self._model.fit(X, y)
        return self

    def forward(self, X):
        return self._model.predict(X)
