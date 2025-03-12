import types
from typing import Any
from functools import partial
from abc import ABC, abstractmethod
import pandas as pd
from sklearn.base import TransformerMixin


class Transformer(ABC):
    @abstractmethod
    def transform(self, *args, **kwargs) -> Any:
        pass

    def __call__(self, *args, **kwargs) -> Any:
        return self.transform(*args, **kwargs)


class SequenceTransformer(Transformer):
    def __init__(self, *transformers: list[Transformer]):
        self.transformers: list[Transformer] = []
        for segment in transformers:
            if isinstance(segment, list):
                self.transformers.append(Parallel(segment))
            elif isinstance(segment, TransformerMixin):
                self.transformers.append(ScikitTransformer(segment))
            else:
                self.transformers.append(segment)

    @abstractmethod
    def transform(self, *args, **kwargs) -> Any:
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
        self.transformer = transformer

    def transform(self, data, *args, **kwargs):
        if isinstance(data, pd.DataFrame):
            self.transformer.set_output(transform="pandas")
        return self.transformer.fit_transform(data, *args, **kwargs)
