from typing import Any
from abc import ABC, abstractmethod


class Transformer(ABC):
    @abstractmethod
    def transform(self, *args, **kwargs) -> Any:
        pass

    def __call__(self, *args, **kwargs) -> Any:
        return self.transform(*args, **kwargs)


class Parallel(Transformer):
    def __init__(self, transformers: list[Transformer]):
        self.transformers: list[Transformer] = []
        for segment in transformers:
            if isinstance(segment, list):
                self.transformers.append(Parallel(segment))
            else:
                self.transformers.append(segment)

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


class Pipeline(Transformer):
    def __init__(self, *pipeline: list[Transformer]):
        self.transformers: list[Transformer] = []
        for segment in pipeline:
            if isinstance(segment, list):
                self.transformers.append(Parallel(segment))
            else:
                self.transformers.append(segment)

    def transform(self, *args, **kwargs) -> Any:
        out = self.transformers[0](*args, **kwargs)
        for transformer in self.transformers[1:]:
            out = transformer(out)
        return out
