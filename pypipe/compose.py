from typing import Any
from abc import ABC, abstractmethod


class Transformer(ABC):
    @abstractmethod
    def transform(self, data) -> Any:
        pass

    def __call__(self, *args, **kwargs) -> Any:
        return self.transform(*args, **kwargs)


class Pipeline:
    def __init__(self, transformers: list[Transformer]):
        self.transformers = transformers

    def transform(self, *args, **kwargs) -> Any:
        out = self.transformers[0](*args, **kwargs)
        for transformer in self.transformers[1:]:
            out = transformer(out)
        return out

    def __call__(self, *args, **kwargs) -> Any:
        return self.transform(*args, **kwargs)
