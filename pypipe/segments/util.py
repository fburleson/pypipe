from typing import Any
import numpy as np
import pandas as pd
from pypipe.compose import Transformer


class Subset(Transformer):
    def __init__(self, columns: list[str], exclude: bool = False):
        self.columns = columns
        self.exclude = exclude

    def _transform_numpy(self, data: np.ndarray) -> np.ndarray:
        if self.exclude:
            return np.delete(data, self.columns, axis=1)
        return data[:, self.columns]

    def _transform_pandas(self, data: pd.DataFrame) -> np.ndarray:
        if self.exclude:
            return data.drop(self.columns, axis=1)
        return data[self.columns]

    def transform(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        if isinstance(data, pd.DataFrame):
            return self._transform_pandas(data)
        if isinstance(data, np.ndarray):
            return self._transform_numpy(data)
        raise TypeError(f"{type(data)} is not of type {pd.DataFrame} or {np.ndarray}")


class Split(Transformer):
    def __init__(self, *subsets: list[str], drop_excluded: bool = False):
        self.subsets = subsets
        self.drop_excluded = drop_excluded

    def _transform_numpy(self, data: np.ndarray) -> tuple[np.ndarray]:
        return tuple([data[:, subset] for subset in self.subsets])

    def _transform_pandas(self, data: pd.DataFrame) -> tuple[pd.DataFrame]:
        return tuple([data[subset] for subset in self.subsets])

    def transform(self, data: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        if isinstance(data, pd.DataFrame):
            return self._transform_pandas(data)
        if isinstance(data, np.ndarray):
            return self._transform_numpy(data)
        raise TypeError(f"{type(data)} is not of type {pd.DataFrame} or {np.ndarray}")


class Passthrough(Transformer):
    def transform(self, data) -> Any:
        return data


class ToPandas(Transformer):
    def __init__(self, columns: list[str] = None, index: list[str] = None, dtype=None):
        self.columns = columns
        self.index = index
        self.dtype = dtype

    def transform(self, array: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            array, columns=self.columns, index=self.index, dtype=self.dtype
        )
