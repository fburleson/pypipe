from itertools import chain
from typing import Any
import numpy as np
import pandas as pd
from pypipe.compose import Transformer


class Subset(Transformer):
    def __init__(self, columns: list[str], exclude: bool = False):
        self.columns = columns
        self.exclude = exclude

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.exclude:
            return data.drop(self.columns, axis=1)
        return data[self.columns]


class Split(Transformer):
    def __init__(self, *splits: list[str], drop_excluded: bool = True):
        self.splits = splits
        self.drop_excluded = drop_excluded

    def transform(self, data: pd.DataFrame) -> tuple[pd.DataFrame]:
        output: list[pd.DataFrame] = []
        for split in self.splits:
            output.append(data[split])
        if not self.drop_excluded:
            all_columns: set = set(data.columns)
            split_columns: set = set(chain(*self.splits))
            output.append(data[list(all_columns - split_columns)])
        return tuple(output)


class Concat(Transformer):
    def transform(self, *data: tuple[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(list(*data), axis=1)


class Passthrough(Transformer):
    def transform(self, data) -> Any:
        return data


class ToNumpy(Transformer):
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        return data.to_numpy()


class ToPandas(Transformer):
    def __init__(self, columns: list[str] = None, index: list[str] = None, dtype=None):
        self.columns = columns
        self.index = index
        self.dtype = dtype

    def transform(self, array: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            array, columns=self.columns, index=self.index, dtype=self.dtype
        )
