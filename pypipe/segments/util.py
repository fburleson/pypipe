from itertools import chain
from typing import Any
import pandas as pd
from pypipe.compose import Transformer


class Passthrough(Transformer):
    def transform(self, data) -> Any:
        return data


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
