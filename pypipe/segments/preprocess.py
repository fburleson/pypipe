import pandas as pd
from sklearn.preprocessing import scale, minmax_scale
from sklearn.model_selection import train_test_split
from pypipe.compose import Transformer
from pypipe.segments.util import Subset


class StandardScale(Transformer):
    def __init__(self, columns: list[str] = None, exclude: bool = False):
        self.columns = columns
        self.exclude = exclude

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.columns is None:
            subset: pd.DataFrame = data[data.columns]
        else:
            subset: pd.DataFrame = Subset(self.columns, self.exclude)(data)
        subset = pd.DataFrame(
            scale(subset, axis=0), columns=subset.columns, index=data.index
        )
        output: pd.DataFrame = data.copy()
        output[subset.columns] = subset[subset.columns]
        return output


class MinMaxScale(Transformer):
    def __init__(self, columns: list[str] = None, exclude: bool = False):
        self.columns = columns
        self.exclude = exclude

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.columns is None:
            subset: pd.DataFrame = data[data.columns]
        else:
            subset: pd.DataFrame = Subset(self.columns, self.exclude)(data)
        subset = pd.DataFrame(
            minmax_scale(subset, axis=0), columns=subset.columns, index=data.index
        )
        output: pd.DataFrame = data.copy()
        output[subset.columns] = subset[subset.columns]
        return output


class TrainTestSplit(Transformer):
    def __init__(
        self,
        X: list[str],
        y: list[str],
        test_size: float = 0.2,
        shuffle: bool = True,
    ):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.shuffle = shuffle

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return train_test_split(
            data[self.X], data[self.y], test_size=self.test_size, shuffle=self.shuffle
        )
