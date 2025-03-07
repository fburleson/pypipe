import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.preprocessing import scale, minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from pypipe.compose import Transformer


class Subset(Transformer):
    def __init__(self, columns: list[str], exclude: bool = False):
        self.columns = columns
        self.exclude = exclude

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.exclude:
            return data.drop(self.columns, axis=1)
        return data[self.columns]


class Impute(Transformer):
    def __init__(self, columns: list[str] = None, exclude: bool = False):
        self.columns = columns
        self.exclude = exclude

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.columns is None:
            subset: pd.DataFrame = data[data.columns]
        else:
            subset: pd.DataFrame = Subset(self.columns, self.exclude)(data)
        num_columns = subset.select_dtypes(include="number").columns
        cat_columns = subset.select_dtypes(include="object").columns
        transformer: ColumnTransformer = ColumnTransformer(
            transformers=[
                ("mean", SimpleImputer(strategy="mean"), num_columns),
                ("mode", SimpleImputer(strategy="most_frequent"), cat_columns),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        return pd.DataFrame(
            transformer.fit_transform(data.replace([None], np.nan)),
            columns=transformer.get_feature_names_out(),
            index=data.index,
        ).astype(data.dtypes)


class Encode(Transformer):
    def __init__(self, columns: list[str] = None, exclude: bool = False):
        self.columns = columns
        self.exclude = exclude

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.columns is None:
            subset: pd.DataFrame = data[data.columns]
        else:
            subset: pd.DataFrame = Subset(self.columns, self.exclude)(data)
        multiclass = subset.loc[:, subset.nunique() > 2].columns
        binary = subset.loc[:, subset.nunique() <= 2].columns
        transformer: ColumnTransformer = ColumnTransformer(
            transformers=[
                ("onehot", OneHotEncoder(), multiclass),
                ("label", OrdinalEncoder(), binary),
            ],
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        return pd.DataFrame(
            transformer.fit_transform(data),
            columns=transformer.get_feature_names_out(),
            index=data.index,
        )


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
            scale(subset, axis=0), columns=self.columns, index=data.index
        )
        output: pd.DataFrame = data.copy()
        output[self.columns] = subset[self.columns]
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
            minmax_scale(subset, axis=0), columns=self.columns, index=data.index
        )
        output: pd.DataFrame = data.copy()
        output[self.columns] = subset[self.columns]
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
