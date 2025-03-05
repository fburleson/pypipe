import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
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
    def __init__(self, columns: list[str], exclude: bool = False):
        self.columns = columns
        self.exclude = exclude

    def transform(
        self,
        data: pd.DataFrame,
        num_strat: str = "mean",
        cat_strat: str = "most_frequent",
    ) -> pd.DataFrame:
        subset: pd.DataFrame = Subset(self.columns, self.exclude)(data)
        num_columns = subset.select_dtypes(include="number").columns
        cat_columns = subset.select_dtypes(include="object").columns
        transformer: ColumnTransformer = ColumnTransformer(
            transformers=[
                ("mean", SimpleImputer(strategy=num_strat), num_columns),
                ("mode", SimpleImputer(strategy=cat_strat), cat_columns),
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
    def __init__(self, columns: list[str], exclude: bool = False):
        self.columns = columns
        self.exclude = exclude

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
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
