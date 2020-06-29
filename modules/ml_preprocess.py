import pandas as pd
import numpy as np
from pandas.api.types import is_string_dtype, is_numeric_dtype

from typing import Any, Optional


# Convert strings into categorical for training data
def _categorize_train(x_train_df: pd.DataFrame,
                      y_train_series: pd.Series = pd.Series(),
                      x_skip_cols: Optional[list] = None) -> dict:
    x_train_df: pd.DataFrame = x_train_df.copy()
    y_train_series: pd.Series = y_train_series.copy()
    if x_skip_cols is None:
        x_skip_cols: list = []

    for col, series in x_train_df.items():
        if col not in x_skip_cols:
            if is_string_dtype(series):
                x_train_df[col] = series.astype('category').cat.as_ordered()

    categorize_train_dict = {'x_train_df': x_train_df,
                             'y_train_series': y_train_series,
                             'x_skip_cols': x_skip_cols}

    return categorize_train_dict


# Convert strings into categorical for validation data
def _categorize_val(categorize_train_dict: dict,
                    x_val_df: pd.DataFrame,
                    y_val_series: Optional[pd.Series] = None) -> tuple:
    x_val_df: pd.DataFrame = x_val_df.copy()
    if y_val_series is None:
        y_val_series: pd.Series = pd.Series()
    else:
        y_val_series: pd.Series = y_val_series.copy()

    x_train_df: pd.DataFrame = categorize_train_dict['x_train_df']
    y_train_series: pd.Series = categorize_train_dict['y_train_series']
    x_skip_cols: list = categorize_train_dict['x_skip_cols']

    for col, series in x_val_df.items():
        if (col in x_train_df.columns) and (col not in x_skip_cols):
            if x_train_df[col].dtype.name == 'category':
                x_val_df[col] = pd.Categorical(series,
                                               categories=x_train_df[col].cat.categories,
                                               ordered=True)

    return x_val_df, y_val_series


# Common non-text preprocessing for training data
def _data_preproc_train(x_train_df: pd.DataFrame,
                        y_train_series: pd.Series = pd.Series(),
                        x_skip_cols: Optional[list] = None,
                        max_cat_count: int = 0) -> dict:
    x_train_df: pd.DataFrame = x_train_df.copy()
    y_train_series: pd.Series = y_train_series.copy()
    if x_skip_cols is None:
        x_skip_cols: list = []

    x_train_df.drop(x_skip_cols, axis=1, inplace=True)

    for col, series in x_train_df.items():
        if (not is_numeric_dtype(series)
                and series.nunique() > max_cat_count):
            x_train_df[col] = series.cat.codes + 1
    x_train_df = pd.get_dummies(x_train_df, dummy_na=True)

    if len(y_train_series) != 0:
        if not is_numeric_dtype(y_train_series):
            y_train_series = y_train_series.cat.codes

    data_preproc_train_dict = {'x_train_df': x_train_df,
                               'y_train_series': y_train_series,
                               'x_skip_cols': x_skip_cols,
                               'max_n_cat': max_cat_count}

    return data_preproc_train_dict


# Ensure validation data has the same columns as the training data
def _handle_missing_cols(x_train_df: pd.DataFrame,
                         x_val_df: pd.DataFrame,
                         fill_value: Any = np.nan) -> pd.DataFrame:
    missing_cols: set = set(x_train_df.columns) - set(x_val_df.columns)
    for col in missing_cols:
        x_val_df[col] = fill_value
    return x_val_df[x_train_df.columns]


# Common non-text preprocessing for validation data
def _data_preproc_val(data_preproc_train_dict: dict,
                      x_val_df: pd.DataFrame,
                      y_val_series: Optional[pd.Series] = None) -> list:
    x_val_df: pd.DataFrame = x_val_df.copy()
    if y_val_series is None:
        y_val_series: pd.Series = pd.Series()
    else:
        y_val_series: pd.Series = y_val_series.copy()

    x_train_df: pd.DataFrame = data_preproc_train_dict['x_train_df']
    y_train_series: pd.Series = data_preproc_train_dict['y_train_series']
    x_skip_cols: list = data_preproc_train_dict['x_skip_cols']
    max_cat_count: int = data_preproc_train_dict['max_n_cat']

    x_val_df.drop(x_skip_cols, axis=1, inplace=True)

    for col, series in x_val_df.items():
        if (not is_numeric_dtype(series)
                and series.nunique() > max_cat_count):
            x_val_df[col] = series.cat.codes + 1
    x_val_df = pd.get_dummies(x_val_df, dummy_na=True)
    x_val_df = _handle_missing_cols(x_train_df, x_val_df)
    x_val_df.fillna(-1, inplace=True)

    if len(y_val_series) != 0:
        if not is_numeric_dtype(y_val_series):
            y_val_series = y_val_series.cat.codes
            y_val_series.fillna(-1, inplace=True)

    return [x_val_df, y_val_series]
