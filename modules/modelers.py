# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 19:03:48 2018

@author: enlighter
"""
import pandas as pd
import numpy as np

import traceback
from typing import Optional

from modules.models import rf_regressor, et_regressor, xgb_regressor, lgbm_regressor, lsvm_regressor
from modules.models import rf_classifier, et_classifier, xgb_classifier, lgbm_classifier, lsvm_classifier


# application specific scikit classification modeler
def high_level_rf_clf_modeler(x_train: pd.DataFrame,
                              y_train_df: pd.Series,
                              x_test: pd.DataFrame,
                              y_test_df: pd.Series) -> tuple:
    y_train: pd.Series = y_train_df.astype(bool)
    y_train: np.ndarray = y_train.values.reshape(1, -1)[0]
    y_test: pd.Series = y_test_df.astype(bool)
    y_test: np.ndarray = y_test.values.reshape(1, -1)[0]

    clf_train_true_pct = sum(y_train) / len(y_train)
    clf_test_true_pct = sum(y_test) / len(y_test)

    model_clf, training_score, acc_clf, \
        clf_report, ck_score = et_classifier(x_train, y_train, x_test, y_test)

    return model_clf, clf_train_true_pct, clf_test_true_pct, \
        training_score, acc_clf, clf_report, ck_score


# application specific scikit regression modeler
def high_level_rf_reg_modeler(x_train: pd.DataFrame,
                              y_train_df: pd.Series,
                              x_test: pd.DataFrame,
                              y_test_df: pd.Series) -> tuple:
    y_train: np.ndarray = y_train_df.values.reshape(1, -1)[0]
    y_test: np.ndarray = y_test_df.values.reshape(1, -1)[0]

    model_rf, trn_score_rf, acc_rf = et_regressor(x_train, y_train,
                                                  x_test, y_test)

    forecast_type = 'et_regression'

    return model_rf, trn_score_rf, acc_rf
