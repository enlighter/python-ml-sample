import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier
from sklearn.svm import LinearSVC, LinearSVR
from xgboost import XGBRegressor, XGBClassifier
from statsmodels.regression.linear_model import OLS
from lightgbm import LGBMRegressor, LGBMClassifier

from statsmodels.tools.tools import add_constant
from sklearn.metrics import classification_report, cohen_kappa_score
import scipy.stats as stats


def mlr(x_trn, x_val, y_trn, take_intercept):
    # print(x_trn)
    # print(x_val)
    # print(take_intercept)
    x_trn = x_trn.copy()
    x_val = x_val.copy()
    y_trn = y_trn.copy()
    if take_intercept:
        x_trn = add_constant(x_trn, has_constant='add')
        x_val = add_constant(x_val, has_constant='add')

    model = OLS(y_trn, x_trn, hasConstant=take_intercept)
    result = model.fit()

    # print(x_trn)
    # print(x_val)
    y_hat = result.predict(x_val)
    if take_intercept:
        intercept = result.params['const']
    else:
        intercept = 0

    return y_hat, intercept, result.rsquared, result.f_pvalue, result.predict(x_trn)


def mlr_test(x_trn,
             y_trn,
             take_intercept=True,
             significance_level=0.05):
    x_trn = x_trn.copy()
    y_trn = y_trn.copy()
    if take_intercept:
        x_trn = add_constant(x_trn, has_constant='add')
        # debug('ols test x train', x_trn)

    # print(x_trn, y_trn)
    model = OLS(y_trn, x_trn, hasConstant=take_intercept)
    result = model.fit()

    # print(result)
    # print('MLR Summary', result.summary())

    # print('model f_score:', result.f_pvalue)
    if result.f_pvalue >= significance_level:
        # debug('discarding model')
        return [], False, result.f_pvalue

    t_vals = (2 * (1 - stats.t.cdf(result.tvalues, df=result.df_resid)))
    res_df = pd.DataFrame(data={'t_vals': t_vals, 'significant': t_vals < 0.05},
                          index=x_trn.columns)
    # print(res_df)

    # debug('Significant Features for MLR', res_df)
    # print(res_df)

    mlr_cols = res_df.index[res_df.significant].tolist()
    # print(mlr_cols)

    has_const = 'const' in mlr_cols
    mlr_cols = [x for x in mlr_cols if x != 'const']

    return mlr_cols, has_const, result.f_pvalue


def rf_regressor(x_trn: pd.DataFrame,
                 y_trn: np.ndarray,
                 x_val: pd.DataFrame,
                 y_val: np.ndarray) -> tuple:
    x_trn, x_val = x_trn.copy(), x_val.copy()
    y_trn, y_val = y_trn.copy(), y_val.copy()
    model = RandomForestRegressor(n_estimators=400, min_samples_leaf=3,
                                  n_jobs=-1, random_state=7)
    _ = model.fit(x_trn, y_trn)

    training_score = model.score(x_trn, y_trn)
    validation_score = model.score(x_val, y_val)

    return model, training_score, validation_score


def et_regressor(x_trn: pd.DataFrame,
                 y_trn: np.ndarray,
                 x_val: pd.DataFrame,
                 y_val: np.ndarray) -> tuple:
    x_trn, x_val = x_trn.copy(), x_val.copy()
    y_trn, y_val = y_trn.copy(), y_val.copy()
    model = ExtraTreesRegressor(n_estimators=400, min_samples_leaf=3,
                                n_jobs=-1, random_state=7)
    _ = model.fit(x_trn, y_trn)

    training_score = model.score(x_trn, y_trn)
    validation_score = model.score(x_val, y_val)

    return model, training_score, validation_score


def xgb_regressor(x_trn: pd.DataFrame,
                  y_trn: np.ndarray,
                  x_val: pd.DataFrame,
                  y_val: np.ndarray) -> tuple:
    x_trn, x_val = x_trn.copy(), x_val.copy()
    y_trn, y_val = y_trn.copy(), y_val.copy()
    model = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=3,
                         n_jobs=-1, random_state=7)
    _ = model.fit(x_trn, y_trn)

    training_score = model.score(x_trn, y_trn)
    validation_score = model.score(x_val, y_val)

    return model, training_score, validation_score


def lgbm_regressor(x_trn: pd.DataFrame,
                   y_trn: np.ndarray,
                   x_val: pd.DataFrame,
                   y_val: np.ndarray) -> tuple:
    x_trn, x_val = x_trn.copy(), x_val.copy()
    y_trn, y_val = y_trn.copy(), y_val.copy()

    model = LGBMRegressor(boosting_type='gbdt', objective='regression', metric='mse',
                          n_estimators=400, learning_rate=0.05, min_child_samples=3,
                          num_iterations=700, n_jobs=-1, random_state=7)
    _ = model.fit(x_trn, y_trn)

    training_score = model.score(x_trn, y_trn)
    validation_score = model.score(x_val, y_val)

    return model, training_score, validation_score


def lsvm_regressor(x_trn: pd.DataFrame,
                   y_trn: np.ndarray,
                   x_val: pd.DataFrame,
                   y_val: np.ndarray) -> tuple:
    x_trn, x_val = x_trn.copy(), x_val.copy()
    y_trn, y_val = y_trn.copy(), y_val.copy()
    model = LinearSVR(max_iter=400, C=0.05,
                      random_state=7)
    _ = model.fit(x_trn, y_trn)

    training_score = model.score(x_trn, y_trn)
    validation_score = model.score(x_val, y_val)

    return model, training_score, validation_score


def rf_classifier(x_trn: pd.DataFrame,
                  y_trn: np.ndarray,
                  x_val: pd.DataFrame,
                  y_val: np.ndarray) -> tuple:
    x_trn, x_val = x_trn.copy(), x_val.copy()
    y_trn, y_val = y_trn.copy(), y_val.copy()

    model = RandomForestClassifier(n_estimators=400, min_samples_leaf=16,
                                   class_weight='balanced',
                                   n_jobs=-1, random_state=7)
    _ = model.fit(x_trn, y_trn)

    training_score = model.score(x_trn, y_trn)
    validation_score = model.score(x_val, y_val)

    clf_report = classification_report(y_val, model.predict(x_val))
    ck_score = cohen_kappa_score(y_val, model.predict(x_val))

    return model, training_score, validation_score, clf_report, ck_score


def et_classifier(x_trn: pd.DataFrame,
                  y_trn: np.ndarray,
                  x_val: pd.DataFrame,
                  y_val: np.ndarray) -> tuple:
    x_trn, x_val = x_trn.copy(), x_val.copy()
    y_trn, y_val = y_trn.copy(), y_val.copy()
    model = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=16,
                                 class_weight='balanced',
                                 n_jobs=-1, random_state=7)
    _ = model.fit(x_trn, y_trn)

    training_score = model.score(x_trn, y_trn)
    validation_score = model.score(x_val, y_val)

    clf_report = classification_report(y_val, model.predict(x_val))
    ck_score = cohen_kappa_score(y_val, model.predict(x_val))

    return model, training_score, validation_score, clf_report, ck_score


def xgb_classifier(x_trn: pd.DataFrame,
                   y_trn: np.ndarray,
                   x_val: pd.DataFrame,
                   y_val: np.ndarray,
                   scale_pos_weight=1) -> tuple:
    # scale_pos_weight is num(False)/num(True) in training data
    x_trn, x_val = x_trn.copy(), x_val.copy()
    y_trn, y_val = y_trn.copy(), y_val.copy()
    model = XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=3,
                          scale_pos_weight=scale_pos_weight,
                          n_jobs=-1, random_state=7)
    _ = model.fit(x_trn, y_trn)

    training_score = model.score(x_trn, y_trn)
    validation_score = model.score(x_val, y_val)

    clf_report = classification_report(y_val, model.predict(x_val))
    ck_score = cohen_kappa_score(y_val, model.predict(x_val))

    return model, training_score, validation_score, clf_report, ck_score


def lgbm_classifier(x_trn: pd.DataFrame,
                    y_trn: np.ndarray,
                    x_val: pd.DataFrame,
                    y_val: np.ndarray) -> tuple:
    x_trn, x_val = x_trn.copy(), x_val.copy()
    y_trn, y_val = y_trn.copy(), y_val.copy()

    model = LGBMClassifier(boosting_type='gbdt', objective='binary', metric='binary_logloss',
                           n_estimators=400, learning_rate=0.05, min_child_samples=16,
                           is_unbalance=True,
                           num_iterations=700, n_jobs=-1, random_state=7)
    _ = model.fit(x_trn, y_trn)

    training_score = model.score(x_trn, y_trn)
    validation_score = model.score(x_val, y_val)

    clf_report = classification_report(y_val, model.predict(x_val))
    ck_score = cohen_kappa_score(y_val, model.predict(x_val))

    return model, training_score, validation_score, clf_report, ck_score


def lsvm_classifier(x_trn: pd.DataFrame,
                    y_trn: np.ndarray,
                    x_val: pd.DataFrame,
                    y_val: np.ndarray) -> tuple:
    x_trn, x_val = x_trn.copy(), x_val.copy()
    y_trn, y_val = y_trn.copy(), y_val.copy()
    model = LinearSVC(max_iter=400, C=0.05,
                      class_weight='balanced',
                      random_state=7)
    _ = model.fit(x_trn, y_trn)

    training_score = model.score(x_trn, y_trn)
    validation_score = model.score(x_val, y_val)

    clf_report = classification_report(y_val, model.predict(x_val))
    ck_score = cohen_kappa_score(y_val, model.predict(x_val))

    return model, training_score, validation_score, clf_report, ck_score
