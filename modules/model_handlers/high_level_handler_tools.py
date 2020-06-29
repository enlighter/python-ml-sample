import pandas as pd
import numpy as np
from joblib import dump, load

from typing import Optional
import os

from modules.ml_preprocess import _categorize_train, _categorize_val
from modules.ml_preprocess import _data_preproc_train, _data_preproc_val
from modules.modelers import high_level_rf_reg_modeler, high_level_rf_clf_modeler
from modules.aggregation import high_level_post_model_aggregator
from config import clf_drop_cols
from config import use_drop_reg_cols_instead_of_keep, reg_drop_cols
from config import regression_mode, reg_keep_cols_1, reg_keep_cols_2
from config import keep_in_reg_result
from config import high_std_thr, low_density_thr, low_vp_thr


# modelling flow handler per product for one date; for either annual, monthly or weekly modelling flows
# depending on 'model_period' parameter. Performs corresponding operations of train-test split,
# modelling and post model adjustments, buffering and aggregations.
def single_date_split_model_forecast(product_data: pd.DataFrame,
                                     inflated_demand_marker_data: pd.DataFrame,
                                     present_agent_map_data: pd.DataFrame,
                                     past_agent_map_data: pd.DataFrame,
                                     pred_date: pd.Timestamp,
                                     remove_first_month_from_training: bool,
                                     y_col: str,
                                     model_period: str,
                                     hyperparams_dict: dict,
                                     product_code: str,
                                     so_code: str,
                                     model_pickle_path: str,
                                     run_log_path: str,
                                     int_result_dump_path: str) -> Optional[tuple]:
    period_column_switcher: dict = {
        'annual': 'year',
        'monthly': 'month',
        'weekly': 'week'}
    period_column: str = period_column_switcher[model_period]
    period_designator: int = eval('pred_date.{}'.format(period_column))
    # print(model_period)

    # split data into train & test segments
    _train_test_split = train_test_split(
        product_data, pred_date, remove_first_month_from_training, model_period)
    if _train_test_split is not None:
        df_train, df_test, df_leftover, extra_test_features_introduced = _train_test_split
    else:
        return None

    # check for condition for re-training period model
    period_data: pd.DataFrame = product_data.loc[product_data['year'] == pred_date.year]
    period_data = period_data.loc[period_data[period_column] == period_designator]
    dates_in_period_sorted: pd.DatetimeIndex = period_data.sort_index().index.unique()
    is_pred_date_first_in_period: bool = (dates_in_period_sorted[0] == pred_date)

    # model data & get forecast
    df_test_after_clf, forecast_result = model(
        df_train,
        df_test,
        y_col,
        is_pred_date_first_in_period,
        extra_test_features_introduced,
        model_period,
        product_code,
        so_code,
        model_pickle_path,
        run_log_path
    )

    # post model forecast data prep
    forecast_result: pd.DataFrame = forecast_result[
        keep_in_reg_result + extra_test_features_introduced]
    forecast_result = forecast_result.rename(columns={'invoice_date': 'visit_date'})
    forecast_result = forecast_result.merge(
        present_agent_map_data,
        how='left',
        on=['visit_date', 'pos_code'])
    forecast_result['orig_pred_load'] = forecast_result['predicted_loading']
    # print(forecast_result.head())
    train_data_with_agent = df_train.reset_index()
    train_data_with_agent = train_data_with_agent.merge(past_agent_map_data,
                                                        how='left',
                                                        on=['invoice_date', 'pos_code'])
    train_data_with_agent.rename(columns={'invoice_date': 'visit_date'}, inplace=True)
    # print(train_data_with_agent.head())
    past_data_not_in_train_with_agent = df_leftover.reset_index()
    past_data_not_in_train_with_agent = past_data_not_in_train_with_agent.merge(
        past_agent_map_data,
        how='left',
        on=['invoice_date', 'pos_code'])
    past_data_not_in_train_with_agent.rename(columns={'invoice_date': 'visit_date'}, inplace=True)
    # print(past_data_not_in_train_with_agent)

    # forecast aggregation & buffer addition
    forecast_df: pd.DataFrame = forecast_result.groupby(
        'agent_code').progress_apply(high_level_post_model_operations,
                                     train_data_with_agent=train_data_with_agent,
                                     past_test_data_with_agent=past_data_not_in_train_with_agent,
                                     inflated_demand_marker_data=inflated_demand_marker_data,
                                     model_type=model_period,
                                     product_code=product_code,
                                     hyperparams_dict=hyperparams_dict,
                                     int_result_dump_path=int_result_dump_path,
                                     run_log_path=run_log_path)
    forecast_df = forecast_df.drop(columns=['agent_code'])

    return forecast_result, forecast_df


# splits the dataset into train and test sets according to 'model_period' parameter (annual, monthly or weekly).
# It also returns an overflow set containing data not belonging to either train or test sets.
def train_test_split(product_data: pd.DataFrame,
                     pred_date: pd.Timestamp,
                     remove_first_month_from_training: bool,
                     model_period: Optional[str] = None) -> Optional[tuple]:
    first_month_in_data = product_data['month'].min()
    first_year_in_data = product_data['year'].min()

    if model_period == 'annual':
        df_train: pd.DataFrame = product_data.loc[product_data['year'] < pred_date.year].copy()
    elif model_period == 'monthly':
        df_train: pd.DataFrame = product_data.loc[product_data.index
                                                  < pd.Timestamp(year=pred_date.year,
                                                                 month=pred_date.month,
                                                                 day=1)].copy()
    elif model_period == 'weekly':
        df_train: pd.DataFrame = product_data.loc[(product_data['year'] < pred_date.year)
                                                  | (product_data['week'] < pred_date.weekofyear)].copy()
    else:
        df_train: pd.DataFrame = product_data.loc[product_data.index < pred_date].copy()

    if remove_first_month_from_training:
        df_train = df_train.loc[
            ~((df_train['month'] == first_month_in_data)
              & (df_train['year'] == first_year_in_data))].copy()

    df_test: pd.DataFrame = product_data.loc[
        product_data.index == pred_date]
    df_test = df_test.loc[df_test['isVisitPlan']].copy()
    if len(df_test) == 0:
        return None
    problematic_pos, df_test, extra_test_features_introduced = \
        _flag_problematic_pos(df_train, df_test.reset_index())
    df_test = df_test.set_index('invoice_date')

    df_leftover = product_data.loc[(product_data.index > df_train.index.max())
                                   & (product_data.index < df_test.index.min())]

    return df_train, df_test, df_leftover, extra_test_features_introduced


# transforms the features to a format compatible with sklearn ml models
def prepare_features_for_sklearn_modelling(x_train_df: pd.DataFrame,
                                           x_test_df: pd.DataFrame) -> tuple:
    categorize_train_dict = _categorize_train(x_train_df)
    x_train = categorize_train_dict['x_train_df']

    data_preproc_train_dict = _data_preproc_train(x_train, max_cat_count=0)
    x_train = data_preproc_train_dict['x_train_df']

    x_test, _ = _categorize_val(categorize_train_dict, x_test_df)
    x_test, _ = _data_preproc_val(data_preproc_train_dict, x_test)
    # for cases where category variable seen for first time in x_test
    x_test = x_test.fillna(0)

    return x_train, x_test


# implements classification modelling at the beginning of modelling section
# to train classifier per product and predict whether a shipment will happen
# for a particular pos for said date and product.
def nonzero_shipment_classification(df_train_clf: pd.DataFrame,
                                    df_test_clf: pd.DataFrame,
                                    y_col: str,
                                    clf_cols_to_remove: list,
                                    extra_test_features_introduced: list,
                                    product_code: str,
                                    rerun_model: bool,
                                    model_pickle_path: str,
                                    run_log_path: str) -> pd.DataFrame:
    clf_train_features_to_remove = clf_cols_to_remove
    clf_train_features_to_remove.append(y_col)
    # print(clf_train_features_to_remove)
    y_train_clf: pd.Series = df_train_clf[y_col]
    x_train_clf: pd.DataFrame = df_train_clf.drop(columns=clf_train_features_to_remove)
    # print('no. of features:', glen(x_train.columns))

    clf_test_features_to_remove = clf_train_features_to_remove.copy()
    clf_test_features_to_remove.extend(extra_test_features_introduced)
    # print(clf_test_features_to_remove)
    y_test_clf: pd.Series = df_test_clf[y_col]
    x_test_clf: pd.DataFrame = df_test_clf.drop(columns=clf_test_features_to_remove)
    # print(x_test_clf.columns.tolist())

    x_train_clf_proc, x_test_clf_proc = \
        prepare_features_for_sklearn_modelling(x_train_clf, x_test_clf)

    if rerun_model:
        model_clf, clf_trn_true_pct, clf_test_true_pct, \
            clf_trn_score, clf_test_score, clf_report, clf_ck_score = \
            high_level_rf_clf_modeler(x_train_clf_proc,
                                      y_train_clf,
                                      x_test_clf_proc,
                                      y_test_clf)
        dump(model_clf, model_pickle_path)
        with open(run_log_path, 'a') as model_log:
            model_log.write(product_code + '\n')
            model_log.write('clf train true percent: ' + str(clf_trn_true_pct) + '\n')
            model_log.write('clf test true percent: ' + str(clf_test_true_pct) + '\n')
            model_log.write('clf train score: ' + str(clf_trn_score) + '\n')
            model_log.write('clf test score: ' + str(clf_test_score) + '\n')
            model_log.write('clf report:\n' + str(clf_report) + '\n')
            model_log.write('clf ck score: ' + str(clf_ck_score) + '\n')
    else:
        model_clf = load(model_pickle_path)

    is_nonzero_test: np.ndarray = model_clf.predict(x_test_clf_proc)
    assert len(is_nonzero_test) == len(df_test_clf)
    df_test_clf.loc[:, 'pred_is_nonzero_shipments'] = is_nonzero_test
    # print(df_test_clf.head())

    return df_test_clf


# implements regression modelling to come after classification step in the modelling section.
# This method drops features marked as not required and trains a single regressor model per product.
# If classifier predicts shipment will occur for a POS for said product and date;
# this regressor predicts the shipment amount for all such POSes.
def drop_cols_regression(df_train_reg: pd.DataFrame,
                         df_test_reg: pd.DataFrame,
                         y_col: str,
                         extra_test_features_introduced: list,
                         rerun_model: bool,
                         model_pickle_path: str) -> np.ndarray:
    reg_train_features_to_remove = reg_drop_cols.copy()
    reg_train_features_to_remove.append(y_col)
    x_train_reg: pd.DataFrame = df_train_reg.drop(
        columns=reg_train_features_to_remove)
    # print(x_train_reg.columns)
    y_train_reg: pd.Series = df_train_reg[y_col]

    reg_test_features_to_remove = reg_train_features_to_remove.copy()
    reg_test_features_to_remove.extend(extra_test_features_introduced)
    reg_test_features_to_remove.append('pred_is_nonzero_shipments')
    x_test_reg: pd.DataFrame = df_test_reg.drop(
        columns=reg_test_features_to_remove)
    # print(x_test_reg.columns)
    y_test_reg: pd.Series = df_test_reg[y_col]

    x_train_reg_proc, x_test_reg_proc = \
        prepare_features_for_sklearn_modelling(x_train_reg, x_test_reg)

    if rerun_model:
        model_reg, x_test_reg_proc, trn_score, val_score = \
            high_level_rf_reg_modeler(x_train_reg_proc,
                                      y_train_reg,
                                      x_test_reg_proc,
                                      y_test_reg)
        dump(model_reg, model_pickle_path)
    else:
        model_reg = load(model_pickle_path)

    y_pred: np.ndarray = model_reg.predict(x_test_reg_proc)

    return y_pred


# implements regression modelling to come after classification step in the modelling section.
# This method keeps features marked as required and trains 2 regressor models per product,
# one general regressor and one regressor geared towards special days and selectively ensembles the results.
# If classifier predicts shipment will occur for a POS for said product and date;
# this regressor predicts the shipment amount for all such POSes.
def multi_model_keep_cols_regression(df_train_reg: pd.DataFrame,
                                     df_test_reg: pd.DataFrame,
                                     y_col: str,
                                     rerun_model: bool,
                                     model_pickle_path_1: str,
                                     model_pickle_path_2: str) -> np.ndarray:
    if regression_mode != 'model_2':
        x_train_reg_1: pd.DataFrame = df_train_reg[reg_keep_cols_1]
        # print(x_train_reg_1.columns.tolist())
        y_train_reg_1: pd.Series = df_train_reg[y_col]

        x_test_reg_1: pd.DataFrame = df_test_reg[reg_keep_cols_1]
        # print(x_test_reg_1.columns.tolist())
        y_test_reg_1: pd.Series = df_test_reg[y_col]

        x_train_reg_proc_1, x_test_reg_proc_1 = \
            prepare_features_for_sklearn_modelling(x_train_reg_1, x_test_reg_1)

        if rerun_model:
            model_reg_1, trn_score_1, val_score_1 = \
                high_level_rf_reg_modeler(x_train_reg_proc_1,
                                          y_train_reg_1,
                                          x_test_reg_proc_1,
                                          y_test_reg_1)
            dump(model_reg_1, model_pickle_path_1)
        else:
            model_reg_1 = load(model_pickle_path_1)

        y_pred_1: np.ndarray = model_reg_1.predict(x_test_reg_proc_1)
        forecast_1: pd.DataFrame = df_test_reg.reset_index()[['invoice_date', 'pos_code']]
        forecast_1['predicted_loading_1'] = y_pred_1

    if regression_mode != 'model_1':
        x_train_reg_2: pd.DataFrame = df_train_reg[reg_keep_cols_2]
        # print(x_train_reg_2.columns.tolist())
        y_train_reg_2: pd.Series = df_train_reg[y_col]

        x_test_reg_2: pd.DataFrame = df_test_reg[reg_keep_cols_2]
        # print(x_test_reg_2.columns.tolist())
        y_test_reg_2: pd.Series = df_test_reg[y_col]

        x_train_reg_proc_2, x_test_reg_proc_2 = \
            prepare_features_for_sklearn_modelling(x_train_reg_2, x_test_reg_2)

        if rerun_model:
            model_reg_2, trn_score_2, val_score_2 = \
                high_level_rf_reg_modeler(x_train_reg_proc_2,
                                          y_train_reg_2,
                                          x_test_reg_proc_2,
                                          y_test_reg_2)
            dump(model_reg_2, model_pickle_path_2)
        else:
            model_reg_2 = load(model_pickle_path_2)

        y_pred_2: np.ndarray = model_reg_2.predict(x_test_reg_proc_2)
        forecast_2: pd.DataFrame = df_test_reg.reset_index()[['invoice_date', 'pos_code']]
        forecast_2['predicted_loading_2'] = y_pred_2

    if regression_mode == 'model_1':
        y_pred: np.ndarray = y_pred_1
    elif regression_mode == 'model_2':
        y_pred: np.ndarray = y_pred_2
    elif regression_mode == 'ensemble':
        temp = df_test_reg.reset_index()
        assert (temp['pre_nonreplacement_holiday'].dtype == 'bool'
                and temp['triple_sell'].dtype == 'bool'
                and temp['double_sell'].dtype == 'bool')
        special_days_mask = (temp['pre_nonreplacement_holiday']
                             | temp['triple_sell']
                             | temp['double_sell'])
        special_days_mask = (special_days_mask
                             | (temp['days_since_price_chg_ann'] >= 0)
                             | (temp['credit_request_coeff'] > 0)
                             | (temp['days_from_easter'] >= 0))
        bad_weather_mask = (temp['is_bad_weather_last_15'] == 1)
        bad_weather_mask = (bad_weather_mask & ~special_days_mask)
        forecast: pd.DataFrame = forecast_1.rename(
            columns={'predicted_loading_1': 'predicted_loading'})
        forecast.loc[special_days_mask, 'predicted_loading'] = \
            [max(item) for item in zip(forecast_1.loc[special_days_mask, 'predicted_loading_1'],
                                       forecast_2.loc[special_days_mask, 'predicted_loading_2'])]
        forecast.loc[bad_weather_mask, 'predicted_loading'] = forecast_2.loc[
            bad_weather_mask, 'predicted_loading_2']
        assert len(forecast) == len(y_pred_1) == len(y_pred_2)
        y_pred: np.ndarray = forecast['predicted_loading'].to_numpy()

    return y_pred


# implements modelling section per product, according to 'model_period' parameter (annual, monthly or weekly).
# Classifier predicts whether a shipment will happen for a particular POS for said date and product;
# if shipment occurrence is predicted regressor predicts the shipment amount for all such POSes.
# To save time, it takes parameter demarcating whether current date is first in the model period;
# if it is then it retrains the models and pickles for subsequent usage, else it loads previously pickled model
# and predicts using them; if it fails to find a saved model then it retrains and saves the model again.
def model(df_train: pd.DataFrame,
          df_test: pd.DataFrame,
          y_col: str,
          is_pred_date_first_in_period: bool,
          extra_test_features_introduced: list,
          model_period: str,
          current_product: str,
          so_code: str,
          model_pickle_path: str,
          run_log_path: str) -> tuple:

    clf_model_pickle_path = os.path.join(
        model_pickle_path,
        'model_clf_{}_{}_{}.jpkl'.format(model_period, so_code, current_product))
    if is_pred_date_first_in_period:
        # Do training till last period marker & pickle the model for later use
        df_test = nonzero_shipment_classification(
            df_train,
            df_test,
            y_col,
            clf_drop_cols,
            extra_test_features_introduced,
            current_product,
            True,
            clf_model_pickle_path,
            run_log_path
        )
    else:
        try:
            df_test = nonzero_shipment_classification(
                df_train,
                df_test,
                y_col,
                clf_drop_cols,
                extra_test_features_introduced,
                current_product,
                False,
                clf_model_pickle_path,
                run_log_path
            )
        except FileNotFoundError:
            # Do training till last period marker & pickle the model for later use
            df_test = nonzero_shipment_classification(
                df_train,
                df_test,
                y_col,
                clf_drop_cols,
                extra_test_features_introduced,
                current_product,
                True,
                clf_model_pickle_path,
                run_log_path
            )

    # non-zero marked shipments forecast using regression
    df_train_reg = df_train.loc[~df_train['is_zero_sale']].copy()
    # Take only visit plan wise
    df_test_reg = df_test.loc[(df_test['isVisitPlan']
                               & df_test['pred_is_nonzero_shipments'])].copy()
    if len(df_test_reg) == 0:
        forecast_result: pd.DataFrame = df_test.loc[
            df_test['isVisitPlan']].rename(columns={'shipments': 'known_shipment'})
        forecast_result = forecast_result.reset_index()
        forecast_result['predicted_loading'] = 0
        return df_test, forecast_result
    if use_drop_reg_cols_instead_of_keep:
        reg_model_pickle_path = os.path.join(
            model_pickle_path,
            'model_reg_{}_{}_{}.jpkl'.format(model_period, so_code, current_product))
        if is_pred_date_first_in_period:
            # Do training till last period marker & pickle the model for later use
            y_pred: np.ndarray = drop_cols_regression(
                df_train_reg,
                df_test_reg,
                y_col,
                extra_test_features_introduced,
                True,
                reg_model_pickle_path
            )
        else:
            try:
                y_pred: np.ndarray = drop_cols_regression(
                    df_train_reg,
                    df_test_reg,
                    y_col,
                    extra_test_features_introduced,
                    False,
                    reg_model_pickle_path
                )
            except FileNotFoundError:
                # Do training till last period marker & pickle the model for later use
                y_pred: np.ndarray = drop_cols_regression(
                    df_train_reg,
                    df_test_reg,
                    y_col,
                    extra_test_features_introduced,
                    True,
                    reg_model_pickle_path
                )
        forecast: pd.DataFrame = df_test_reg.reset_index()[['invoice_date', 'pos_code']]
        forecast['predicted_loading'] = y_pred
    else:
        reg_model_pickle_path_1 = os.path.join(
            model_pickle_path,
            'model_reg_{}_{}_{}_1.jpkl'.format(model_period, so_code, current_product))
        reg_model_pickle_path_2 = os.path.join(
            model_pickle_path,
            'model_reg_{}_{}_{}_2.jpkl'.format(model_period, so_code, current_product))
        if is_pred_date_first_in_period:
            # Do training till last period marker & pickle the model for later use
            y_pred: np.ndarray = multi_model_keep_cols_regression(
                df_train_reg,
                df_test_reg,
                y_col,
                True,
                reg_model_pickle_path_1,
                reg_model_pickle_path_2
            )
        else:
            try:
                y_pred: np.ndarray = multi_model_keep_cols_regression(
                    df_train_reg,
                    df_test_reg,
                    y_col,
                    False,
                    reg_model_pickle_path_1,
                    reg_model_pickle_path_2
                )
            except FileNotFoundError:
                # Do training till last period marker & pickle the model for later use
                y_pred: np.ndarray = multi_model_keep_cols_regression(
                    df_train_reg,
                    df_test_reg,
                    y_col,
                    True,
                    reg_model_pickle_path_1,
                    reg_model_pickle_path_2
                )
    forecast: pd.DataFrame = df_test_reg.reset_index()[['invoice_date', 'pos_code']]
    forecast['predicted_loading'] = y_pred
    forecast_result: pd.DataFrame = df_test.loc[
        df_test['isVisitPlan']].rename(columns={'shipments': 'known_shipment'})
    forecast_result = forecast_result.merge(forecast,
                                            how='left',
                                            on=['invoice_date', 'pos_code'])
    forecast_result['predicted_loading'] = forecast_result['predicted_loading'].fillna(0)

    return df_test, forecast_result


# flow handler for post model operations on the output of modelling section per product per date.
# This flow involves force inflation of results for special days, aggregation and buffer addition on agg results.
# Takes in paramaters through 'hyperparams_dict' determining behavior of post model operations flow.
def high_level_post_model_operations(agent_forecasts: pd.DataFrame,
                                     train_data_with_agent: pd.DataFrame,
                                     past_test_data_with_agent: pd.DataFrame,
                                     inflated_demand_marker_data: pd.DataFrame,
                                     model_type: str,
                                     product_code: str,
                                     hyperparams_dict: dict,
                                     int_result_dump_path: str,
                                     run_log_path: str) -> Optional[pd.DataFrame]:
    assert agent_forecasts['agent_code'].nunique() == 1
    assert agent_forecasts['visit_date'].nunique() == 1
    current_pred_date = agent_forecasts['visit_date'].iloc[0]
    current_agent: str = agent_forecasts['agent_code'].iloc[0]
    current_product_cat: str = agent_forecasts['product_cat'].iloc[0]
    # print('current agent:', current_agent)

    agent_params: dict = hyperparams_dict[current_agent]
    chosen_agg_buffer_pct = agent_params['agg_buffer_pct']
    chosen_forced_loading_upper_limit = agent_params['force_upper_limit']
    chosen_force_inflation_factor = agent_params['force_inflation_factor']
    chosen_std_adjust_factor = agent_params['std_adjust_factor']

    agg_forecast_df = __high_level_std_inflate_agg_wrapper(
        agent_forecasts,
        train_data_with_agent,
        past_test_data_with_agent,
        inflated_demand_marker_data,
        current_pred_date,
        model_type,
        product_code,
        current_agent,
        int_result_dump_path,
        chosen_agg_buffer_pct,
        chosen_forced_loading_upper_limit,
        chosen_force_inflation_factor,
        chosen_std_adjust_factor
    )

    with open(run_log_path, 'a') as model_log:
        model_log.write('hyperparams loaded from storage:\n')
        model_log.write('final params used for {}, {}: \n'.format(current_agent, current_pred_date))
        model_log.write('agg buffer pct: {}\n'.format(chosen_agg_buffer_pct))
        model_log.write('force upper limit: {}\n'.format(chosen_forced_loading_upper_limit))
        model_log.write('force inflation factor: {}\n'.format(chosen_force_inflation_factor))
        model_log.write('std adjust factor: {}\n'.format(chosen_std_adjust_factor))

    return agg_forecast_df


# legacy method which was determining if a POS is problematic, also calculates some
# POS-wise measures from the training data
def _flag_problematic_pos(train_data: pd.DataFrame,
                          test_data: pd.DataFrame) -> tuple:
    train_df = train_data.copy()
    test_df = test_data.copy()

    train_measures: pd.DataFrame = train_df.groupby(
        'pos_code').agg({'shipments': ['std', 'median', 'mean', 'count']})
    train_measures.columns = ['train_' + '_'.join(col) for
                              col in train_measures.columns.values]
    train_measures['train_shipments_vp_count'] = \
        train_df[train_df['isVisitPlan']].groupby(
            'pos_code').agg({'shipments': 'count'})
    # print(train_measures)
    test_df = test_df.merge(train_measures, how='left', on='pos_code')

    train_dates_count = len(set(train_df.index))
    train_df['shipments'] = train_df['shipments'].replace(0, np.nan)
    train_df = train_df.dropna(subset=['shipments'])

    train_nonzero_measures: pd.DataFrame = train_df.groupby(
        'pos_code').agg({'shipments': ['std', 'median', 'mean', 'count'],
                         'isVisitPlan': 'sum'})
    train_nonzero_measures.columns = ['train_nonzero_' + '_'.join(col) for
                                      col in train_nonzero_measures.columns.values]
    train_nonzero_measures['train_nonzero_shipments_vp_count'] = \
        train_df[train_df['isVisitPlan']].groupby(
            'pos_code').agg({'shipments': 'count'})

    test_df = test_df.merge(train_nonzero_measures, how='left', on='pos_code')
    test_df['Shipments_train_density'] = test_df['train_nonzero_shipments_count'] / train_dates_count
    test_df['Shipments_train_sparsity'] = 1 - (test_df['train_nonzero_shipments_vp_count']
                                               / test_df['train_shipments_vp_count'])
    test_df['isVisitPlan_train_density'] = (test_df['train_nonzero_isVisitPlan_sum']
                                            / test_df['train_nonzero_shipments_count'])
    # print(test_df)

    mask = ((test_df['train_nonzero_shipments_std']
             >= test_df['train_nonzero_shipments_std'].quantile(high_std_thr))
            | (test_df['train_shipments_std']
               >= test_df['train_shipments_std'].quantile(high_std_thr))) \
           & ((test_df['Shipments_train_density'] <= low_density_thr)
              | (test_df['isVisitPlan_train_density'] <= low_vp_thr))
    problematic_pos = pd.Series(test_df.loc[mask, 'pos_code'].unique())

    extra_test_features_introduced = ['train_nonzero_shipments_std',
                                      'train_nonzero_shipments_median',
                                      'train_nonzero_shipments_mean',
                                      'train_nonzero_shipments_count',
                                      'train_nonzero_isVisitPlan_sum',
                                      'train_nonzero_shipments_vp_count',
                                      'train_shipments_std',
                                      'train_shipments_mean',
                                      'train_shipments_median',
                                      'train_shipments_count',
                                      'train_shipments_vp_count',
                                      'Shipments_train_density',
                                      'Shipments_train_sparsity',
                                      'isVisitPlan_train_density']

    return problematic_pos, test_df, extra_test_features_introduced


# Adjustments based on standard deviation
def _post_model_std_based_adjust(forecast_result: pd.DataFrame,
                                 std_adjustment_factor: float) -> pd.DataFrame:
    forecast_result.loc[:, 'predicted_loading_std_adj_amt'] = 0
    high_std_thr_val = forecast_result['train_nonzero_shipments_std'].quantile(high_std_thr)
    high_overall_std_thr_val = forecast_result['train_shipments_std'].quantile(high_std_thr)

    if std_adjustment_factor > 0:
        std_adjust_mask = (((forecast_result['train_nonzero_shipments_std'] >= high_std_thr_val)
                            | (forecast_result['train_shipments_std'] >= high_overall_std_thr_val))
                           & (forecast_result['Shipments_train_density'] <= low_density_thr)
                           & (forecast_result['isVisitPlan_train_density'] > low_vp_thr))
        predicted_loading_std_adj = forecast_result.loc[
                                        std_adjust_mask,
                                        'train_nonzero_shipments_std'] * std_adjustment_factor
        predicted_loading_std_adj = predicted_loading_std_adj.fillna(0)
        forecast_result.loc[
            std_adjust_mask, 'predicted_loading'] = forecast_result.loc[
            std_adjust_mask, 'predicted_loading'] + predicted_loading_std_adj
        forecast_result.loc[
            std_adjust_mask, 'predicted_loading_std_adj_amt'] = predicted_loading_std_adj

    return forecast_result


# special days' prediction inflation before aggregation
def _post_model_result_inflation(forecast_result: pd.DataFrame,
                                 inflation_factor: float,
                                 int_result_dump_path: str,
                                 pred_date: pd.Timestamp,
                                 model_type: str,
                                 product_code: str,
                                 agent_code: Optional[str] = None) -> pd.DataFrame:
    def __inflation_levels_list(row, result_column_name):
        _inflation_factor = inflation_factor
        if row['triple_sell'] or row['credit_request_type'] == 'TRIPLE':
            _inflation_factor = _inflation_factor + 0.2

        if not row['pred_is_nonzero_shipments']:
            return [row[result_column_name] * _inflation_factor, 0]
        else:
            return [row[result_column_name] * _inflation_factor,
                    (row[result_column_name]
                     + row['train_nonzero_shipments_mean'] * (_inflation_factor - 1))]

    forecast_result.loc[:, 'predicted_loading'] = \
        forecast_result.apply(lambda row:
                              np.nanmax(__inflation_levels_list(row, 'predicted_loading'))
                              if (row['triple_sell']
                                  or row['double_sell']
                                  or (row['credit_request_type'] == 'TRIPLE')
                                  or (row['credit_request_type'] == 'DOUBLE')
                                  or (row['days_from_easter'] >= 0)
                                  or row['pre_nonreplacement_holiday'])

                              else np.nanmean(__inflation_levels_list(row, 'predicted_loading'))
                              if (row['days_since_price_chg_ann'] > 0
                                  and row['days_from_price_chg'] > 0)

                              else row['predicted_loading'],
                              axis=1)

    return forecast_result


# wrapper method performing std based adjustments, special days' result inflation
# and result aggregation and buffering.
def __high_level_std_inflate_agg_wrapper(agent_forecasts: pd.DataFrame,
                                         train_data_with_agent: pd.DataFrame,
                                         past_test_data_with_agent: pd.DataFrame,
                                         inflated_demand_marker_data: pd.DataFrame,
                                         pred_date: pd.Timestamp,
                                         model_type: str,
                                         product_code: str,
                                         agent_code: str,
                                         int_result_dump_path: str,
                                         agg_buffer_pct: float,
                                         force_upper_limit: bool,
                                         inflation_factor: float,
                                         std_adjustment_factor: float) -> pd.DataFrame:
    agent_forecasts = _post_model_std_based_adjust(
        agent_forecasts,
        std_adjustment_factor
    )
    agent_forecasts = _post_model_result_inflation(
        agent_forecasts,
        inflation_factor,
        int_result_dump_path,
        pred_date,
        model_type,
        product_code,
        agent_code
    )
    agg_forecast_df: pd.DataFrame = __high_level_agg(
        agent_forecasts,
        train_data_with_agent,
        past_test_data_with_agent,
        inflated_demand_marker_data,
        product_code,
        agg_buffer_pct,
        force_upper_limit
    )

    return agg_forecast_df


# wrapper method calling the result aggregation and buffer addition method
def __high_level_agg(agent_forecasts: pd.DataFrame,
                     train_data_with_agent: pd.DataFrame,
                     past_test_data_with_agent: pd.DataFrame,
                     inflated_demand_marker_data: pd.DataFrame,
                     product_code: str,
                     agg_buffer_pct: float,
                     force_upper_limit: bool) -> pd.DataFrame:

    agg_forecast_df: pd.DataFrame = high_level_post_model_aggregator(
        agent_forecasts,
        train_data_with_agent,
        past_test_data_with_agent,
        inflated_demand_marker_data,
        agg_buffer_pct,
        force_upper_limit
    )
    agg_forecast_df['product_code'] = product_code

    return agg_forecast_df


def __form_tuning_result_df(keys: list, values: list) -> pd.DataFrame:
    assert len(keys) == len(values)
    assert isinstance(values[0], list)
    df_len = len(values[0])
    if df_len > 1:
        return pd.DataFrame(dict(zip(keys, values)))
    elif df_len == 1:
        return pd.DataFrame(dict(zip(keys, values)), index=[0])
