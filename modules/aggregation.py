import pandas as pd
import numpy as np

from config import quantity_lower_limit, loading_size_thr


# aggregates predicted shipments agent level for given product and date
# and applies dynamic buffering to get loading prediction.
# For buffer and limits calculation, it uses historical agg sample created from both pos set and agent set.
def high_level_post_model_aggregator(product_date_agent_forecasts_data: pd.DataFrame,
                                     train_data_with_agent: pd.DataFrame,
                                     past_test_data_with_agent: pd.DataFrame,
                                     inflated_demand_marker_data: pd.DataFrame,
                                     agg_buffer_pct: float,
                                     force_loading_upper_limit: bool) -> pd.DataFrame:
    current_date: pd.Timestamp = product_date_agent_forecasts_data['visit_date'].iloc[0]
    current_agent: str = product_date_agent_forecasts_data['agent_code'].iloc[0]
    # print(current_date, current_agent)

    train_data_cols = ['visit_date', 'agent_code', 'pos_code', 'shipments']
    poswise_forecast: pd.DataFrame = product_date_agent_forecasts_data.copy()
    current_pos_list: pd.Series = poswise_forecast['pos_code']
    train_df: pd.DataFrame = train_data_with_agent.loc[
        train_data_with_agent['pos_code'].isin(current_pos_list), train_data_cols]
    # The pos-set wise data for days in latest week can paint an incomplete aggregate picture.
    # Thus pos-set wise data should only be taken till last week to current date
    # past_test_data_with_agent = past_test_data_with_agent.loc[
    #     (past_test_data_with_agent['year'] < current_date.year)
    #     | (past_test_data_with_agent['week'] < current_date.weekofyear)]
    if len(past_test_data_with_agent) > 0:
        past_test_df: pd.DataFrame = past_test_data_with_agent.loc[
            past_test_data_with_agent['pos_code'].isin(current_pos_list), train_data_cols]
        past_test_df = past_test_df.loc[past_test_df['visit_date'] < current_date]
        train_df = pd.concat([train_df, past_test_df], ignore_index=True)
    train_agent_df: pd.DataFrame = train_data_with_agent.loc[
        train_data_with_agent['agent_code'] == current_agent, train_data_cols]
    if len(past_test_data_with_agent) > 0:
        past_test_agent_df: pd.DataFrame = past_test_data_with_agent.loc[
            past_test_data_with_agent['agent_code'] == current_agent, train_data_cols]
        past_test_agent_df = past_test_agent_df.loc[past_test_agent_df['visit_date'] < current_date]
        train_agent_df = pd.concat([train_agent_df, past_test_agent_df], ignore_index=True)
    new_pos_list: pd.Series = current_pos_list[~current_pos_list.isin(train_df['pos_code'])]
    # print('new pos count:', glen(new_pos_list))
    new_pos_count: int = len(new_pos_list)
    train_pos_count: int = len(current_pos_list) - new_pos_count
    current_pos_count: int = len(current_pos_list)
    if train_pos_count != 0:
        extrapolation_factor: float = current_pos_count/train_pos_count
    else:
        extrapolation_factor: float = np.nan

    # Agg the results
    agg_forecast: pd.DataFrame = poswise_forecast.groupby(['visit_date']).agg(
        {'predicted_loading': 'sum',
         'known_shipment': 'sum',
         'orig_pred_load': 'sum',
         'pred_is_nonzero_shipments': 'mean',
         'pre_nonreplacement_holiday': 'mean',
         'double_sell': 'mean',
         'triple_sell': 'mean',
         'VisitPlanWeek': 'first',
         'week': 'first',
         'month': 'first',
         'year': 'first',
         'credit_request_type': 'first',
         'days_from_easter': 'first',
         'days_since_price_chg_ann': 'first',
         'days_from_price_chg': 'first',
         'agent_code': 'first',
         'product_cat': 'first'})
    assert len(agg_forecast) == 1
    agg_forecast['original_predicted_loading'] = agg_forecast['predicted_loading']
    agg_forecast['pred_is_nonzero_shipments'] = \
        agg_forecast['pred_is_nonzero_shipments'].astype('float')
    agg_forecast['pred_zero_pct'] = 1 - agg_forecast['pred_is_nonzero_shipments']
    agg_forecast = agg_forecast.drop(columns=['pred_is_nonzero_shipments'])
    agg_forecast = agg_forecast.rename(
        columns={'double_sell': 'double_sell_proportion',
                 'triple_sell': 'triple_sell_proportion'})
    agg_forecast['pre_nonreplacement_holiday'] = \
        agg_forecast['pre_nonreplacement_holiday'].astype('float')
    agg_forecast['NationalDoubleSell'] = inflated_demand_marker_data['NationalDoubleSell']
    agg_forecast['NationalTripleSell'] = inflated_demand_marker_data['NationalTripleSell']
    agg_forecast['curr_pos_count'] = [current_pos_count]
    agg_forecast['new_pos_count'] = [new_pos_count]
    assert len(agg_forecast) == 1

    nan_count = 0
    if train_pos_count == 0:
        limit_pos_set_max_sale = np.nan
        limit_pos_set_peak_sale = np.nan
        limit_pos_set_very_high_sale = np.nan
        limit_pos_set_high_sale = np.nan
        limit_pos_set_median_sale = np.nan
        train_shipments_mean = np.nan
        train_shipments_min = np.nan
        nan_count += 1
    else:
        # prepare
        shipments_train_sample: pd.DataFrame = train_df.groupby(
            [pd.Grouper(key='visit_date', freq='W'), 'pos_code']).agg({'shipments': 'mean'})
        shipments_train_sample = shipments_train_sample.reset_index().groupby(
            'visit_date').agg({'shipments': 'sum', 'pos_code': 'count'})
        shipments_train_sample.columns = ['shipments_train_sum', 'pos_code_count']
        limit_pos_set_max_sale = (shipments_train_sample['shipments_train_sum'].quantile(1)
                                  * extrapolation_factor)
        limit_pos_set_peak_sale = (shipments_train_sample['shipments_train_sum'].quantile(0.999)
                                   * extrapolation_factor)
        limit_pos_set_very_high_sale = (shipments_train_sample['shipments_train_sum'].quantile(0.99)
                                        * extrapolation_factor)
        limit_pos_set_high_sale = (shipments_train_sample['shipments_train_sum'].quantile(0.98)
                                   * extrapolation_factor)
        limit_pos_set_median_sale = (shipments_train_sample['shipments_train_sum'].median()
                                     * extrapolation_factor)

        shipments_train_sample = shipments_train_sample.last('6M')
        train_shipments_mean = shipments_train_sample['shipments_train_sum'].mean()
        train_shipments_min = shipments_train_sample['shipments_train_sum'].min()
    if len(train_agent_df) > 0:
        train_agent_shipments: pd.DataFrame = train_agent_df.groupby(
            'visit_date').agg({'shipments': 'sum'})
        train_agent_shipments.columns = ['agent_shipments_train_sum']
        limit_agent_max_sale = train_agent_shipments[
            'agent_shipments_train_sum'].quantile(1)
        limit_agent_peak_sale = train_agent_shipments[
            'agent_shipments_train_sum'].quantile(0.999)
        limit_agent_very_high_sale = train_agent_shipments[
            'agent_shipments_train_sum'].quantile(0.99)
        limit_agent_high_sale = train_agent_shipments[
            'agent_shipments_train_sum'].quantile(0.98)
        limit_agent_median_sale = train_agent_shipments[
            'agent_shipments_train_sum'].median()
    else:
        limit_agent_max_sale = np.nan
        limit_agent_peak_sale = np.nan
        limit_agent_very_high_sale = np.nan
        limit_agent_high_sale = np.nan
        limit_agent_median_sale = np.nan
        nan_count += 1
    if nan_count < 2:
        limit_max_sale = np.nanmax([limit_pos_set_max_sale, limit_agent_max_sale])
        limit_peak_sale = np.nanmax([limit_pos_set_peak_sale, limit_agent_peak_sale])
        limit_very_high_sale = np.nanmax([limit_pos_set_very_high_sale, limit_agent_very_high_sale])
        limit_high_sale = np.nanmax([limit_pos_set_high_sale, limit_agent_high_sale])
    else:
        # case where both agent and pos set is entirely new
        limit_max_sale = np.nan
        limit_peak_sale = np.nan
        limit_very_high_sale = np.nan
        limit_high_sale = np.nan
    if limit_pos_set_median_sale != np.nan and limit_agent_median_sale != np.nan:
        limit_median_sale = np.nanmean([limit_pos_set_median_sale, limit_agent_median_sale])
    elif limit_pos_set_median_sale != np.nan:
        limit_median_sale = limit_pos_set_median_sale
    elif limit_agent_median_sale != np.nan:
        limit_median_sale = limit_agent_median_sale
    else:
        limit_median_sale = quantity_lower_limit
    agg_forecast['train_shipments_mean'] = [train_shipments_mean]
    agg_forecast['train_shipments_min'] = [train_shipments_min]
    agg_forecast['limit_max_sale'] = [limit_max_sale]
    agg_forecast['limit_peak_sale'] = [limit_peak_sale]
    agg_forecast['limit_very_high_sale'] = [limit_very_high_sale]
    agg_forecast['limit_high_sale'] = [limit_high_sale]
    agg_forecast['limit_median_sale'] = [limit_median_sale]

    # agg loading buffer
    def _agg_loading_levels_list(predicted_loading):
        return [predicted_loading * (1 + agg_buffer_pct),
                predicted_loading + agg_loading_buffer]

    if agg_forecast['NationalTripleSell'].iloc[0]:
        agg_buffer_pct = agg_buffer_pct * agg_forecast['triple_sell_proportion'].iloc[0]
    elif agg_forecast['NationalDoubleSell'].iloc[0]:
        agg_buffer_pct = agg_buffer_pct * agg_forecast['double_sell_proportion'].iloc[0]
    agg_forecast['nominal_agg_buffer'] = [train_shipments_mean*agg_buffer_pct]
    train_shipments_mean = train_shipments_mean*extrapolation_factor
    agg_loading_buffer = train_shipments_mean*agg_buffer_pct
    agg_forecast['used_agg_buffer'] = [agg_loading_buffer]
    # print(agg_loading_buffer)
    agg_forecast['predicted_loading'] = \
        agg_forecast['predicted_loading'].apply(lambda x:
                                                np.nanmean(_agg_loading_levels_list(x)))

    # Upper limit
    if force_loading_upper_limit:
        if (agg_forecast['NationalTripleSell'].iloc[0]
                or (agg_forecast['credit_request_type'].iloc[0] == 'TRIPLE')):
            agg_forecast['predicted_loading'] = [np.nanmin(
                [agg_forecast['predicted_loading'].iloc[0], limit_max_sale])]
        elif (agg_forecast['NationalDoubleSell'].iloc[0]
              or (agg_forecast['credit_request_type'].iloc[0] == 'DOUBLE')
              or (agg_forecast['days_from_easter'].iloc[0] >= 0)
              or agg_forecast['pre_nonreplacement_holiday'].iloc[0] > 0):
            agg_forecast['predicted_loading'] = [np.nanmin(
                [agg_forecast['predicted_loading'].iloc[0], limit_peak_sale])]
        elif agg_forecast['predicted_loading'].iloc[0] <= loading_size_thr:
            agg_forecast['predicted_loading'] = [np.nanmin(
                [agg_forecast['predicted_loading'].iloc[0], limit_very_high_sale])]
        else:
            agg_forecast['predicted_loading'] = [np.nanmin(
                [agg_forecast['predicted_loading'].iloc[0], limit_high_sale])]

    agg_forecast['predicted_loading'] = \
        agg_forecast['predicted_loading'].clip(limit_median_sale, None)

    # Finalize
    agg_forecast['predicted_loading'] = \
        np.round(agg_forecast['predicted_loading'] + np.nanmin([2, train_shipments_min * 0.2]), 1)
    if agg_forecast['predicted_loading'].iloc[0] <= quantity_lower_limit:
        agg_forecast['predicted_loading'] = quantity_lower_limit + 0.1
    else:
        agg_forecast['predicted_loading'] = np.round(agg_forecast['predicted_loading'] + 0.5, 1)

    return agg_forecast


# calculates reporting metrics corresponding to each entry in the final aggregated forecast.
def high_level_post_agg_comparison_prep(agg_result_data: pd.DataFrame,
                                        product_code: str,
                                        shipment_data: pd.DataFrame,
                                        shipment_split_data: pd.DataFrame,
                                        sr_loading_data: pd.DataFrame,
                                        sr_unloading_data: pd.DataFrame) -> pd.DataFrame:
    forecast_comparison_df: pd.DataFrame = agg_result_data.reset_index().set_index(
        ['visit_date', 'agent_code', 'product_code'])

    # Add shipment split data: plan compliant & ER
    forecast_comparison_df = forecast_comparison_df.merge(
        shipment_split_data,
        how='left',
        on=['visit_date', 'agent_code', 'product_code'])
    forecast_comparison_df.loc[:, ['non_vp_shipment',
                                   'vp_shipment']] = \
        forecast_comparison_df.loc[:, ['non_vp_shipment', 'vp_shipment']].fillna(0)

    # Add actual shipment for comparison with predicted loading
    shipment_compare = shipment_data[shipment_data['product_code'] == product_code]
    shipment_compare = shipment_compare.groupby(['invoice_date',
                                                 'agent_code',
                                                 'product_code'])[['quantity']].sum()
    shipment_compare.reset_index(inplace=True)
    shipment_compare.rename(columns={'invoice_date': 'visit_date'}, inplace=True)
    shipment_compare.set_index(['visit_date', 'agent_code', 'product_code'], inplace=True)
    forecast_comparison_df['total_shipment'] = shipment_compare['quantity'].round(1)
    forecast_comparison_df['actual_shipment'] = forecast_comparison_df['vp_shipment'].round(1)

    # add actual loading data
    forecast_comparison_df['sr_loading'] = sr_loading_data['quantity']
    forecast_comparison_df['sr_unloading'] = sr_unloading_data['quantity']

    # if all loaded quantities are unloaded, loading won't be null but shipment wil be
    forecast_comparison_df.loc[
        forecast_comparison_df['sr_loading'].notna(), 'actual_shipment'] = forecast_comparison_df.loc[
        forecast_comparison_df['sr_loading'].notna(), 'actual_shipment'].fillna(0)

    # performance comparison
    residual = forecast_comparison_df['predicted_loading'] - forecast_comparison_df['actual_shipment']
    forecast_comparison_df['asl_unloading'] = residual.apply(lambda x: x if x > 0 else 0)
    forecast_comparison_df['OOS_amt_indicative'] = residual.apply(lambda x: -x if x < 0 else 0)
    forecast_comparison_df['is_OOS'] = residual.apply(lambda x: 1 if x <= 0 else 0)
    jti_residual = forecast_comparison_df['sr_loading'] - forecast_comparison_df['actual_shipment']
    forecast_comparison_df['jti_unloading'] = jti_residual.apply(lambda x: x if x > 0 else 0)
    forecast_comparison_df['jti_OOS'] = jti_residual.apply(lambda x: 1 if x <= 0 else 0)

    return forecast_comparison_df


# calculates topline metrics for the forecasts.
def get_forecast_performance_figures(forecast_comparison_data: pd.DataFrame,) -> tuple:
    forecast_comparison_df = forecast_comparison_data[
        forecast_comparison_data['sr_loading'].notna()]
    pred_unloading = (forecast_comparison_df['asl_unloading'].sum()
                      / forecast_comparison_df['predicted_loading'].sum())*100
    pred_oos = forecast_comparison_df['is_OOS'].mean()*100
    jti_unloading = (forecast_comparison_df['jti_unloading'].sum()
                     / forecast_comparison_df['sr_loading'].sum())*100
    jti_oos = forecast_comparison_df['jti_OOS'].mean()*100

    return pred_unloading, pred_oos, jti_unloading, jti_oos
