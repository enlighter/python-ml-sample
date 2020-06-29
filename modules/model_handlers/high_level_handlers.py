import pandas as pd
import numpy as np
import json
from tqdm import tqdm

from typing import Optional
from multiprocessing import Pool
import traceback
import os

from modules.model_handlers.high_level_handler_tools import single_date_split_model_forecast
from modules.aggregation import high_level_post_agg_comparison_prep
from modules.aggregation import get_forecast_performance_figures


# main modelling flow handler per product; for each date in validation period performs
# annual, monthly and weekly modelling flows and finally ensembles the 3 different modelling
# flow results, calculates metrics and stores results+metrics file per product for further processing
def high_level_model_handler(prepared_product_data: pd.DataFrame,
                             present_agent_map_data: pd.DataFrame,
                             shipment_data: pd.DataFrame,
                             inflated_demand_marker_data: pd.DataFrame,
                             shipment_split_data: pd.DataFrame,
                             sr_loading_data: pd.DataFrame,
                             sr_unloading_data: pd.DataFrame,
                             pred_start_date: pd.Timestamp,
                             y_col: str,
                             remove_first_month_from_training: bool,
                             so_code: str,
                             output_path: str,
                             run_log_path: str,
                             process_pool: Pool) -> Optional[pd.DataFrame]:
    model_pickle_path: str = 'model_pickles'
    os.makedirs(model_pickle_path, exist_ok=True)
    # Model pickles are only utilised for reusing during the full run for 1 product
    # So at the start of run for a new product any existing model pickle file can be
    # discarded to save disk space.
    exisiting_model_pickle_files = os.listdir(model_pickle_path)
    if len(exisiting_model_pickle_files) > 0:
        for model_pickle in exisiting_model_pickle_files:
            file_path = os.path.join(model_pickle_path, model_pickle)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                pass
    int_result_dump_path: str = os.path.join(output_path, 'intermediate_dumps')
    os.makedirs(int_result_dump_path, exist_ok=True)  # succeeds even if directory exists.
    hyperparams_storage_path: str = 'hyperparams'
    os.makedirs(hyperparams_storage_path, exist_ok=True)

    current_product: str = prepared_product_data.index.get_level_values(0)[0]
    # print("current_product", current_product)
    product_df: pd.DataFrame = prepared_product_data.copy()
    product_df.index = product_df.index.droplevel()
    product_df = product_df.sort_index()
    dates_in_data: pd.DatetimeIndex = product_df.index
    pred_dates: np.ndarray = dates_in_data[dates_in_data >= pred_start_date].unique().array
    daywise_forecast_df_list: list = []
    daywise_forecast_raw_annual_model_list: list = []
    daywise_forecast_raw_monthly_model_list: list = []
    daywise_forecast_raw_weekly_model_list: list = []

    past_agent_map: pd.DataFrame = shipment_data.loc[shipment_data['product_code'] == current_product]
    past_agent_map = past_agent_map.reset_index()
    past_agent_map = past_agent_map[['invoice_date', 'pos_code', 'agent_code']]

    with open(os.path.join(
            hyperparams_storage_path,
            '{}_{}_hyperparams.json'.format(so_code, current_product)), 'r') as fp:
        hyperparams_dict: dict = json.load(fp)

    with open(run_log_path, 'a') as model_log:
        model_log.write('New product run: {}\n'.format(current_product))

    for pred_date in tqdm(pred_dates):
        # print(pred_date)
        daywise_forecasts = high_level_daywise_model_handler(
            product_df,
            inflated_demand_marker_data,
            present_agent_map_data,
            past_agent_map,
            pred_date,
            remove_first_month_from_training,
            y_col,
            hyperparams_dict,
            current_product,
            so_code,
            run_log_path,
            model_pickle_path,
            int_result_dump_path,
            process_pool
        )
        if daywise_forecasts is not None:
            curr_forecast_df, curr_forecast_raw_annual_model, \
                curr_forecast_raw_monthly_model, curr_forecast_raw_weekly_model = daywise_forecasts
            daywise_forecast_df_list.append(curr_forecast_df)
            daywise_forecast_raw_annual_model_list.append(curr_forecast_raw_annual_model)
            daywise_forecast_raw_monthly_model_list.append(curr_forecast_raw_monthly_model)
            daywise_forecast_raw_weekly_model_list.append(curr_forecast_raw_weekly_model)
        else:
            continue

    if len(daywise_forecast_df_list) == 0:
        return None
    else:
        forecast_df: pd.DataFrame = pd.concat(daywise_forecast_df_list, sort=False)
        forecast_raw_annual_model: pd.DataFrame = pd.concat(daywise_forecast_raw_annual_model_list, sort=False)
        forecast_raw_annual_model.to_csv(
            os.path.join(int_result_dump_path,
                         'result_with_zeroes_annual_model_{}.csv'.format(current_product))
        )
        forecast_raw_monthly_model: pd.DataFrame = pd.concat(daywise_forecast_raw_monthly_model_list, sort=False)
        forecast_raw_monthly_model.to_csv(
            os.path.join(int_result_dump_path,
                         'result_with_zeroes_monthly_model_{}.csv'.format(current_product))
        )
        forecast_raw_weekly_model: pd.DataFrame = pd.concat(daywise_forecast_raw_weekly_model_list, sort=False)
        forecast_raw_weekly_model.to_csv(
            os.path.join(int_result_dump_path,
                         'result_with_zeroes_weekly_model_{}.csv'.format(current_product))
        )

        forecast_comparison_df = high_level_post_agg_comparison_prep(
            forecast_df,
            current_product,
            shipment_data,
            shipment_split_data,
            sr_loading_data,
            sr_unloading_data
        )
        asl_unload_pct, asl_oos_pct, jti_unload_pct, jti_oos_pct = get_forecast_performance_figures(
            forecast_comparison_df
        )
        asl_cost = asl_unload_pct + asl_oos_pct
        jti_cost = jti_unload_pct + jti_oos_pct
        with open(run_log_path, 'a') as model_log:
            model_log.write('product {} final result:'.format(current_product))
            model_log.write('asl_unload_pct: ' + str(asl_unload_pct)
                            + 'asl_oos_pct: ' + str(asl_oos_pct)
                            + 'asl_cost: ' + str(asl_cost) + '\n')
            model_log.write('jti_unload_pct: ' + str(jti_unload_pct)
                            + 'jti_oos_pct: ' + str(jti_oos_pct)
                            + 'jti_cost: ' + str(jti_cost) + '\n')
        forecast_comparison_df.to_csv(os.path.join(
            output_path,
            'final_output_rf_high_no_pos_removed_{}.csv'.format(current_product)))

        return forecast_comparison_df


# modelling flow handler per product for one date; runs annual, monthly and weekly modelling flows
# and finally ensembles the 3 different modelling flow results and returns the ensembled results
# as well as the individual modelling flow results
def high_level_daywise_model_handler(product_data: pd.DataFrame,
                                     inflated_demand_marker_data: pd.DataFrame,
                                     present_agent_map_data: pd.DataFrame,
                                     past_agent_map_data: pd.DataFrame,
                                     pred_date: pd.Timestamp,
                                     remove_first_month_from_training: bool,
                                     y_col: str,
                                     hyperparams_dict: dict,
                                     product_code: str,
                                     so_code: str,
                                     run_log_path: str,
                                     model_pickle_path: str,
                                     int_result_dump_path: str,
                                     process_pool: Pool) -> Optional[tuple]:
    try:
        # Annual model section
        annual_model_forecasts = single_date_split_model_forecast(
            product_data,
            inflated_demand_marker_data,
            present_agent_map_data,
            past_agent_map_data,
            pred_date,
            remove_first_month_from_training,
            y_col,
            'annual',
            hyperparams_dict,
            product_code,
            so_code,
            model_pickle_path,
            run_log_path,
            int_result_dump_path
        )
        if annual_model_forecasts is not None:
            forecast_raw_annual_model, forecast_df_annual_model = annual_model_forecasts
        else:
            return None

        # Monthly model section
        monthly_model_forecasts = single_date_split_model_forecast(
            product_data,
            inflated_demand_marker_data,
            present_agent_map_data,
            past_agent_map_data,
            pred_date,
            remove_first_month_from_training,
            y_col,
            'monthly',
            hyperparams_dict,
            product_code,
            so_code,
            model_pickle_path,
            run_log_path,
            int_result_dump_path
        )
        if monthly_model_forecasts is not None:
            forecast_raw_monthly_model, forecast_df_monthly_model = monthly_model_forecasts
        else:
            return None

        # Weekly model section
        weekly_model_forecasts = single_date_split_model_forecast(
            product_data,
            inflated_demand_marker_data,
            present_agent_map_data,
            past_agent_map_data,
            pred_date,
            remove_first_month_from_training,
            y_col,
            'weekly',
            hyperparams_dict,
            product_code,
            so_code,
            model_pickle_path,
            run_log_path,
            int_result_dump_path
        )
        if weekly_model_forecasts is not None:
            forecast_raw_weekly_model, forecast_df_weekly_model = weekly_model_forecasts
        else:
            return None

        assert len(forecast_df_annual_model) == len(forecast_df_monthly_model) == len(forecast_df_weekly_model)
        # check if some other columns need to be dropped
        forecast_df: pd.DataFrame = forecast_df_annual_model.drop(columns=['predicted_loading'])
        forecast_df.loc[:, 'predicted_loading_annual_model'] = forecast_df_annual_model['predicted_loading'].array
        forecast_df.loc[:, 'predicted_loading_monthly_model'] = forecast_df_monthly_model['predicted_loading'].array
        forecast_df.loc[:, 'predicted_loading_weekly_model'] = forecast_df_weekly_model['predicted_loading'].array
        forecast_df['predicted_loading'] = forecast_df[
            ['predicted_loading_annual_model',
             'predicted_loading_monthly_model',
             'predicted_loading_weekly_model']].mean(axis=1, skipna=True)
        forecast_df['predicted_loading'] = forecast_df['predicted_loading'].round(1)

        return forecast_df, forecast_raw_annual_model, forecast_raw_monthly_model, forecast_raw_weekly_model
    except Exception as e:
        print('error in {}: check log'.format(product_code))
        with open(run_log_path, 'a') as model_log:
            model_log.write(product_code + ' ' + str(pred_date) + ' error encountered: \n')
            model_log.write(traceback.format_exc())
