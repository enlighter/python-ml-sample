import pandas as pd
from tqdm import tqdm

import os
import time
from multiprocessing import Pool

from config import root_path, get_cleaned_data_paths
from modules.load import extract_n_pickle_raw_data, load_data
from modules.data_preparation import prepare_data
from modules.data_preparation import preprocess_ml
from modules.model_handlers import high_level_model_handler

tqdm.pandas()

# Initiating parameters for script
sales_office = 'BC'
current_part = '1'
do_raw_data_extraction = False

output_path = os.path.join(os.path.join(root_path, 'output_final_trials'), sales_office)

test_mode = False
short_run_mode = False
if test_mode:  # values fixed in this case
    rerun_generic_prepare = True
    rerun_premodel_prep = True
else:  # change as necessary
    rerun_generic_prepare = False
    rerun_premodel_prep = False

method: str = "rf"
test_period_start_date: pd.Timestamp = pd.Timestamp(year=2020, month=1, day=1)
remove_first_month_from_training: bool = False


# Load different types of relevant raw data, clean and pickle the datasets (one-time operation)
# and prepare past shipments' timeseries according for each visit plan entry
def generic_load_and_prepare_data():
    cleaned_data_path, visit_plan_data_path, shipment_data_path, date_analysis_data_path, \
        shipment_split_path, sr_loading_data_path, sr_unloading_data_path, \
        stock_collection_data_path, credit_requests_data_path, \
        pre_easter_effect_data_path = get_cleaned_data_paths(
            sales_office
        )
    os.makedirs(cleaned_data_path, exist_ok=True)
    if do_raw_data_extraction:
        extract_n_pickle_raw_data(sales_office,
                                  visit_plan_data_path,
                                  shipment_data_path,
                                  date_analysis_data_path,
                                  shipment_split_path,
                                  sr_loading_data_path,
                                  sr_unloading_data_path,
                                  stock_collection_data_path,
                                  credit_requests_data_path)
    global test_period_start_date
    visit_plan_test_df, pos_visit_plan_df, shipment_df, \
        date_analysis_df, shipment_split_df, stock_collection_df, \
        sr_loading_df, sr_unloading_df, credit_requests_df, \
        pre_easter_effect_df = \
        load_data(visit_plan_data_path,
                  shipment_data_path,
                  date_analysis_data_path,
                  shipment_split_path,
                  sr_loading_data_path,
                  sr_unloading_data_path,
                  stock_collection_data_path,
                  credit_requests_data_path,
                  pre_easter_effect_data_path,
                  test_period_start_date)

    # Prepare POS-Product wise timeseries data
    # print(visit_plan_test_df)
    global rerun_generic_prepare
    if rerun_premodel_prep:
        if rerun_generic_prepare:
            pos_product_ts_df = visit_plan_test_df.groupby(level=0,
                                                           axis=0).progress_apply(prepare_data,
                                                                                  shipment_data=shipment_df)
            pos_product_ts_df.reset_index(level=['invoice_date'], inplace=True)
            print(pos_product_ts_df.head())
            if not (test_mode or short_run_mode):
                pos_product_ts_df.to_pickle('generic_prepared_data_{}.pkl'.format(
                    sales_office))
        else:
            # load already pickled solution
            pos_product_ts_df = pd.read_pickle('generic_prepared_data_{}.pkl'.format(
                sales_office))
            print(pos_product_ts_df.head())
    else:
        pos_product_ts_df = None

    return pos_product_ts_df, visit_plan_test_df, pos_visit_plan_df, \
        shipment_df, date_analysis_df, shipment_split_df, stock_collection_df, \
        sr_loading_df, sr_unloading_df, credit_requests_df, pre_easter_effect_df


if __name__ == '__main__':
    process_pool = Pool(3)  # For all multi-processing in project
    os.makedirs(output_path, exist_ok=True)
    run_log_path = os.path.join(output_path,
                                'run_log_{}_{}.txt'.format(sales_office,
                                                           current_part))
    with open(run_log_path, 'w+') as run_log:
        run_log.write('new run\n')

    start_time = time.perf_counter()

    # Load the prepared timeseries structure and other relevant datasets
    pos_product_ts_data, visit_plan_test_data, pos_visit_plan_data, \
        shipment_data, date_analysis_data, shipment_split_data, stock_collection_data, \
        sr_loading_data, sr_unloading_data, credit_requests_data, pre_easter_effect_data = \
        generic_load_and_prepare_data()

    if rerun_premodel_prep:
        # Create all features for each pos-product combination timeseries
        product_list = pos_product_ts_data['product_code'].unique()
        if short_run_mode:
            product_list = product_list[:10]
            pos_product_ts_data = pos_product_ts_data[
                pos_product_ts_data['product_code'].isin(product_list)]

        # model preprocess step for ml with many features
        shipment_temp: pd.DataFrame = shipment_data.reset_index()
        shipment_temp = shipment_temp.rename(columns={'invoice_date': 'visit_date'})
        shipment_temp = shipment_temp[['visit_date', 'pos_code', 'agent_code']]
        # we need to take all poses visited by the agent on a particular date,
        # irrespective of product. So we are kind of taking the union of pos sets from all
        # products on a particular date for each agent.
        shipment_temp = shipment_temp.drop_duplicates()
        credit_requests_pos_map: pd.DataFrame = pd.merge(credit_requests_data,
                                                         shipment_temp,
                                                         how='left',
                                                         on=['visit_date', 'agent_code'])
        credit_requests_pos_map = credit_requests_pos_map.dropna()
        credit_requests_pos_map = credit_requests_pos_map.sort_values(
            ['visit_date', 'increment_coeff'], ascending=False)
        credit_requests_pos_map = credit_requests_pos_map.drop_duplicates(
            subset=['visit_date', 'pos_code'], keep='first')
        prepared_ts_data = pos_product_ts_data.groupby(
            level='pos_product').progress_apply(preprocess_ml,
                                                pos_visit_plan_data=pos_visit_plan_data,
                                                inflated_demand_marker_data=date_analysis_data,
                                                stock_collection_data=stock_collection_data,
                                                credit_requests_data=credit_requests_pos_map,
                                                pre_easter_effect_data=pre_easter_effect_data)
        print(prepared_ts_data.head())
        if not (test_mode or short_run_mode):
            prepared_ts_data.to_pickle('ml_modeller_input_{}_{}.pkl'.format(
                sales_office, current_part))
    else:
        # load already pickled solution for all features created
        prepared_ts_data = pd.read_pickle('ml_modeller_input_{}_{}.pkl'.format(
            sales_office, current_part))
        print(prepared_ts_data.head())

    prepared_ts_data.sort_values(['product_code'], inplace=True)
    # if short_run_mode:
        # product_list = prepared_ts_data['product_code'].unique()
        # product_list = product_list[:20]
        # prepared_ts_data = prepared_ts_data[
        #     prepared_ts_data['product_code'].isin(product_list)]

    prepared_product_data = prepared_ts_data.reset_index()
    prepared_product_data = prepared_product_data.drop(columns=['pos_product'])

    prepared_product_data = prepared_product_data.set_index(
        ['product_code', 'invoice_date']).sort_index()
    print(prepared_product_data.head())
    if test_mode or short_run_mode:
        prepared_product_data.to_csv(os.path.join(output_path,
                                                  "ml_modeller_input_sample.csv"))
    else:
        prepared_product_data.to_csv(
            os.path.join(output_path,
                         "ml_modeller_input_{}.csv".format(current_part)))

    preprocess_time = time.perf_counter()

    # product level rf modelling
    prepared_product_data.groupby(level=0, axis=0).progress_apply(
        high_level_model_handler,
        present_agent_map_data=visit_plan_test_data.reset_index(),
        shipment_data=shipment_data,
        inflated_demand_marker_data=date_analysis_data,
        shipment_split_data=shipment_split_data,
        sr_loading_data=sr_loading_data,
        sr_unloading_data=sr_unloading_data,
        pred_start_date=test_period_start_date,
        y_col='shipments',
        remove_first_month_from_training=remove_first_month_from_training,
        so_code=sales_office,
        output_path=output_path,
        run_log_path=run_log_path,
        process_pool=process_pool
    )
    norm_model_time = time.perf_counter()

    sparse_model_time = time.perf_counter()

    after_modelling_time = time.perf_counter()
    print("Total preprocessing time taken: {} seconds.".format(
        round(preprocess_time - start_time, 3)))
    print("Normal {} modelling time taken: {} seconds.".format(
        method, round(norm_model_time - preprocess_time, 3)))
    print("Post-model sparse modelling time taken: {} seconds.".format(
        round(sparse_model_time - norm_model_time, 3)))
    print("Full modelling time taken: {} seconds.".format(
        round(after_modelling_time - preprocess_time, 3)))

    print('Run complete.')
    print("Total time taken: {} seconds.".format(
        round(time.perf_counter() - start_time, 3)))
