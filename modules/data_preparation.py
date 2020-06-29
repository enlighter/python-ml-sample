import pandas as pd
import numpy as np
from tqdm import tqdm

import traceback

tqdm.pandas()
preprocess_constant_cols = ['product_code', 'pos_code', 'product_cat']


# get past shipment timeseries for each pos-product visit plan entry
# in prediction period
def extract_historical_timeseries(agentwise_visit_data: pd.DataFrame,
                                  shipment_data: pd.DataFrame,
                                  current_date: pd.Timestamp):

    agentwise_visit_df: pd.DataFrame = agentwise_visit_data.copy(deep=True)
    
    # current_agent = agentwise_visit_df.index.get_level_values(0)[0]
    # print(current_agent)
    
    agentwise_visit_df.index = agentwise_visit_df.index.droplevel()
    # print(agentwise_visit_df.head())
    
    pos_codes_to_cover = agentwise_visit_df.index

    # print(pos_codes_to_cover)
    shipment_df: pd.DataFrame = shipment_data.loc[:current_date]
    shipment_df = shipment_df.loc[shipment_df.pos_code.isin(
        pos_codes_to_cover)].copy()
    shipment_df['pos_product'] = shipment_df['pos_code'] + \
        '_' + shipment_df['product_code']
    shipment_df = shipment_df.reset_index().set_index(['invoice_date',
                                                       'pos_product'])
    shipment_df = shipment_df.drop(columns=['agent_code'])
    # print(shipment_df.head())

    return shipment_df


# extracts historical timeseries as per visit plan entries
# in prediction period
def prepare_data(full_visit_data: pd.DataFrame,
                 shipment_data: pd.DataFrame):
    try:
        full_visit_data_df: pd.DataFrame = full_visit_data.copy(deep=True)
        # print(full_visit_data_df)

        if len(full_visit_data_df) > 0:
            current_date = full_visit_data_df.index.get_level_values(0)[0]
            # print(current_date)
            full_visit_data_df.index = full_visit_data_df.index.droplevel()

            result_ts = full_visit_data_df.groupby(level=0,
                                                   axis=0).apply(extract_historical_timeseries,
                                                                 shipment_data,
                                                                 current_date)
            # Drop agent code as it is inconsequential here and is adding to confusion
            result_ts.index = result_ts.index.droplevel()

            # result_ts['so_code'] = full_visit_data_df.so_code.values[0]

            # print(result_ts.head())
            # exit()
            return result_ts

    except Exception as ex:
        print(ex)
        print(traceback.format_exc())


# pos-product past timeseries-wise model preprocess step
# for ml with many derived features,
# prepares data not just according to past actual shipments
# but also including all past planned shipments without actual shipment
def preprocess_ml(pos_pro_data: pd.DataFrame,
                  pos_visit_plan_data: pd.DataFrame,
                  inflated_demand_marker_data: pd.DataFrame,
                  stock_collection_data: pd.DataFrame,
                  credit_requests_data: pd.DataFrame,
                  pre_easter_effect_data: pd.DataFrame) -> pd.DataFrame:
    pos_pro_df: pd.DataFrame = pos_pro_data.copy()
    inflated_demand_marker_df = inflated_demand_marker_data.copy()

    current_combination = pos_pro_df.index.get_level_values('pos_product')[0]
    # print(current_combination)
    pos_pro_df = pos_pro_df.reset_index(level='pos_product', drop=True)

    last_visit_date = pos_pro_df.index.max()
    # print(last_visit_date)
    # loc returns Series if only one matching present
    # to always return DataFrame, loc parameter has been encapsulated in a list
    pos_pro_df = pos_pro_df.loc[[last_visit_date]]

    current_pos_code = pos_pro_df['pos_code'].iloc[0]
    visit_plan_dates = pos_visit_plan_data.loc[
        pos_visit_plan_data.index == current_pos_code]
    visit_plan_dates = visit_plan_dates.reset_index().set_index('visit_date')
    visit_plan_dates = visit_plan_dates[~visit_plan_dates.index.duplicated(keep='last')]
    visit_plan_dates['plan_date'] = True

    current_product_code = pos_pro_df['product_code'].iloc[0]
    stock_df = stock_collection_data.loc[
        (stock_collection_data['pos_code'] == current_pos_code) &
        (stock_collection_data['product_code'] == current_product_code)]
    stock_df = stock_df.set_index('invoice_date')

    credit_requests_df = credit_requests_data.loc[
        credit_requests_data['pos_code'] == current_pos_code]
    credit_requests_df = credit_requests_df.set_index('visit_date')

    pos_pro_df.set_index(["invoice_date"], inplace=True)
    pos_pro_df.sort_index(inplace=True)
    assert pos_pro_df['return'].dtype == 'bool'
    pos_pro_df.rename(columns={'quantity': 'shipments'}, inplace=True)

    # features formed before padding with 0 shipment dates
    first_nonzero_sale_date = pos_pro_df.index[0]
    if len(pos_pro_df) < 2:
        second_nonzero_sale_date = None
    else:
        second_nonzero_sale_date = pos_pro_df.index[1]
    if len(pos_pro_df) < 3:
        third_nonzero_sale_date = None
    else:
        third_nonzero_sale_date = pos_pro_df.index[2]
    if len(pos_pro_df) < 4:
        fourth_nonzero_sale_date = None
    else:
        fourth_nonzero_sale_date = pos_pro_df.index[3]
    if len(pos_pro_df) < 5:
        fifth_nonzero_sale_date = None
    else:
        fifth_nonzero_sale_date = pos_pro_df.index[4]
    if len(pos_pro_df) < 6:
        sixth_nonzero_sale_date = None
    else:
        sixth_nonzero_sale_date = pos_pro_df.index[5]
    nonzero_shipment_dates: pd.Series = pd.Series(pos_pro_df.index)
    pos_pro_df.at[pos_pro_df.index[0], 'is_first_nonzero_sale_date'] = True
    pos_pro_df.at[pos_pro_df.index[-1], 'is_last_sale_date'] = True
    pos_pro_df['nonzero_Shipments_1'] = pos_pro_df['shipments'].shift(1)
    overflow_shipment_1 = pos_pro_df['shipments'].iloc[-1]
    pos_pro_df['nonzero_Shipments_2'] = pos_pro_df['nonzero_Shipments_1'].shift(1)
    overflow_shipment_2 = pos_pro_df['nonzero_Shipments_1'].iloc[-1]
    pos_pro_df['nonzero_Shipments_3'] = pos_pro_df['nonzero_Shipments_2'].shift(1)
    overflow_shipment_3 = pos_pro_df['nonzero_Shipments_2'].iloc[-1]
    pos_pro_df['nonzero_Shipments_4'] = pos_pro_df['nonzero_Shipments_3'].shift(1)
    overflow_shipment_4 = pos_pro_df['nonzero_Shipments_3'].iloc[-1]
    pos_pro_df['nonzero_Shipments_5'] = pos_pro_df['nonzero_Shipments_4'].shift(1)
    overflow_shipment_5 = pos_pro_df['nonzero_Shipments_4'].iloc[-1]
    pos_pro_df['nonzero_Shipments_6'] = pos_pro_df['nonzero_Shipments_5'].shift(1)
    overflow_shipment_6 = pos_pro_df['nonzero_Shipments_5'].iloc[-1]
    pos_pro_df['days_since_last_nonzero_sale'] = nonzero_shipment_dates.diff().dt.days.values
    pos_pro_df['days_since_last_nonzero_sale_1'] = (nonzero_shipment_dates.shift(1)
                                                    - nonzero_shipment_dates.shift(2)).dt.days.values
    overflow_days_since_last_nonzero_sale_1 = pos_pro_df['days_since_last_nonzero_sale'].iloc[-1]
    pos_pro_df['days_since_last_nonzero_sale_2'] = (nonzero_shipment_dates.shift(2)
                                                    - nonzero_shipment_dates.shift(3)).dt.days.values
    overflow_days_since_last_nonzero_sale_2 = pos_pro_df[
        'days_since_last_nonzero_sale'].shift(1).iloc[-1]
    pos_pro_df['days_since_last_nonzero_sale_3'] = (nonzero_shipment_dates.shift(3)
                                                    - nonzero_shipment_dates.shift(4)).dt.days.values
    overflow_days_since_last_nonzero_sale_3 = pos_pro_df[
        'days_since_last_nonzero_sale'].shift(2).iloc[-1]
    # print(pos_pro_df.head(10))

    try:
        assert len(pos_pro_df[pos_pro_df.index.duplicated(keep='last')]) == 0
    except AssertionError:
        with open('duplicate_pos_pro.txt', 'a') as efile:
            efile.write(current_combination + '\n')

    new_index = visit_plan_dates.index.union(pos_pro_df.index)
    new_index = new_index.union([last_visit_date])
    pos_pro_df = pos_pro_df.reindex(new_index)

    global preprocess_constant_cols
    pos_pro_df.loc[:, preprocess_constant_cols] = \
        pos_pro_df.loc[:, preprocess_constant_cols].fillna(method='ffill')
    pos_pro_df.loc[:, preprocess_constant_cols] = \
        pos_pro_df.loc[:, preprocess_constant_cols].fillna(method='bfill')
    pos_pro_df.index.names = ['invoice_date']

    pos_pro_df['is_first_nonzero_sale_date'] = pos_pro_df['is_first_nonzero_sale_date'].fillna(False)
    pos_pro_df['is_last_sale_date'] = pos_pro_df['is_last_sale_date'].fillna(False)

    pos_pro_df['nonzero_Shipments_1'] = pos_pro_df['nonzero_Shipments_1'].fillna(method='bfill')
    pos_pro_df.loc[pos_pro_df.index <= first_nonzero_sale_date,
                   'nonzero_Shipments_1'] = 0
    pos_pro_df['nonzero_Shipments_1'] = pos_pro_df['nonzero_Shipments_1'].fillna(overflow_shipment_1)
    pos_pro_df.loc[pos_pro_df.index <= first_nonzero_sale_date,
                   'days_since_last_nonzero_sale'] = -1
    if second_nonzero_sale_date:
        pos_pro_df['nonzero_Shipments_2'] = pos_pro_df['nonzero_Shipments_2'].fillna(method='bfill')
        pos_pro_df.loc[pos_pro_df.index <= second_nonzero_sale_date,
                       'nonzero_Shipments_2'] = 0
        pos_pro_df['nonzero_Shipments_2'] = pos_pro_df['nonzero_Shipments_2'].fillna(overflow_shipment_2)
        pos_pro_df['days_since_last_nonzero_sale_1'] = \
            pos_pro_df['days_since_last_nonzero_sale_1'].fillna(method='bfill')
        pos_pro_df.loc[pos_pro_df.index <= second_nonzero_sale_date,
                       'days_since_last_nonzero_sale_1'] = -1
        pos_pro_df['days_since_last_nonzero_sale_1'] = \
            pos_pro_df['days_since_last_nonzero_sale_1'].fillna(overflow_days_since_last_nonzero_sale_1)
    else:
        pos_pro_df['nonzero_Shipments_2'] = 0
        pos_pro_df['days_since_last_nonzero_sale_1'] = -1
    if third_nonzero_sale_date:
        pos_pro_df['nonzero_Shipments_3'] = pos_pro_df['nonzero_Shipments_3'].fillna(method='bfill')
        pos_pro_df.loc[pos_pro_df.index <= third_nonzero_sale_date,
                       'nonzero_Shipments_3'] = 0
        pos_pro_df['nonzero_Shipments_3'] = pos_pro_df['nonzero_Shipments_3'].fillna(overflow_shipment_3)
        pos_pro_df['days_since_last_nonzero_sale_2'] = \
            pos_pro_df['days_since_last_nonzero_sale_2'].fillna(method='bfill')
        pos_pro_df.loc[pos_pro_df.index <= third_nonzero_sale_date,
                       'days_since_last_nonzero_sale_2'] = -1
        pos_pro_df['days_since_last_nonzero_sale_2'] = \
            pos_pro_df['days_since_last_nonzero_sale_2'].fillna(overflow_days_since_last_nonzero_sale_2)
    else:
        pos_pro_df['nonzero_Shipments_3'] = 0
        pos_pro_df['days_since_last_nonzero_sale_2'] = -1
    if fourth_nonzero_sale_date:
        pos_pro_df['nonzero_Shipments_4'] = pos_pro_df['nonzero_Shipments_4'].fillna(method='bfill')
        pos_pro_df.loc[pos_pro_df.index <= fourth_nonzero_sale_date,
                       'nonzero_Shipments_4'] = 0
        pos_pro_df['nonzero_Shipments_4'] = pos_pro_df['nonzero_Shipments_4'].fillna(overflow_shipment_4)
        pos_pro_df['days_since_last_nonzero_sale_3'] = \
            pos_pro_df['days_since_last_nonzero_sale_3'].fillna(method='bfill')
        pos_pro_df.loc[pos_pro_df.index <= fourth_nonzero_sale_date,
                       'days_since_last_nonzero_sale_3'] = -1
        pos_pro_df['days_since_last_nonzero_sale_3'] = \
            pos_pro_df['days_since_last_nonzero_sale_3'].fillna(overflow_days_since_last_nonzero_sale_3)
    else:
        pos_pro_df['nonzero_Shipments_4'] = 0
        pos_pro_df['days_since_last_nonzero_sale_3'] = -1
    if fifth_nonzero_sale_date:
        pos_pro_df['nonzero_Shipments_5'] = pos_pro_df['nonzero_Shipments_5'].fillna(method='bfill')
        pos_pro_df.loc[pos_pro_df.index <= fifth_nonzero_sale_date,
                       'nonzero_Shipments_5'] = 0
        pos_pro_df['nonzero_Shipments_5'] = pos_pro_df['nonzero_Shipments_5'].fillna(overflow_shipment_5)
    else:
        pos_pro_df['nonzero_Shipments_5'] = 0
    if sixth_nonzero_sale_date:
        pos_pro_df['nonzero_Shipments_6'] = pos_pro_df['nonzero_Shipments_6'].fillna(method='bfill')
        pos_pro_df['nonzero_Shipments_6'] = pos_pro_df['nonzero_Shipments_6'].fillna(overflow_shipment_6)
        pos_pro_df.loc[pos_pro_df.index <= sixth_nonzero_sale_date,
                       'nonzero_Shipments_6'] = 0
    else:
        pos_pro_df['nonzero_Shipments_6'] = 0
    pos_pro_df['return_1'] = pos_pro_df['return'].shift()
    pos_pro_df['return_1'] = pos_pro_df['return_1'].fillna(False)
    pos_pro_df['shipments'] = pos_pro_df['shipments'].fillna(0)
    pos_pro_df['promo_id'] = pos_pro_df['promo_id'].fillna(0).astype('int')

    pos_pro_df.at[pos_pro_df.index[0], 'is_first_plan_date'] = True
    pos_pro_df['is_first_plan_date'] = pos_pro_df['is_first_plan_date'].fillna(False)
    pos_pro_df['days_since_first_sale'] = (pos_pro_df.index
                                           - pos_pro_df.index[0]).days
    pos_pro_df['days_since_first_nonzero_sale'] = (pos_pro_df.index
                                                   - first_nonzero_sale_date).days
    pos_pro_df.loc[pos_pro_df.index < first_nonzero_sale_date,
                   'days_since_first_nonzero_sale'] = -1
    pos_pro_df['days_since_last_sale'] = pos_pro_df['days_since_first_sale'].diff().fillna(0)
    pos_pro_df['stock'] = stock_df['stock']
    pos_pro_df['stock'] = pos_pro_df['stock'].fillna(-999)
    pos_pro_df['week'] = pos_pro_df.index.week
    pos_pro_df['month'] = pos_pro_df.index.month
    pos_pro_df['is_odd_month'] = pos_pro_df['month'].apply(lambda x: x & 1)
    pos_pro_df['quarter'] = pos_pro_df.index.quarter
    pos_pro_df['year'] = pos_pro_df.index.year
    # This is the i-th year this pos is buying this product
    pos_pro_df['year_of_engagement'] = (pos_pro_df['year']
                                        - pos_pro_df['year'].min() + 1)
    pos_pro_df['VisitPlanWeek'] = inflated_demand_marker_df['VisitPlanWeek']
    pos_pro_df['DayNumberOfWeek'] = inflated_demand_marker_df['DayNumberOfWeek']
    pos_pro_df['IsWorkingDate'] = inflated_demand_marker_df['IsWorkingDate']
    pos_pro_df['weekday_holiday'] = inflated_demand_marker_df['WeekdayHoliday']
    pos_pro_df['days_since_price_chg_ann'] = inflated_demand_marker_df['daysSincePriceChgAnn']
    pos_pro_df['days_from_price_chg'] = inflated_demand_marker_df['daysFromPriceChg']
    pos_pro_df['PricelistChg'] = inflated_demand_marker_df['PricelistChg']
    pos_pro_df['double_sell'] = visit_plan_dates['double_sell']
    pos_pro_df['double_sell'] = pos_pro_df['double_sell'].fillna(False)
    pos_pro_df['triple_sell'] = visit_plan_dates['triple_sell']
    pos_pro_df['triple_sell'] = pos_pro_df['triple_sell'].fillna(False)
    # For ER
    pos_pro_df['isVisitPlan'] = visit_plan_dates['plan_date']
    pos_pro_df['isVisitPlan'] = pos_pro_df['isVisitPlan'].fillna(False)

    pos_pro_df['pre_nonreplacement_holiday'] = \
        visit_plan_dates['pre_nonreplacement_holiday']
    pos_pro_df['pre_nonreplacement_holiday'] = pos_pro_df['pre_nonreplacement_holiday'].fillna(False)
    pos_pro_df.loc[
        (pos_pro_df['double_sell'] | pos_pro_df['triple_sell']),
        'pre_nonreplacement_holiday'] = False
    pos_pro_df['credit_request_coeff'] = credit_requests_df['increment_coeff']
    pos_pro_df['credit_request_coeff'] = pos_pro_df['credit_request_coeff'].fillna(0)
    pos_pro_df['credit_request_type'] = credit_requests_df['request_type']
    pos_pro_df['credit_request_type'] = pos_pro_df['credit_request_type'].fillna('nill')
    pos_pro_df['days_from_easter'] = pre_easter_effect_data['days_from_easter']
    pos_pro_df['days_from_easter'] = pos_pro_df['days_from_easter'].fillna(-1)
    pos_pro_df['isMissedPlan'] = ((pos_pro_df['stock'] == -999)
                                  & (pos_pro_df['shipments'] == 0)
                                  & pos_pro_df['isVisitPlan'])
    pos_pro_df['is_zero_sale'] = (pos_pro_df['shipments'] == 0)
    # Convert to sparse then query index to find block locations
    is_nonstart_zero: pd.Series = pos_pro_df.loc[
        pos_pro_df['days_since_first_nonzero_sale'] != -1,
        'is_zero_sale']
    temp_ts: pd.SparseSeries = is_nonstart_zero.to_sparse(fill_value=False)
    block_locs = zip(temp_ts.sp_index.blocs, temp_ts.sp_index.blengths)
    # Map the sparse blocks back to the dense timeseries
    blocks = [is_nonstart_zero.iloc[start-1:(start + length)] for (start, length) in block_locs]
    blocks = [pd.Series((block.index - block.index[0]),
                        index=block.index).dt.days.iloc[1:]
              for block in blocks if not block.empty]
    # Map the days since last nonzero sale blocks back to original df
    for block in blocks:
        pos_pro_df.loc[block.index, 'days_since_last_nonzero_sale'] = block

    pos_pro_df['shipments_1'] = pos_pro_df['shipments'].shift(1).fillna(0)
    pos_pro_df['days_since_last_sale_1'] = pos_pro_df['days_since_last_sale'].shift(1).fillna(-1)
    pos_pro_df['double_sell_1'] = pos_pro_df['double_sell'].shift(1).fillna(False)
    pos_pro_df.loc[
        (pos_pro_df['double_sell'] | pos_pro_df['triple_sell']), 'double_sell_1'] = False
    pos_pro_df['triple_sell_1'] = pos_pro_df['triple_sell'].shift(1).fillna(False)
    pos_pro_df.loc[
        (pos_pro_df['double_sell'] | pos_pro_df['triple_sell']), 'triple_sell_1'] = False
    pos_pro_df['isVisitPlan_1'] = pos_pro_df['isVisitPlan'].shift(1).fillna(False)  # After ER
    pos_pro_df['PricelistChg_1'] = pos_pro_df['PricelistChg'].shift(1).fillna(False)
    pos_pro_df['return_1'] = pos_pro_df['return'].shift(1).fillna(False)  # After return
    pos_pro_df['isMissedPlan_1'] = pos_pro_df['isMissedPlan'].shift(1).fillna(False)
    pos_pro_df['stock_1'] = pos_pro_df['stock'].shift(1).fillna(-999)
    pos_pro_df['is_zero_sale_1'] = pos_pro_df['is_zero_sale'].shift(1).fillna(False)
    is_zero_1: pd.Series = pos_pro_df['is_zero_sale_1']
    # Convert to sparse then query index to find block locations
    temp_ts: pd.SparseSeries = is_zero_1.to_sparse(fill_value=False)
    block_locs = zip(temp_ts.sp_index.blocs, temp_ts.sp_index.blengths)
    # Map the sparse blocks back to the dense timeseries
    blocks = [is_zero_1.iloc[start:(start + length)] for (start, length) in block_locs]
    blocks = [block.astype(int).cumsum() for block in blocks if not block.empty]
    # Map the cumsum blocks back to original df
    pos_pro_df['num_consecutive_zero_sales'] = 0
    for block in blocks:
        pos_pro_df.loc[block.index, 'num_consecutive_zero_sales'] = block
    is_nonzero_1: pd.Series = (pos_pro_df['shipments'] != 0).shift(1).fillna(False)
    # Convert to sparse then query index to find block locations
    temp_ts: pd.SparseSeries = is_nonzero_1.to_sparse(fill_value=False)
    block_locs = zip(temp_ts.sp_index.blocs, temp_ts.sp_index.blengths)
    # Map the sparse blocks back to the dense timeseries
    blocks = [is_nonzero_1.iloc[start:(start + length)] for (start, length) in block_locs]
    blocks = [block.astype(int).cumsum() for block in blocks if not block.empty]
    # Map the cumsum blocks back to original df
    pos_pro_df['num_consecutive_nonzero_sales'] = 0
    for block in blocks:
        pos_pro_df.loc[block.index, 'num_consecutive_nonzero_sales'] = block

    pos_pro_df['double_sell_lead_1'] = pos_pro_df[
        'double_sell'].shift(-1).fillna(False)  # Before double sell
    pos_pro_df.loc[
        (pos_pro_df['double_sell'] | pos_pro_df['triple_sell']
         | pos_pro_df['double_sell_1'] | pos_pro_df['triple_sell_1']),
        'double_sell_lead_1'] = False
    pos_pro_df['triple_sell_lead_1'] = pos_pro_df[
        'triple_sell'].shift(-1).fillna(False)  # Before triple sell
    pos_pro_df.loc[
        (pos_pro_df['double_sell'] | pos_pro_df['triple_sell']
         | pos_pro_df['double_sell_1'] | pos_pro_df['triple_sell_1']),
        'triple_sell_lead_1'] = False

    pos_pro_df['shipments_2'] = pos_pro_df['shipments_1'].shift(1).fillna(0)
    pos_pro_df['days_since_last_sale_2'] = pos_pro_df['days_since_last_sale_1'].shift(1).fillna(-1)
    pos_pro_df['double_sell_2'] = pos_pro_df['double_sell_1'].shift(1).fillna(False)
    pos_pro_df.loc[
        (pos_pro_df['double_sell'] | pos_pro_df['triple_sell']), 'double_sell_2'] = False
    pos_pro_df['triple_sell_2'] = pos_pro_df['triple_sell_1'].shift(1).fillna(False)
    pos_pro_df.loc[
        (pos_pro_df['double_sell'] | pos_pro_df['triple_sell']), 'triple_sell_2'] = False
    pos_pro_df['isVisitPlan_2'] = pos_pro_df['isVisitPlan_1'].shift(1).fillna(False)
    pos_pro_df['isMissedPlan_2'] = pos_pro_df['isMissedPlan_1'].shift(1).fillna(False)
    pos_pro_df['stock_2'] = pos_pro_df['stock_1'].shift(1).fillna(-999)
    pos_pro_df['is_zero_sale_2'] = pos_pro_df['is_zero_sale_1'].shift(1).fillna(False)

    pos_pro_df['shipments_3'] = pos_pro_df['shipments_2'].shift(1).fillna(0)
    pos_pro_df['days_since_last_sale_3'] = pos_pro_df['days_since_last_sale_2'].shift(1).fillna(-1)
    pos_pro_df['stock_3'] = pos_pro_df['stock_2'].shift(1).fillna(-999)
    pos_pro_df['is_zero_sale_3'] = pos_pro_df['is_zero_sale_2'].shift(1).fillna(False)
    pos_pro_df['shipments_4'] = pos_pro_df['shipments_3'].shift(1).fillna(0)
    pos_pro_df['stock_4'] = pos_pro_df['stock_3'].shift(1).fillna(-999)
    pos_pro_df['shipments_5'] = pos_pro_df['shipments_4'].shift(1).fillna(0)
    pos_pro_df['stock_5'] = pos_pro_df['stock_4'].shift(1).fillna(-999)
    pos_pro_df['shipments_6'] = pos_pro_df['shipments_5'].shift(1).fillna(0)
    pos_pro_df['stock_6'] = pos_pro_df['stock_5'].shift(1).fillna(-999)

    past_3_shipment_labels = ['shipments_{}'.format(i) for i in range(1, 4)]
    past_6_shipment_labels = ['shipments_{}'.format(i) for i in range(1, 7)]
    past_3_shipments_interval_labels = ['days_since_last_sale_{}'.format(i) for i in range(1, 4)]
    past_3_stock_labels = ['stock_{}'.format(i) for i in range(1, 4)]
    past_6_stock_labels = ['stock_{}'.format(i) for i in range(1, 7)]
    past_3_nonzero_shipment_labels = ['nonzero_Shipments_{}'.format(i) for i in range(1, 4)]
    past_6_nonzero_shipment_labels = ['nonzero_Shipments_{}'.format(i) for i in range(1, 7)]
    past_3_nonzero_shipments_interval_labels = ['days_since_last_nonzero_sale_{}'.format(i)
                                                for i in range(1, 4)]
    pos_pro_df['ma_3'] = pos_pro_df[past_3_shipment_labels].mean(axis=1)
    pos_pro_df['ma_6'] = pos_pro_df[past_6_shipment_labels].mean(axis=1)
    pos_pro_df['mm_3'] = pos_pro_df[past_3_shipment_labels].median(axis=1)
    pos_pro_df['mm_6'] = pos_pro_df[past_6_shipment_labels].median(axis=1)
    pos_pro_df['ma_interval_3'] = pos_pro_df[past_3_shipments_interval_labels].mean(axis=1)
    pos_pro_df['ma_Stock_3'] = pos_pro_df[past_3_stock_labels].mean(axis=1)
    pos_pro_df['ma_Stock_6'] = pos_pro_df[past_6_stock_labels].mean(axis=1)
    pos_pro_df['ma_nonzero_3'] = pos_pro_df[past_3_nonzero_shipment_labels].mean(axis=1)
    pos_pro_df['ma_nonzero_6'] = pos_pro_df[past_6_nonzero_shipment_labels].mean(axis=1)
    pos_pro_df['mm_nonzero_3'] = pos_pro_df[past_3_nonzero_shipment_labels].median(axis=1)
    pos_pro_df['mm_nonzero_6'] = pos_pro_df[past_6_nonzero_shipment_labels].median(axis=1)
    pos_pro_df['ma_nonzero_interval_3'] = \
        pos_pro_df[past_3_nonzero_shipments_interval_labels].mean(axis=1)

    pos_pro_df['Chg_Shipments_1_from_ma_3'] = pos_pro_df['shipments_1'] - pos_pro_df['ma_3']
    pos_pro_df['Chg_Shipments_1_from_ma_6'] = pos_pro_df['shipments_1'] - pos_pro_df['ma_6']
    pos_pro_df['Chg_Shipments_1_from_mm_3'] = pos_pro_df['shipments_1'] - pos_pro_df['mm_3']
    pos_pro_df['Chg_Shipments_1_from_mm_6'] = pos_pro_df['shipments_1'] - pos_pro_df['mm_6']
    pos_pro_df['Chg_pct_Shipments_1_from_Shipments_2'] = ((pos_pro_df['shipments_1']
                                                           - pos_pro_df['shipments_2'])
                                                          / pos_pro_df['shipments_2']).clip(-5, 5)
    pos_pro_df['Chg_pct_Shipments_1_from_Shipments_2'] = \
        pos_pro_df['Chg_pct_Shipments_1_from_Shipments_2'].fillna(-999)
    pos_pro_df['Chg_Stock_1_from_ma_Stock_3'] = pos_pro_df['stock_1'] - pos_pro_df['ma_Stock_3']
    pos_pro_df['Chg_Stock_1_from_ma_Stock_6'] = pos_pro_df['stock_1'] - pos_pro_df['ma_Stock_6']
    pos_pro_df['Chg_pct_Stock_1_from_Stock_2'] = ((pos_pro_df['stock_1']
                                                   - pos_pro_df['stock_2'])
                                                  / pos_pro_df['stock_2']).clip(-5, 5)
    pos_pro_df['Chg_pct_Stock_1_from_Stock_2'] = \
        pos_pro_df['Chg_pct_Stock_1_from_Stock_2'].fillna(-999)
    pos_pro_df['Chg_nonzero_Shipments_1_from_ma_nonzero_3'] = \
        pos_pro_df['nonzero_Shipments_1'] - pos_pro_df['ma_nonzero_3']
    pos_pro_df['Chg_nonzero_Shipments_1_from_ma_nonzero_6'] = \
        pos_pro_df['nonzero_Shipments_1'] - pos_pro_df['ma_nonzero_6']
    pos_pro_df['Chg_nonzero_Shipments_1_from_mm_nonzero_3'] = \
        pos_pro_df['nonzero_Shipments_1'] - pos_pro_df['mm_nonzero_3']
    pos_pro_df['Chg_nonzero_Shipments_1_from_mm_nonzero_6'] = \
        pos_pro_df['nonzero_Shipments_1'] - pos_pro_df['mm_nonzero_6']
    pos_pro_df['Chg_pct_nonzero_Shipments_1_from_nonzero_Shipments_2'] = \
        ((pos_pro_df['nonzero_Shipments_1'] - pos_pro_df['nonzero_Shipments_2'])
         / pos_pro_df['nonzero_Shipments_2']).clip(-5, 5)
    pos_pro_df['Chg_pct_nonzero_Shipments_1_from_nonzero_Shipments_2'] = \
        pos_pro_df['Chg_pct_nonzero_Shipments_1_from_nonzero_Shipments_2'].fillna(-999)

    pos_pro_df = pos_pro_df.loc[(pos_pro_df['IsWorkingDate'] == 1) &
                                (pos_pro_df['DayNumberOfWeek'] <= 5)]
    pos_pro_df.drop(columns=['return', 'IsWorkingDate', 'PricelistChg'],
                    inplace=True)

    # try:
    #     assert not pos_pro_df.isnull().any().any()
    # except AssertionError as ae:
    #     null_track = pos_pro_df.isnull().any()
    #     print(null_track[null_track])
    #     pos_pro_df.to_csv('temp.csv')
    #     exit()
    pos_pro_df.fillna(0, inplace=True)
    # print(pos_pro_df.head())

    return pos_pro_df
