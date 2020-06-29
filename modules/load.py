import pandas as pd
from tqdm import tqdm

from config import visit_plan_raw_data_path, agent_replacements_raw_data_path
from config import date_analysis_raw_data_drop_cols, credit_requests_raw_data_drop_cols
from config import sr_loading_raw_data_drop_cols, sr_unloading_raw_data_drop_cols
from config import take_shipment_data_cols
from config import take_future_visit_plan_data_cols, take_pos_visit_plan_data_cols
from config import take_date_analysis_data_cols, take_credit_req_data_cols
from config import shipment_raw_data_path, pos_master_raw_data_path
from config import sr_loading_raw_data_path, sr_unloading_raw_data_path
from config import stock_collection_raw_data_path
from config import esgo_date_analysis_raw_data_path, credit_requests_raw_data_path

tqdm.pandas()


# Reconcile agent replacements, holiday adjustments in visit plan data
def __visit_plan_reconciliation(visit_plan_data: pd.DataFrame,
                                weekday_holiday_dates: list) -> pd.DataFrame:
    visit_plan_df: pd.DataFrame = visit_plan_data.reset_index()
    visit_plan_df = visit_plan_df.sort_values(['visit_date'])
    visit_plan_df.loc[
        visit_plan_df['visit_date'].isin(weekday_holiday_dates),
        'weekday_holiday'] = True
    visit_plan_df['weekday_holiday'] = visit_plan_df['weekday_holiday'].fillna(False)

    def __create_new_markers(pos_plan_df: pd.DataFrame) -> pd.DataFrame:
        current_pos = pos_plan_df['pos_code'].iloc[0]
        # print(current_pos)
        pos_plan_df['type_lead_1'] = pos_plan_df['type'].shift(-1)
        pos_plan_df.loc[
            pos_plan_df['type_lead_1'] == 'Holiday without replacement',
            'pre_nonreplacement_holiday'] = True
        pos_plan_df['pre_nonreplacement_holiday'] = \
            pos_plan_df['pre_nonreplacement_holiday'].fillna(False)
        pos_plan_df['double_sell'] = pos_plan_df['weekday_holiday'].shift(-1).fillna(False)
        pos_plan_df['weekday_holiday(-2)'] = pos_plan_df['weekday_holiday'].shift(-2).fillna(False)
        pos_plan_df['triple_sell'] = (pos_plan_df['double_sell']
                                      & pos_plan_df['weekday_holiday(-2)'])
        pos_plan_df.loc[pos_plan_df['triple_sell'], 'double_sell'] = False
        return pos_plan_df.drop(columns=['type_lead_1', 'weekday_holiday(-2)'])

    visit_plan_df = visit_plan_df.groupby(['pos_code']).progress_apply(
        __create_new_markers)
    # Holiday without replacement means no agent actually went there
    visit_plan_df = visit_plan_df.loc[
        visit_plan_df['type'] != 'Holiday without replacement']
    # Take only non-weekday-holiday dates
    visit_plan_df = visit_plan_df.loc[~visit_plan_df['weekday_holiday']]
    visit_plan_df.loc[visit_plan_df['type'] == 'Replacement', 'agent_code'] = \
        visit_plan_df.loc[visit_plan_df['type'] == 'Replacement',
                          'backup_agent_code'].astype(int)

    return visit_plan_df


# Add derived columns as required to date_analysis_data
def __date_analysis_additions(date_analysis_data: pd.DataFrame) -> pd.DataFrame:
    # manual correction according to seen data
    date_analysis_data.loc[
        [pd.Timestamp(day=7, month=12, year=2018)],
        'IsWorkingDate'] = 0
    date_analysis_data['WeekdayHoliday'] = ((date_analysis_data['DayNumberOfWeek'] <= 5)
                                            & (date_analysis_data['IsWorkingDate'] == 0))
    date_analysis_data['PriceChgPeriod'] = (date_analysis_data['PricelistChgAnn']
                                            | date_analysis_data['PricelistChg'])
    date_analysis_data['PriceChgPeriod'] = date_analysis_data['PriceChgPeriod'].cumsum()
    date_analysis_data['PriceChgEffect'] = \
        date_analysis_data['PriceChgPeriod'].apply(lambda x: x % 2 != 0)
    date_analysis_data['PriceChgPeriod'] = (date_analysis_data['PriceChgEffect']
                                            | date_analysis_data['PricelistChg'])
    # Convert to sparse then query index to find block locations
    temp_ts: pd.SparseSeries = date_analysis_data['PriceChgPeriod'].to_sparse(
        fill_value=False)
    block_locs = zip(temp_ts.sp_index.blocs, temp_ts.sp_index.blengths)
    # Map the sparse blocks back to the dense timeseries
    block_infos = [(date_analysis_data['PriceChgPeriod'].iloc[start:(start + length)],
                    length)
                   for (start, length) in block_locs]
    for series_block, length in block_infos:
        values = range(length)
        date_analysis_data.loc[series_block.index, 'daysSincePriceChgAnn'] = values
        date_analysis_data.loc[series_block.index, 'daysFromPriceChg'] = values[::-1]
    date_analysis_data['daysSincePriceChgAnn'] = \
        date_analysis_data['daysSincePriceChgAnn'].fillna(-1).astype(int)
    date_analysis_data['daysFromPriceChg'] = \
        date_analysis_data['daysFromPriceChg'].fillna(-1).astype(int)

    return date_analysis_data.drop(columns=['PricelistChgAnn',
                                            'PriceChgPeriod',
                                            'PriceChgEffect'])


# prepare shipment data into consumable format
def __prep_consumable_shipments_data(shipment_data: pd.DataFrame,
                                     train_period_shipments: pd.DataFrame) -> pd.DataFrame:
    shipment_records: pd.DataFrame = shipment_data.loc[shipment_data['return'] == 0].copy()
    shipments_train_agg: pd.DataFrame = train_period_shipments.groupby(
        ['invoice_date', 'product_code']).agg({'quantity': 'sum'})

    shipments_train_agg = shipments_train_agg.reset_index().groupby(
        ['product_code']).agg({'quantity': 'mean'})
    shipments_train_agg['contribution%'] = (shipments_train_agg['quantity']
                                            / shipments_train_agg['quantity'].sum())*100
    shipments_train_agg = shipments_train_agg.sort_values(['contribution%'])
    shipments_train_agg['contribution%_cumsum'] = shipments_train_agg['contribution%'].cumsum()
    # take products which make up 5% or a little more than 5% of the total quantity
    num_small_products = len(shipments_train_agg.loc[
                                 shipments_train_agg['contribution%_cumsum'] < 5])
    small_products = shipments_train_agg.index[:num_small_products+1]
    # pd.Series(small_products).to_csv('small_products_BW.csv')
    num_medium_products = len(shipments_train_agg.loc[
                                  (shipments_train_agg['contribution%_cumsum'] >= 5)
                                  & (shipments_train_agg['contribution%_cumsum'] < 20)])
    medium_products = shipments_train_agg.index[num_small_products+1
                                                : num_small_products+num_medium_products+1]
    # pd.Series(medium_products).to_csv('medium_products_BW.csv')

    shipment_records = shipment_records.groupby(['invoice_date', 'pos_code',
                                                 'product_code']).agg({'agent_code': 'first',
                                                                       'quantity': 'sum',
                                                                       'promo_id': 'first'})

    return_records: pd.DataFrame = shipment_data.loc[shipment_data['return'] == 1].copy()
    return_records = return_records.groupby(['invoice_date', 'pos_code',
                                             'product_code']).agg({'agent_code': 'first',
                                                                   'quantity': 'sum',
                                                                   'return': 'first'})
    return_records['return'] = return_records['return'].astype('bool')

    shipment_records['return'] = return_records['return']
    shipment_records['return'] = shipment_records['return'].fillna(False)
    shipment_records = shipment_records.reset_index()
    shipment_records.loc[
        shipment_records['product_code'].isin(small_products), 'product_cat'] = 'small'
    shipment_records['product_cat'] = shipment_records['product_cat'].fillna('large')
    shipment_records = shipment_records.set_index('invoice_date')
    return shipment_records.sort_index()


# flatten credit request data into a timeseries
def __prep_credit_req_data(credit_req_data: pd.DataFrame):
    credit_req_components: list = []

    def __flatten_credit_req_days(row: pd.DataFrame) -> None:
        start_date = row["PlannedStartDate"]
        end_date = row["PlannedClosedDate"]

        ret = pd.DataFrame(index=pd.date_range(start_date, end_date))
        ret.loc[:, 'request_type'] = row['ReqType']
        ret.loc[:, 'increment_coeff'] = row['IncrCoef']
        ret.loc[:, 'agent_code'] = row['AgentCode']

        credit_req_components.append(ret)

    credit_req_data.progress_apply(__flatten_credit_req_days, axis=1)
    processed_credit_req_data = pd.concat(credit_req_components)
    processed_credit_req_data['agent_code'] = processed_credit_req_data[
        'agent_code'].astype('int').astype('str')
    processed_credit_req_data.index.name = 'visit_date'
    processed_credit_req_data = processed_credit_req_data.reset_index()
    processed_credit_req_data = processed_credit_req_data.drop_duplicates().reset_index(drop=True)

    return processed_credit_req_data


# main method which loads the data and provides them for consumption
# by the project, in appropriate formats
def load_data(visit_plan_path: str,
              shipment_data_path: str,
              date_analysis_data_path: str,
              shipment_split_data_path: str,
              sr_loading_data_path: str,
              sr_unloading_data_path: str,
              stock_collection_data_path: str,
              credit_requests_data_path: str,
              pre_easter_effect_data_path: str,
              pred_start_date: pd.Timestamp) -> tuple:
    shipment_data: pd.DataFrame = pd.read_pickle(shipment_data_path)
    shipment_data = shipment_data[take_shipment_data_cols]
    shipment_data['agent_code'] = shipment_data['agent_code'].astype(str)
    shipment_data['pos_code'] = shipment_data['pos_code'].astype(str)
    products_to_consider = shipment_data.loc[
        shipment_data['invoice_date']
        >= pd.Timestamp(day=1, month=1, year=2018)]['product_code'].unique()
    products_to_consider = [product for product in products_to_consider if 'RLGC' not in product]
    shipment_data = shipment_data.loc[
        shipment_data['product_code'].isin(products_to_consider)]
    shipment_records = __prep_consumable_shipments_data(shipment_data,
                                                        shipment_data[
                                                            (shipment_data['invoice_date']
                                                             < pd.Timestamp(day=1,
                                                                            month=1,
                                                                            year=2019))
                                                        ])
    print('Completed loading shipment data.')

    date_analysis_data: pd.DataFrame = pd.read_pickle(date_analysis_data_path)
    date_analysis_data = date_analysis_data.set_index('Date')
    date_analysis_data = date_analysis_data[take_date_analysis_data_cols]
    date_analysis_data = __date_analysis_additions(date_analysis_data)
    print('Completed loading different date features data.')

    visit_plan_data: pd.DataFrame = pd.read_pickle(visit_plan_path)
    weekday_holiday_dates = date_analysis_data[date_analysis_data['WeekdayHoliday']].index.tolist()
    visit_plan_data = __visit_plan_reconciliation(visit_plan_data, weekday_holiday_dates)
    visit_plan_data['agent_code'] = visit_plan_data['agent_code'].astype(str)
    visit_plan_data['pos_code'] = visit_plan_data['pos_code'].astype(str)
    # remove any weekend dates, if any, from visit plan
    visit_plan_data = visit_plan_data[
        visit_plan_data['visit_date'].isin(
            date_analysis_data[date_analysis_data['IsWorkingDate'] == 1].index)]

    future_visit_plan_data = visit_plan_data.loc[
        visit_plan_data['visit_date'] >= pred_start_date,
        take_future_visit_plan_data_cols].copy()
    future_visit_plan_data.set_index(['visit_date', 'agent_code', 'pos_code'],
                                     inplace=True)
    future_visit_plan_data.sort_index(inplace=True)

    pos_visit_plan_data = visit_plan_data
    pos_visit_plan_data = pos_visit_plan_data[take_pos_visit_plan_data_cols]
    pos_visit_plan_data.set_index(['pos_code'], inplace=True)
    print('Completed loading visit plan related data.')

    shipment_split_data: pd.DataFrame = pd.read_pickle(shipment_split_data_path)
    shipment_split_data['agent_code'] = shipment_split_data['agent_code'].astype(str)
    shipment_split_data.set_index(['visit_date', 'agent_code', 'product_code'],
                                  inplace=True)
    print('Completed loading ER, non-ER shipment split data.')

    sr_loading_data: pd.DataFrame = pd.read_pickle(sr_loading_data_path)
    sr_loading_data['agent_code'] = sr_loading_data['agent_code'].astype(str)
    sr_loading_data.set_index(['visit_date', 'agent_code', 'product_code'],
                              inplace=True)
    print('Completed loading SR loading data.')

    sr_unloading_data: pd.DataFrame = pd.read_pickle(sr_unloading_data_path)
    sr_unloading_data['agent_code'] = sr_unloading_data['agent_code'].astype(str)
    sr_unloading_data.set_index(['visit_date', 'agent_code', 'product_code'],
                                inplace=True)
    print('Completed loading SR unloading data.')

    stock_collection_data: pd.DataFrame = pd.read_pickle(stock_collection_data_path)
    stock_collection_data.rename(columns={'stock_date': 'invoice_date'}, inplace=True)
    stock_collection_data['pos_code'] = stock_collection_data['pos_code'].astype(str)
    print('Completed loading stock collection data.')

    credit_requests_data: pd.DataFrame = pd.read_pickle(credit_requests_data_path)
    credit_requests_data = credit_requests_data[take_credit_req_data_cols]
    credit_requests_data = __prep_credit_req_data(credit_requests_data)
    print('Completed loading credit requests data.')

    pre_easter_effect_data: pd.DataFrame = pd.read_pickle(pre_easter_effect_data_path)
    pre_easter_effect_data = pre_easter_effect_data.set_index('dates')
    print('Completed loading pre easter effect data.')

    return future_visit_plan_data, pos_visit_plan_data, shipment_records, \
        date_analysis_data, shipment_split_data, stock_collection_data,\
        sr_loading_data, sr_unloading_data, credit_requests_data, pre_easter_effect_data


# read raw data from the input data files and clean/rationalize the data & store in pickle form
# meant to be a one-time step whenever the input data files change
def extract_n_pickle_raw_data(sales_office: str,
                              visit_plan_data_path: str,
                              shipment_data_path: str,
                              date_analysis_data_path: str,
                              shipment_split_path: str,
                              sr_loading_data_path: str,
                              sr_unloading_data_path: str,
                              stock_collection_data_path: str,
                              credit_requests_data_path: str) -> None:
    visit_plan_data: pd.DataFrame = pd.read_csv(visit_plan_raw_data_path, parse_dates=['Visit_Date'])
    visit_plan_data = visit_plan_data.loc[visit_plan_data['SO_CODE'] == sales_office]

    agent_replacements_data: pd.DataFrame = pd.read_csv(
        agent_replacements_raw_data_path,
        parse_dates=['RegistrationDate', 'DateFrom', 'DateTo']
    )
    agent_replacements_data = agent_replacements_data.loc[agent_replacements_data['SO_CODE'] == sales_office]
    agent_replacement_components: list = []

    shipment_data: pd.DataFrame = pd.read_csv(shipment_raw_data_path, parse_dates=['Invoice_Date'])
    shipment_data = shipment_data.loc[shipment_data['SO_CODE'] == sales_office]

    pos_master_data: pd.DataFrame = pd.read_csv(pos_master_raw_data_path)
    pos_master_data = pos_master_data.loc[pos_master_data['SO_CODE'] == sales_office]

    date_analysis_data: pd.DataFrame = pd.read_csv(esgo_date_analysis_raw_data_path, parse_dates=['Date'])

    sr_loading_data: pd.DataFrame = pd.read_csv(sr_loading_raw_data_path,
                                                parse_dates=['DocumentDate', 'CreatedDate'])

    sr_unloading_data: pd.DataFrame = pd.read_csv(sr_unloading_raw_data_path,
                                                  parse_dates=['Documentdate', 'createddate'])

    stock_collection_data: pd.DataFrame = pd.read_csv(stock_collection_raw_data_path, parse_dates=['Stock_Date'])
    stock_collection_data = stock_collection_data.loc[stock_collection_data['SO_CODE'] == sales_office]

    credit_requests_data: pd.DataFrame = pd.read_csv(
        credit_requests_raw_data_path,
        parse_dates=['PlannedStartDate', 'PlannedClosedDate', 'RegistrationDate']
    )
    credit_requests_data = credit_requests_data.loc[credit_requests_data['SO'] == sales_office]

    visit_plan_data.columns = [x.lower() for x in visit_plan_data.columns]
    shipment_data.columns = [x.lower() for x in shipment_data.columns]
    pos_master_data.columns = [x.lower() for x in pos_master_data.columns]
    sr_loading_data.columns = [x.lower() for x in sr_loading_data.columns]
    sr_unloading_data.columns = [x.lower() for x in sr_unloading_data.columns]
    stock_collection_data.columns = [x.lower() for x in stock_collection_data.columns]

    agent_preorder_pos_list: pd.Series = pos_master_data.loc[
        pos_master_data['delivered_by'] == 'Agent Preorder',
        'pos_code']

    def __flatten_agent_replacement_days(row: pd.DataFrame) -> None:
        start_date = row["DateFrom"]
        end_date = row["DateTo"]

        ret = pd.DataFrame(index=pd.date_range(start_date, end_date))
        ret.loc[:, 'agent_code'] = row['OriginalSalesAgent_Code']
        ret.loc[:, 'backup_agent_code'] = row['BackupSalesAgent_Code']
        ret.loc[:, 'type'] = row['Type']

        agent_replacement_components.append(ret)

    agent_replacements_data.progress_apply(__flatten_agent_replacement_days, axis=1)
    processed_agent_replacements_data: pd.DataFrame = pd.concat(agent_replacement_components)
    processed_agent_replacements_data.index.name = 'visit_date'
    processed_agent_replacements_data = processed_agent_replacements_data.reset_index()
    processed_agent_replacements_data = processed_agent_replacements_data.drop_duplicates()
    processed_agent_replacements_data['agent_code'] = processed_agent_replacements_data['agent_code'].astype('int')

    visit_plan_data = visit_plan_data.loc[~visit_plan_data['pos_code'].isin(agent_preorder_pos_list)]
    visit_plan_data = visit_plan_data.drop(columns=['id'])
    visit_plan_data = visit_plan_data.drop_duplicates()
    visit_plan_data['agent_code'] = visit_plan_data['agent_code'].astype('int')
    visit_plan_data['pos_code'] = visit_plan_data['pos_code'].astype('int')
    updated_visit_plan_data: pd.DataFrame = visit_plan_data.merge(
        processed_agent_replacements_data,
        how='left', on=['visit_date', 'agent_code']
    )

    shipment_data = shipment_data.loc[shipment_data['delivered_by'] != 'Transporter']
    shipment_data = shipment_data.drop(columns=['id'])
    shipment_data = shipment_data.drop_duplicates()
    shipment_data['agent_code'] = shipment_data['agent_code'].astype('int')
    shipment_data['pos_code'] = shipment_data['pos_code'].astype('int')
    agent_preorder_shipment_data = shipment_data.loc[shipment_data['pos_code'].isin(agent_preorder_pos_list)]
    shipment_data = shipment_data.loc[~shipment_data['pos_code'].isin(agent_preorder_pos_list)]

    agent_preorder_shipment_data = agent_preorder_shipment_data.loc[
        agent_preorder_shipment_data['return'] == 0]
    agent_preorder_shipment_data = agent_preorder_shipment_data.rename(
        columns={'invoice_date': 'visit_date', 'quantity': 'deduction'})
    agent_preorder_shipment_data = agent_preorder_shipment_data.groupby(
        ['visit_date', 'agent_code', 'product_code']).agg({'deduction': 'sum'})

    # merge and form the shipment split data
    def __get_shipment_split(slice_df: pd.DataFrame):
        vp_shipment = slice_df[slice_df['visit_order_day'].notna()]['quantity'].sum()
        non_vp_shipment = slice_df[slice_df['visit_order_day'].isna()]['quantity'].sum()
        return pd.DataFrame({'vp_shipment': vp_shipment,
                             'non_vp_shipment': non_vp_shipment}, index=[0])

    visit_plan_temp: pd.DataFrame = updated_visit_plan_data[
        ['visit_date', 'agent_code', 'pos_code',
         'visit_order_day', 'backup_agent_code', 'type']
    ].copy()
    visit_plan_temp.loc[visit_plan_temp['type'] == 'Replacement', 'agent_code'] = \
        visit_plan_temp.loc[visit_plan_temp['type'] == 'Replacement', 'backup_agent_code'].astype(int)
    visit_plan_temp = visit_plan_temp.drop(columns=['backup_agent_code', 'type'])
    visit_plan_temp = visit_plan_temp.drop_duplicates(
        subset=['visit_date', 'agent_code', 'pos_code'],
        keep='first'
    )
    shipment_temp: pd.DataFrame = shipment_data[
        ['invoice_date', 'agent_code', 'pos_code',
         'product_code', 'quantity', 'return']].rename(columns={'invoice_date': 'visit_date'})
    shipment_temp = shipment_temp[shipment_temp['return'] == 0]
    shipment_by_plan: pd.DataFrame = shipment_temp.merge(
        visit_plan_temp,
        how='left',
        on=['visit_date', 'agent_code', 'pos_code']
    )
    shipment_split_vp_data: pd.DataFrame = shipment_by_plan.groupby(
        ['visit_date', 'agent_code', 'product_code']).progress_apply(__get_shipment_split)
    shipment_split_vp_data.index = shipment_split_vp_data.index.droplevel(level=-1)
    shipment_split_vp_data = shipment_split_vp_data.reset_index()

    date_analysis_data = date_analysis_data.loc[
        date_analysis_data['Date'] <= pd.Timestamp(day=31, month=3, year=2019)]
    date_analysis_data = date_analysis_data.drop(columns=date_analysis_raw_data_drop_cols)

    sr_loading_data = sr_loading_data.drop(columns=sr_loading_raw_data_drop_cols)
    sr_loading_data.rename(columns={'documentdate': 'visit_date'}, inplace=True)
    sr_loading_data['agent_code'] = sr_loading_data['agent_code'].astype(int)
    sr_loading_data = sr_loading_data.set_index(['visit_date', 'agent_code',
                                                 'product_code', 'quantity',
                                                 'documentcode'])
    sr_loading_data = sr_loading_data[~sr_loading_data.index.duplicated(keep='first')]
    sr_loading_data = sr_loading_data.reset_index()
    sr_loading_data = sr_loading_data.groupby(['visit_date', 'agent_code',
                                               'product_code']).agg({'quantity': 'sum'})
    sr_loading_data = sr_loading_data.merge(agent_preorder_shipment_data,
                                            how='left',
                                            left_index=True, right_index=True)
    sr_loading_data['deduction'] = sr_loading_data['deduction'].fillna(0)
    sr_loading_data['quantity'] = sr_loading_data['quantity'] - sr_loading_data['deduction']
    sr_loading_data = sr_loading_data.drop(columns=['deduction'])
    sr_loading_data = sr_loading_data.reset_index()

    sr_unloading_data = sr_unloading_data.drop(columns=sr_unloading_raw_data_drop_cols)
    sr_unloading_data.rename(columns={'documentdate': 'visit_date'}, inplace=True)
    sr_unloading_data['agent_code'] = sr_unloading_data['agent_code'].astype(int)
    sr_unloading_data = sr_unloading_data.set_index(['visit_date', 'agent_code',
                                                     'product_code', 'quantity',
                                                     'documentcode'])
    sr_unloading_data = sr_unloading_data[~sr_unloading_data.index.duplicated(keep='first')]
    sr_unloading_data = sr_unloading_data.reset_index()
    sr_unloading_data = sr_unloading_data.groupby(['visit_date', 'agent_code',
                                                   'product_code']).agg({'quantity': 'sum'})
    sr_unloading_data = sr_unloading_data.reset_index()

    stock_collection_data = stock_collection_data.drop(columns=['id'])
    stock_collection_data = stock_collection_data.drop_duplicates()
    stock_collection_data = stock_collection_data.rename(columns={'quantity': 'stock'})
    stock_collection_data['pos_code'] = stock_collection_data['pos_code'].astype(int)
    stock_collection_data = stock_collection_data.groupby(
        [pd.Grouper(key='stock_date', freq='D'),
         'pos_code',
         'product_code']).agg({'stock': 'mean'})
    stock_collection_data = stock_collection_data.reset_index()

    credit_requests_data = credit_requests_data.drop_duplicates()
    credit_requests_data = credit_requests_data.drop(columns=credit_requests_raw_data_drop_cols)
    credit_requests_data['AgentCode'] = credit_requests_data['AgentCode'].astype(int)
    credit_requests_data = credit_requests_data.reset_index(drop=True)

    updated_visit_plan_data.to_pickle(visit_plan_data_path)
    shipment_data.to_pickle(shipment_data_path)
    shipment_split_vp_data.to_pickle(shipment_split_path)
    date_analysis_data.to_pickle(date_analysis_data_path)
    sr_loading_data.to_pickle(sr_loading_data_path)
    sr_unloading_data.to_pickle(sr_unloading_data_path)
    stock_collection_data.to_pickle(stock_collection_data_path)
    credit_requests_data.to_pickle(credit_requests_data_path)

    print('Completed raw data extraction.')
