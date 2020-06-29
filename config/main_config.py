import os

root_path = '/home/sushovan/python/jti_romania'

data_path = os.path.join(root_path, 'data')
visit_plan_raw_data_path = os.path.join(data_path, 'SRVisitPlan_all_v2.csv')
agent_replacements_raw_data_path = os.path.join(data_path, 'ReplacementTasks_all_v2.csv')
shipment_raw_data_path = os.path.join(data_path, 'Shipments_all_v2.csv')
pos_master_raw_data_path = os.path.join(data_path, 'POSMasterData_all.csv')
sr_loading_raw_data_path = os.path.join(data_path, 'SRLoading_all_v2.csv')
sr_unloading_raw_data_path = os.path.join(data_path, 'SRUnloading_all_v2.csv')
stock_collection_raw_data_path = os.path.join(data_path, 'StocCollection_all_v2.csv')
esgo_date_analysis_raw_data_path = os.path.join(data_path, 'ESGODateAnalysis_all.csv')
credit_requests_raw_data_path = os.path.join(data_path, 'CreditRequests_all_v2.csv')


def get_cleaned_data_paths(sales_office: str) -> tuple:
    cleaned_data_path = os.path.join(data_path, sales_office)

    visit_plan_data_path = os.path.join(cleaned_data_path,
                                        'visit_plan_{}.pkl'.format(sales_office))
    shipment_data_path = os.path.join(cleaned_data_path,
                                      'shipments_{}.pkl'.format(sales_office))
    date_analysis_data_path = os.path.join(cleaned_data_path,
                                           'bank_holidays_{}.pkl'.format(sales_office))
    shipment_split_path = os.path.join(cleaned_data_path,
                                       'shipment_split_vp_{}.pkl'.format(sales_office))
    sr_loading_data_path = os.path.join(cleaned_data_path,
                                        'sr_loading_{}.pkl'.format(sales_office))
    sr_unloading_data_path = os.path.join(cleaned_data_path,
                                          'sr_unloading_{}.pkl'.format(sales_office))
    stock_collection_data_path = os.path.join(cleaned_data_path,
                                              'stock_collection_{}.pkl'.format(sales_office))
    credit_requests_data_path = os.path.join(cleaned_data_path,
                                             'credit_requests_{}.pkl'.format(sales_office))
    pre_easter_effect_data_path = os.path.join(cleaned_data_path,
                                               'pre_easter_week_data.pkl')

    return cleaned_data_path, visit_plan_data_path, shipment_data_path, \
        date_analysis_data_path, shipment_split_path, sr_loading_data_path, sr_unloading_data_path, \
        stock_collection_data_path, credit_requests_data_path, pre_easter_effect_data_path


# which raw data columns to drop
date_analysis_raw_data_drop_cols = ['GID', 'fFiscalPeriodGID', 'fCompanyCode', 'TotalWorkingMinutes']
sr_loading_raw_data_drop_cols = ['id', 'createddate']
sr_unloading_raw_data_drop_cols = ['id', 'createddate']
credit_requests_raw_data_drop_cols = ['TaskCode', 'Status', 'AgentName']

# which data columns to keep
take_shipment_data_cols = ['invoice_date', 'agent_code', 'product_code',
                           'quantity', 'pos_code', 'promo_id', 'return']
take_future_visit_plan_data_cols = ['visit_date', 'agent_code',
                                    'pos_code', 'visit_order_day']
take_pos_visit_plan_data_cols = ['visit_date', 'pos_code',
                                 'pre_nonreplacement_holiday',
                                 'double_sell', 'triple_sell']
take_date_analysis_data_cols = ['DayNumberOfWeek', 'VisitPlanWeek', 'IsWorkingDate',
                                'NationalDoubleSell', 'NationalTripleSell',
                                'PricelistChgAnn', 'PricelistChg']
take_credit_req_data_cols = ['PlannedStartDate', 'PlannedClosedDate',
                             'ReqType', 'IncrCoef', 'AgentCode']
