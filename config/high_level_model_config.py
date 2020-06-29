val_start_date = '2018-01-01'
test_start_date = '2018-07-01'
run_phase = 'test'
# run_phase = 'val'

high_std_thr = 0.95
low_density_thr = 0.10
low_vp_thr = 0.75
high_sparsity_thr = 0.10

regression_mode = 'ensemble'

quantity_lower_limit = 0.5
very_high_oos_threshold = 10
loading_size_thr = 5


clf_drop_cols = ['VisitPlanWeek', 'week', 'is_odd_month', 'DayNumberOfWeek',
                 'quarter', 'year', 'year_of_engagement',
                 'shipments_1', 'shipments_2', 'shipments_3', 'shipments_4', 'shipments_5', 'shipments_6',
                 'nonzero_Shipments_6', 'nonzero_Shipments_5', 'nonzero_Shipments_4',
                 'nonzero_Shipments_3', 'nonzero_Shipments_2',
                 'days_since_last_sale_3', 'days_since_last_sale_2', 'days_since_last_sale_1',
                 'days_since_last_nonzero_sale_3', 'days_since_last_nonzero_sale_2', 'days_since_last_nonzero_sale_1',
                 'mm_3', 'Chg_Shipments_1_from_mm_3',
                 'mm_nonzero_3', 'Chg_nonzero_Shipments_1_from_mm_nonzero_3',
                 'stock_6', 'stock_5', 'stock_4',
                 'credit_request_type',
                 'is_bad_weather_last_15',
                 # below two lines are features that shouldn't ever be included in model #
                 'is_last_sale_date', 'isMissedPlan', 'stock', 'promo_id',
                 'isVisitPlan', 'is_zero_sale', 'is_first_nonzero_sale_date']

keep_in_reg_result = ['invoice_date', 'known_shipment', 'predicted_loading',
                      'product_cat', 'pos_code',
                      'VisitPlanWeek', 'month', 'year', 'week', 'days_from_easter',
                      'pre_nonreplacement_holiday',
                      'pred_is_nonzero_shipments',
                      'double_sell', 'triple_sell', 'credit_request_type',
                      'double_sell_1', 'triple_sell_1',
                      'days_since_price_chg_ann', 'days_from_price_chg',
                      'isVisitPlan_1']
use_drop_reg_cols_instead_of_keep = False
reg_keep_cols_1 = ['pos_code', 'week', 'month',
                   'nonzero_Shipments_1', 'nonzero_Shipments_2', 'nonzero_Shipments_3',
                   'nonzero_Shipments_4', 'nonzero_Shipments_5', 'nonzero_Shipments_6',
                   'days_since_last_nonzero_sale', 'ma_nonzero_interval_3',
                   'is_first_plan_date', 'days_since_first_nonzero_sale',
                   'double_sell', 'triple_sell', 'credit_request_coeff',
                   'double_sell_1', 'triple_sell_1',
                   'days_since_price_chg_ann', 'return_1',
                   'pre_nonreplacement_holiday', 'isVisitPlan_1',
                   'is_zero_sale_1', 'num_consecutive_zero_sales',
                   'ma_3', 'ma_6',
                   'ma_nonzero_3', 'ma_nonzero_6']
reg_keep_cols_2 = ['DayNumberOfWeek', 'pos_code',
                   'pre_nonreplacement_holiday', 'month',
                   'days_since_price_chg_ann',
                   'triple_sell', 'double_sell',
                   'credit_request_coeff', 'days_from_easter',
                   'is_bad_weather_last_15']

reg_drop_cols = ['shipments_6', 'shipments_5', 'shipments_4',  #'shipments_3', 'shipments_2',#'shipments_1',
                 'Chg_pct_Shipments_1_from_Shipments_2',
                 'ma_6', 'ma_3',
                 # 'Chg_Shipments_1_from_ma_6', 'Chg_Shipments_1_from_ma_3',
                 'mm_6', 'mm_3',
                 'Chg_Shipments_1_from_mm_6', 'Chg_Shipments_1_from_mm_3',
                 'days_since_first_sale',  #'is_first_plan_date',
                 # 'ma_interval_3',
                 'days_since_last_sale_3', 'days_since_last_sale_2',
                 'days_since_last_sale_1', 'days_since_last_sale',
                 'double_sell_lead_1',  # 'triple_sell_lead_1',
                 # 'double_sell_2', 'double_sell_1', 'double_sell',
                 # 'triple_sell_2', 'triple_sell_1', 'triple_sell',
                 'credit_request_type',  # 'days_from_easter', 'credit_request_coeff',
                 'isVisitPlan_2',  #'isVisitPlan_1', 'return_1',
                 'days_from_price_chg', # 'days_since_price_chg_ann', 'PricelistChg_1',
                 'num_consecutive_nonzero_sales',  #'num_consecutive_zero_sales',
                 # 'nonzero_Shipments_6', 'nonzero_Shipments_5', 'nonzero_Shipments_4',
                 # 'nonzero_Shipments_3', 'nonzero_Shipments_2', 'nonzero_Shipments_1',
                 'Chg_pct_nonzero_Shipments_1_from_nonzero_Shipments_2',
                 # 'ma_nonzero_6', 'ma_nonzero_3',
                 'Chg_nonzero_Shipments_1_from_ma_nonzero_6',  #'Chg_nonzero_Shipments_1_from_ma_nonzero_3',
                 'mm_nonzero_6', 'mm_nonzero_3',
                 'Chg_nonzero_Shipments_1_from_mm_nonzero_6', 'Chg_nonzero_Shipments_1_from_mm_nonzero_3',
                 'isMissedPlan_2',  #'isMissedPlan_1',
                 'stock_6', 'stock_5', 'stock_4', 'stock_3',  #'stock_2', 'stock_1',
                 'Chg_pct_Stock_1_from_Stock_2',
                 'ma_Stock_6', 'ma_Stock_3',
                 'Chg_Stock_1_from_ma_Stock_6', 'Chg_Stock_1_from_ma_Stock_3',
                 # 'days_since_first_nonzero_sale',
                 # 'ma_nonzero_interval_3',
                 'days_since_last_nonzero_sale_3', 'days_since_last_nonzero_sale_2',
                 # 'days_since_last_nonzero_sale_1', 'days_since_last_nonzero_sale',
                 'week', 'weekday_holiday', 'is_odd_month',
                 # 'month', 'quarter', 'year', 'year_of_engagement', 'DayNumberOfWeek',
                 'promo_id', 'is_zero_sale_3', 'is_zero_sale_2',  #'is_zero_sale_1',
                 'product_cat',  #'pre_nonreplacement_holiday',
                 # 'is_bad_weather_last_three', 'is_bad_weather_last_six',
                 # below two lines are features that shouldn't ever be included in model #
                 'stock', 'isMissedPlan', 'is_last_sale_date', 'is_zero_sale',
                 'is_first_nonzero_sale_date', 'isVisitPlan']
