# -*- coding = UTF-8 -*-
# Author : Kuang Fu
# Time   : 2025/8/16 21:42


from os.path import join

clean_data_path = '/nas/data/data_cn/data_clean'
process_data_path = '/nas/data/data_cn/data_process'
stock_daily_info_path = join(process_data_path, 'daily_info')
stock_exit_info_path = join(process_data_path, 'exit_info')
stock_raw_data_path = join(process_data_path, 'raw_data')
stock_resample_path = join(process_data_path, 'resample')
stock_split_data_path = join(process_data_path, 'split_data')

event_path = '/nas/data/stock_event'
dataset_path = join(event_path, 'dataset')
event_data_path = join(event_path, 'event_data')
event_info_path = join(event_path, 'event_info')
