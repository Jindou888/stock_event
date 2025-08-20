# -*- coding = UTF-8 -*-
# Author : Kuang Fu
# Time   : 2025/8/16 21:42


import numpy as np
import pandas as pd
from os import makedirs
from os.path import join
from paths.paths import event_info_path, stock_raw_data_path
from common_utils import perform_batch_task


def get_trans_info3(date: str):
	df = pd.read_parquet(join(stock_raw_data_path, date, 'trans.parquet'))
	df = df[df.time.ge(93000000) & df.time.le(145700000)]
	df['orderNo'] = np.where(df.transType.eq(1), df.bidNo, df.askNo)
	df1 = df.groupby(['code', 'orderNo'])[['time', 'serverTime', 'bizIndex', 'transType']].last()
	df2 = df.groupby(['code', 'orderNo']).amount.sum()
	df = pd.concat([df1, df2], axis=1).reset_index().drop(['orderNo'], axis=1)
	makedirs(join(event_info_path, date), exist_ok=True)
	df.to_parquet(join(event_info_path, date, 'trans_info3.parquet'))


def read_trans_info3(date: str):
	df = pd.read_parquet(join(event_info_path, date, 'trans_info3.parquet'))
	return df


def get_event_info3(date: str):
	dates = [d for d in trading_dates if d <= date][-10:]
	df = pd.concat(list(map(read_trans_info3, dates)))
	df = df.groupby(['code', 'transType']).amount.quantile(0.997)
	df = df.unstack(level=1).fillna(0).clip(1e5, 1e8)
	df.columns = ['bid', 'ask']
	next_date = min([d for d in trading_dates if d > date])
	makedirs(join(event_info_path, next_date), exist_ok=True)
	df.to_parquet(join(event_info_path, next_date, 'event_info3.parquet'))


if __name__ == '__main__':
	trading_dates = [
		'20250102', '20250103', '20250106', '20250107', '20250108', '20250109', '20250110', '20250113', '20250114',
		'20250115', '20250116', '20250117', '20250120', '20250121', '20250122', '20250123', '20250124', '20250127',
		'20250205', '20250206', '20250207', '20250210', '20250211', '20250212', '20250213', '20250214', '20250217',
		'20250218', '20250219', '20250220', '20250221', '20250224', '20250225', '20250226', '20250227', '20250228',
	]
	# perform_batch_task(get_trans_info3, trading_dates, n_worker=8)
	perform_batch_task(get_event_info3, trading_dates[:-1], n_worker=4)
