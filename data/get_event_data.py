# -*- coding = UTF-8 -*-
# Author : Kuang Fu
# Time   : 2025/8/16 21:42


import sys
sys.path.extend(['/home/kfu/stock_event/'])
import numpy as np
import pandas as pd
from os import makedirs
from os.path import join
from typing import List
from paths.paths import clean_data_path, event_data_path, stock_exit_info_path, stock_split_data_path
from common_utils import milliseconds_to_time, perform_batch_task, time_to_milliseconds


def get_daily_pool(date: str) -> List[str]:
	df = pd.read_parquet(join(clean_data_path, date, 'CSInfo.parquet'))
	pool = df[df.specialTreat.eq(0)].code.to_list()
	return pool


def get_data1(date: str, code: str):
	columns = ['code', 'time', 'serverTime', 'bizIndex', 'bidNo', 'askNo', 'price', 'volume', 'amount']
	trans = pd.read_parquet(join(stock_split_data_path, date, code, 'trans.parquet'), columns=columns)
	trans = trans[trans.time.ge(93000000) & trans.time.le(145700000)]
	
	columns = ['bizIndex', 'orderNo']
	order = pd.read_parquet(join(stock_split_data_path, date, code, 'order.parquet'), columns=columns)
	
	columns = [
		'code', 'time', 'bizIndex', 'cumAmount', 'bidPx1', 'bidPx2', 'bidPx3', 'bidPx4',
		'bidPx5', 'bidVol1', 'bidVol2', 'bidVol3', 'bidVol4', 'bidVol5', 'askPx1', 'askPx2',
		'askPx3', 'askPx4', 'askPx5', 'askVol1', 'askVol2', 'askVol3', 'askVol4', 'askVol5',
	]
	quote = pd.read_parquet(join(stock_split_data_path, date, code, 'quotes.parquet'), columns=columns)
	return trans, order, quote


def merge_occur_quote(event: pd.DataFrame, quote: pd.DataFrame) -> pd.DataFrame:
	for idx, row in event.iterrows():
		time, serverTime = time_to_milliseconds(quote.time), time_to_milliseconds(row.serverTime) + 50
		quote_real = quote[time >= serverTime].head(1)
		if quote_real.empty or quote_real.bizIndex.iloc[0] <= row.bizIndex:
			continue
		event.loc[idx, quote.columns[4:]] = quote_real[quote.columns[4:]].values[0]
	info = quote[['bizIndex', 'bidPx1', 'askPx1', 'cumAmount']].copy()
	info = info.rename(columns={'bidPx1': 'best_bid', 'askPx1': 'best_ask'})
	event = pd.merge(event, info, how='left', on='bizIndex')
	return event


def get_event_data1(date: str):
	def get_event_data_by_code(code: str):
		try:
			trans, order, quote = get_data1(date, code)
		except FileNotFoundError:
			return
		df1 = trans[trans.price.eq(trans.price.cummax())].groupby('price').first().reset_index(drop=True)
		df1['orderNo'] = np.where(df1.bidNo > df1.askNo, df1.bidNo, df1.askNo)
		df2 = order[order.orderNo.isin(df1.orderNo)]
		df1.bizIndex = df1.orderNo.map(df2.set_index('orderNo').bizIndex.to_dict())
		datetime = milliseconds_to_time(time_to_milliseconds(df1.serverTime) // 3e3 * 3e3)
		df1['datetime'] = pd.to_datetime(int(date) * 1e9 + datetime, format='%Y%m%d%H%M%S%f')
		df1 = merge_occur_quote(df1, quote)
		cond1 = df1.time.ge(93001000) & df1.time.lt(113000000)
		cond2 = df1.time.ge(130000000) & df1.time.lt(140000000)
		cond3 = df1.serverTime.lt(145700000)
		df1 = df1[(cond1 | cond2) & cond3]
		return df1
	
	df = pd.concat(list(map(get_event_data_by_code, get_daily_pool(date))), ignore_index=True)
	df = df.drop(['bidNo', 'askNo', 'orderNo', 'volume', 'amount'], axis=1)
	makedirs(join(event_data_path, date), exist_ok=True)
	df.to_parquet(join(event_data_path, date, 'event_data1.parquet'))


# def get_event_data2(date: str):
# 	return
#
#
# def get_event_data3(date: str):
# 	return
#
#
# def get_event_data4(date: str):
# 	return
#
#
# def get_event_data5(date: str):
# 	return
#
#
# def get_event_data6(date: str):
# 	return
#
#
# def get_event_data7(date: str):
# 	return
#
#
# def get_event_data8(date: str):
# 	return
#
#
# def get_event_data9(date: str):
# 	return
#
#
# def get_event_data10(date: str):
# 	return


def get_label(date: str):
	exit_index = pd.read_parquet(join(stock_exit_info_path, date, 'exit_index.parquet'))
	exit_index = exit_index.index240.to_dict()
	exit_price = pd.read_parquet(join(stock_exit_info_path, date, 'exit_price1.parquet'))
	exit_price = exit_price.vwap240.to_dict()
	
	df = pd.read_parquet(join(event_data_path, date, 'event_data1.parquet'))
	df['exit_price'] = df.code.map(exit_price)
	midpx = (df.best_bid.fillna(df.best_ask) + df.best_ask.fillna(df.best_bid)) / 2
	df['y_true'] = (df.exit_price / midpx - 1) * 1e4 - df.datetime.map(exit_index)
	df.to_parquet(join(event_data_path, date, 'label1.parquet'))


if __name__ == '__main__':
	trading_dates = [
		'20250102', '20250103', '20250106', '20250107', '20250108', '20250109', '20250110', '20250113', '20250114',
		'20250115', '20250116', '20250117', '20250120', '20250121', '20250122', '20250123', '20250124', '20250127',
		'20250205', '20250206', '20250207', '20250210', '20250211', '20250212', '20250213', '20250214', '20250217',
		'20250218', '20250219', '20250220', '20250221', '20250224', '20250225', '20250226', '20250227', '20250228',
	]
	perform_batch_task(get_event_data1, trading_dates[1:], n_worker=6)
	perform_batch_task(get_label, trading_dates[:-1], n_worker=6)
