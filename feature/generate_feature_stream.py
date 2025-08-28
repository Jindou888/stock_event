# -*- coding = UTF-8 -*-
# Author : Kuang Fu
# Time   : 2025/8/23 14:02


import sys
sys.path.extend(['/home/kfu/stock_event/'])
import numpy as np
import pandas as pd
from os import makedirs
from os.path import join
from tqdm import tqdm
from typing import List
from paths.paths import clean_data_path, event_data_path, feature_path, stock_split_data_path
from common_utils import milliseconds_to_time, perform_batch_task, time_to_milliseconds


def read_event_data1(date: str):
	columns = ['code', 'serverTime', 'bizIndex']
	df = pd.read_parquet(join(event_data_path, date, 'event_data1.parquet'), columns=columns)
	df = df.drop_duplicates(['code', 'bizIndex']).sort_values(['code', 'bizIndex'], ignore_index=True)
	return df


def get_section_feature(date: str):
	def get_section_feature_by_code(t):
		code, event = t
		columns = [
			'code', 'bizIndex', 'preClose', 'open', 'high', 'low', 'lastPx', 'limitHigh', 'limitLow',
			'cumVolume', 'cumAmount', 'cumNumber', 'bidPx1', 'bidPx2', 'bidPx3', 'bidPx4',
			'bidPx5', 'bidPx6', 'bidPx7', 'bidPx8', 'bidPx9', 'bidPx10', 'bidVol1',
			'bidVol2', 'bidVol3', 'bidVol4', 'bidVol5', 'bidVol6', 'bidVol7',
			'bidVol8', 'bidVol9', 'bidVol10', 'bidNum1', 'bidNum2', 'bidNum3',
			'bidNum4', 'bidNum5', 'bidNum6', 'bidNum7', 'bidNum8', 'bidNum9',
			'bidNum10', 'askPx1', 'askPx2', 'askPx3', 'askPx4', 'askPx5', 'askPx6',
			'askPx7', 'askPx8', 'askPx9', 'askPx10', 'askVol1', 'askVol2',
			'askVol3', 'askVol4', 'askVol5', 'askVol6', 'askVol7', 'askVol8',
			'askVol9', 'askVol10', 'askNum1', 'askNum2', 'askNum3', 'askNum4',
			'askNum5', 'askNum6', 'askNum7', 'askNum8', 'askNum9', 'askNum10',
			'avgBidPx', 'avgAskPx', 'totalBidVol', 'totalAskVol', 'totalBidNum', 'totalAskNum',
		]
		quote = pd.read_parquet(join(stock_split_data_path, date, str(code), 'quotes.parquet'), columns=columns)
		quote = quote[quote.bizIndex.isin(event.bizIndex)]
		
		amount1, amount10 = np.nan, np.nan
		
		# feature
		return quote
	
	df = read_event_data1(date)
	df.serverTime = time_to_milliseconds(df.serverTime)
	df = pd.concat(list(map(get_section_feature_by_code, tqdm(df.groupby('code')))))
	df = df.set_index(['code', 'bizIndex']).drop(['serverTime'], axis=1).replace([np.inf, -np.inf], np.nan)
	makedirs(join(feature_path, date), exist_ok=True)
	df.to_parquet(join(feature_path, date, 'feature_section.parquet'))


def get_series_feature():
	return


def get_indus_feature():
	return


def division(a: float, b: float):
	try:
		result = (a - b) / (a + b)
	except ZeroDivisionError:
		result = np.nan
	return result


def get_trans_feature(date: str):
	def get_trans_feature_by_code(t):
		code, event = t
		columns = ['time', 'serverTime', 'bizIndex', 'transType', 'volume', 'amount']
		trans = pd.read_parquet(join(stock_split_data_path, date, str(code), 'trans.parquet'), columns=columns)
		trans = trans[trans.time.ge(93000000) & trans.time.le(145700000)]
		trans.time, trans.serverTime = time_to_milliseconds(trans.time), time_to_milliseconds(trans.serverTime)
		for idx, row in event.iterrows():
			time, bizIndex = row.serverTime, row.bizIndex
			data1 = trans[trans.serverTime.gt(time // 3e3 * 3e3 - 3e3 * 99) & trans.bizIndex.le(bizIndex)]
			data2 = trans[trans.serverTime.gt(time // 3e3 * 3e3 - 3e3 * 599) & trans.bizIndex.le(bizIndex)]
			data3 = trans[trans.serverTime.gt(time // 3e3 * 3e3 - 3e3 * 4799) & trans.bizIndex.le(bizIndex)]
			volume1 = data1.groupby(['transType']).volume.sum().to_dict()
			volume2 = data2.groupby(['transType']).volume.sum().to_dict()
			volume3 = data3.groupby(['transType']).volume.sum().to_dict()
			amount1 = data1.groupby(['transType']).amount.sum().to_dict()
			amount2 = data2.groupby(['transType']).amount.sum().to_dict()
			amount3 = data3.groupby(['transType']).amount.sum().to_dict()
			count1 = data1.groupby(['transType']).volume.count().to_dict()
			count2 = data2.groupby(['transType']).volume.count().to_dict()
			count3 = data3.groupby(['transType']).volume.count().to_dict()
			event.loc[idx, 'act_trans1'] = division(volume1.get(1, 0), volume1.get(2, 0))
			event.loc[idx, 'act_trans2'] = division(volume2.get(1, 0), volume2.get(2, 0))
			event.loc[idx, 'act_trans3'] = division(volume3.get(1, 0), volume3.get(2, 0))
			event.loc[idx, 'act_trans4'] = division(amount1.get(1, 0), amount1.get(2, 0))
			event.loc[idx, 'act_trans5'] = division(amount2.get(1, 0), amount2.get(2, 0))
			event.loc[idx, 'act_trans6'] = division(amount3.get(1, 0), amount3.get(2, 0))
			event.loc[idx, 'act_trans7'] = division(count1.get(1, 0), count1.get(2, 0))
			event.loc[idx, 'act_trans8'] = division(count2.get(1, 0), count2.get(2, 0))
			event.loc[idx, 'act_trans9'] = division(count3.get(1, 0), count3.get(2, 0))
		return event
	
	df = read_event_data1(date)
	df.serverTime = time_to_milliseconds(df.serverTime)
	df = pd.concat(list(map(get_trans_feature_by_code, tqdm(df.groupby('code')))))
	df = df.set_index(['code', 'bizIndex']).drop(['serverTime'], axis=1).replace([np.inf, -np.inf], np.nan)
	df.to_parquet(join(feature_path, date, 'feature_trans.parquet'))


def get_order_feature(date: str):
	def get_order_feature_by_code(t):
		code, event = t
		columns = ['time', 'serverTime', 'bizIndex', 'side', 'volume', 'amount']
		order = pd.read_parquet(join(stock_split_data_path, date, str(code), 'order.parquet'), columns=columns)
		order = order[order.time.ge(93000000) & order.time.le(145700000)]
		order.time, order.serverTime = time_to_milliseconds(order.time), time_to_milliseconds(order.serverTime)
		amount10 = np.nan
		value_free = np.nan
		for idx, row in event.iterrows():
			time, bizIndex = row.serverTime, row.bizIndex
			data1 = order[order.serverTime.gt(time // 6e3 * 6e3 - 6e3 * 49) & order.bizIndex.le(bizIndex)]
			data2 = order[order.serverTime.gt(time // 6e3 * 6e3 - 6e3 * 299) & order.bizIndex.le(bizIndex)]
			data3 = order[order.serverTime.gt(time // 6e3 * 6e3 - 6e3 * 2399) & order.bizIndex.le(bizIndex)]
			volume1 = data1.groupby(['side']).volume.sum().to_dict()
			volume2 = data2.groupby(['side']).volume.sum().to_dict()
			volume3 = data3.groupby(['side']).volume.sum().to_dict()
			amount1 = data1.groupby(['side']).amount.sum().to_dict()
			amount2 = data2.groupby(['side']).amount.sum().to_dict()
			amount3 = data3.groupby(['side']).amount.sum().to_dict()
			count1 = data1.groupby(['side']).volume.count().to_dict()
			count2 = data2.groupby(['side']).volume.count().to_dict()
			count3 = data3.groupby(['side']).volume.count().to_dict()
			event.loc[idx, 'act_order1'] = division(volume1.get(1, 0), volume1.get(2, 0))
			event.loc[idx, 'act_order2'] = division(volume2.get(1, 0), volume2.get(2, 0))
			event.loc[idx, 'act_order3'] = division(volume3.get(1, 0), volume3.get(2, 0))
			event.loc[idx, 'act_order4'] = division(amount1.get(1, 0), amount1.get(2, 0))
			event.loc[idx, 'act_order5'] = division(amount2.get(1, 0), amount2.get(2, 0))
			event.loc[idx, 'act_order6'] = division(amount3.get(1, 0), amount3.get(2, 0))
			event.loc[idx, 'act_order7'] = division(count1.get(1, 0), count1.get(2, 0))
			event.loc[idx, 'act_order8'] = division(count2.get(1, 0), count2.get(2, 0))
			event.loc[idx, 'act_order9'] = division(count3.get(1, 0), count3.get(2, 0))
			
			data1 = order[order.serverTime.gt(time // 6e3 * 6e3 - 6e3) & order.bizIndex.le(bizIndex)]
			large_bid1 = data1[data1.amount.ge(10e4) & data1.side.eq(1)].amount.sum()
			large_ask1 = data1[data1.amount.ge(10e4) & data1.side.eq(2)].amount.sum()
			event.loc[idx, 'net_bid1'] = (large_bid1 - large_ask1) / np.sqrt(amount10)
			
			large_bid2 = data1[data1.amount.ge(50e4) & data1.side.eq(1)].amount.sum()
			large_ask2 = data1[data1.amount.ge(50e4) & data1.side.eq(2)].amount.sum()
			event.loc[idx, 'net_bid2'] = (large_bid2 - large_ask2) / np.sqrt(amount10)
			
			total_bid1 = data1[data1.side.eq(1)].amount.sum()
			total_ask1 = data1[data1.side.eq(2)].amount.sum()
			event.loc[idx, 'net_bid3'] = (total_bid1 - total_ask1) / np.sqrt(value_free)
			
			data2 = order[order.serverTime.gt(time // 6e3 * 6e3 - 6e3 * 49) & order.bizIndex.le(bizIndex)]
			large_bid3 = data2[data2.amount.ge(10e4) & data2.side.eq(1)].amount.sum()
			total_bid3 = data2[data2.side.eq(1)].amount.sum() + 1
			event.loc[idx, 'net_bid4'] = large_bid3 / total_bid3
			
			total_bid5 = data2[data2.side.eq(1)].amount.sum()
			total_ask5 = data2[data2.side.eq(2)].amount.sum()
			event.loc[idx, 'net_bid5'] = (total_bid5 - total_ask5) / np.sqrt(value_free)
			
			data3 = order[order.serverTime.gt(time // 6e3 * 6e3 - 6e3 * 99) & order.bizIndex.le(bizIndex)]
			large_bid4 = data3[data3.amount.ge(5e4) & data3.side.eq(1)].amount.sum()
			total_bid4 = data3[data3.side.eq(1)].amount.sum() + 1
			event.loc[idx, 'net_bid6'] = large_bid4 / total_bid4
			
			total_bid6 = data3[data3.side.eq(1)].amount.sum()
			total_ask6 = data3[data3.side.eq(2)].amount.sum()
			event.loc[idx, 'net_bid7'] = (total_bid6 - total_ask6) / np.sqrt(value_free)
			
			large_bid5 = data3[data3.amount.ge(10e4) & data3.side.eq(1)].amount.sum()
			large_ask5 = data3[data3.amount.ge(10e4) & data3.side.eq(2)].amount.sum()
			event.loc[idx, 'net_bid8'] = (large_bid5 - large_ask5) / np.sqrt(amount10)
			
			large_bid6 = data3[data3.amount.ge(50e4) & data3.side.eq(1)].amount.sum()
			large_ask6 = data3[data3.amount.ge(50e4) & data3.side.eq(2)].amount.sum()
			event.loc[idx, 'net_bid9'] = (large_bid6 - large_ask6) / np.sqrt(amount10)
			
			large_bid7 = data2[data2.amount.ge(5e4) & data2.side.eq(1)].amount.sum() + 1
			large_ask7 = data2[data2.amount.ge(5e4) & data2.side.eq(2)].amount.sum() + 1
			event.loc[idx, 'net_bid10'] = (large_bid7 - large_ask7) / (large_bid7 + large_ask7)
			
			large_bid8 = data3[data3.amount.ge(10e4) & data3.side.eq(1)].amount.sum() + 1
			large_ask8 = data3[data3.amount.ge(10e4) & data3.side.eq(2)].amount.sum() + 1
			event.loc[idx, 'net_bid11'] = (large_bid8 - large_ask8) / (large_bid8 + large_ask8)
		return event
	
	df = read_event_data1(date)
	df.serverTime = time_to_milliseconds(df.serverTime)
	df = pd.concat(list(map(get_order_feature_by_code, tqdm(df.groupby('code')))))
	df = df.set_index(['code', 'bizIndex']).drop(['serverTime'], axis=1).replace([np.inf, -np.inf], np.nan)
	df.to_parquet(join(feature_path, date, 'feature_order.parquet'))


if __name__ == '__main__':
	# get_trans_feature('20250116')
	# get_order_feature('20250116')
	date = '20250116'
