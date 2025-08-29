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
from typing import Dict
from paths.paths import clean_data_path, feature_path, stock_raw_data_path
from common_utils import milliseconds_to_time, perform_batch_task, time_to_milliseconds


def get_order_book(df1: pd.DataFrame, df2: pd.DataFrame):
	volume1 = df1.groupby(['code', 'side', 'price']).volume.agg([('volume', 'sum'), ('number', 'count')])
	volume2 = df2.groupby(['code', 'side', 'price']).volume.agg([('volume', 'sum'), ('number', 'count')])
	volume1.loc[volume2.index] = volume1.loc[volume2.index] - volume2
	df = volume1[volume1.volume.ne(0)].unstack(level=1, fill_value=0).sort_index().reset_index()
	df.columns = ['code', 'price', 'bid_volume', 'ask_volume', 'bid_number', 'ask_number']
	return df
	
	
def get_open_price(order: pd.DataFrame, cancel: pd.DataFrame, preclose: Dict):
	def get_open_price_by(time: int):
		df = get_order_book(order[order.time.lt(time)], cancel[cancel.time.lt(time)])
		df['bid_cum_volume'] = df.groupby('code', group_keys=False).bid_volume.apply(lambda x: x[::-1].cumsum()[::-1])
		df['ask_cum_volume'] = df.groupby('code').ask_volume.cumsum()
		df['match_volume'] = df[['bid_cum_volume', 'ask_cum_volume']].min(axis=1)
		df['volume_diff'] = (df.bid_cum_volume - df.ask_cum_volume).abs()
		df['preclose'] = df.code.map(preclose)
		df['price_diff'] = (df.price - df.preclose).abs()
		
		cond1 = (df.bid_cum_volume - df.bid_volume).le(df.match_volume)
		cond2 = (df.ask_cum_volume - df.ask_volume).le(df.match_volume)
		df = df[cond1 & cond2]
		
		df = df[df.groupby('code', group_keys=False).match_volume.apply(lambda x: x.eq(x.max()))]
		df = df[df.groupby('code', group_keys=False).volume_diff.apply(lambda x: x.eq(x.min()))]
		df = df[df.groupby('code', group_keys=False).price_diff.apply(lambda x: x.eq(x.min()))]
		df = df[df.groupby('code', group_keys=False).price.apply(lambda x: x.eq(x.min()))]
		df.price = np.where(df.match_volume.eq(0), df.preclose, df.price)
		df.index = pd.MultiIndex.from_product([df.code, [time]], names=['code', 'time'])
		df = df[['price', 'match_volume', 'bid_cum_volume', 'ask_cum_volume']]
		return df
	
	result = pd.concat(list(map(get_open_price_by, tqdm(order.bins.unique())))).sort_index()
	return result


def get_auction_feature(date: str):
	# preclose fuquan
	preclose = pd.read_parquet(join(clean_data_path, date, 'CSQuoteE.parquet'), columns=['code', 'preClose'])
	preclose = preclose.groupby('code').preClose.first().to_dict()
	
	
	columns = ['code', 'time', 'side', 'price', 'volume', 'amount']
	order = pd.read_parquet(join(stock_raw_data_path, date, 'order.parquet'), columns=columns)
	cancel = pd.read_parquet(join(stock_raw_data_path, date, 'cancel.parquet'), columns=columns)
	order, cancel = order[order.time.le(92500000)], cancel[cancel.time.le(92500000)]
	order['bins'] = milliseconds_to_time((time_to_milliseconds(order.time) // 15e3 + 1) * 15e3).astype('int')
	cancel['bins'] = milliseconds_to_time((time_to_milliseconds(cancel.time) // 15e3 + 1) * 15e3).astype('int')
	open_price = get_open_price(order, cancel, preclose)
	
	
	
	return


if __name__ == '__main__':
	date = '20250116'
