# -*- coding = UTF-8 -*-
# Author : Kuang Fu
# Time   : 2025/8/23 14:02


import sys
sys.path.extend(['/home/kfu/stock_event/'])
import numpy as np
import pandas as pd
from os import makedirs
from os.path import join
from typing import Dict
import warnings
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
		df.index = pd.MultiIndex.from_product([df.code, [time]], names=['code', 'bins'])
		df = df[['price', 'match_volume', 'bid_cum_volume', 'ask_cum_volume']]
		return df
	
	result = pd.concat(list(map(get_open_price_by, order.bins.unique()))).sort_index()
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
	
	products = [preclose.keys(), np.sort(order.bins.unique())]
	df1 = pd.DataFrame(index=pd.MultiIndex.from_product(products, names=['code', 'bins']), dtype='float')
	df1[['order_volume', 'order_amount']] = order.groupby(['code', 'bins'])[['volume', 'amount']].sum()
	df1['order_number'] = order.groupby(['code', 'bins']).volume.count()
	group = order[order.side.eq(1)].groupby(['code', 'bins'])
	df1[['bid_order_volume', 'bid_order_amount']] = group[['volume', 'amount']].sum()
	df1['bid_order_number'] = group.volume.count()
	group = order[order.side.eq(2)].groupby(['code', 'bins'])
	df1[['ask_order_volume', 'ask_order_amount']] = group[['volume', 'amount']].sum()
	df1['ask_order_number'] = group.volume.count()
	
	df1[['cancel_volume', 'cancel_amount']] = cancel.groupby(['code', 'bins'])[['volume', 'amount']].sum()
	df1['cancel_number'] = cancel.groupby(['code', 'bins']).volume.count()
	group = cancel[cancel.side.eq(1)].groupby(['code', 'bins'])
	df1[['bid_cancel_volume', 'bid_cancel_amount']] = group[['volume', 'amount']].sum()
	df1['bid_cancel_number'] = group.volume.count()
	group = cancel[cancel.side.eq(2)].groupby(['code', 'bins'])
	df1[['ask_cancel_volume', 'ask_cancel_amount']] = group[['volume', 'amount']].sum()
	df1['ask_cancel_number'] = group.volume.count()
	df1 = df1.fillna(0)
	df1.loc[open_price.index, open_price.columns] = open_price
	
	df1 = df1.reset_index()
	group = df1.groupby('code')
	df1['preclose'] = df1.code.map(preclose)
	df1['amount10'] = np.nan
	df1['timeTag'] = df1.bins.map({time: i for i, time in enumerate(np.sort(df1.bins.unique()))})
	df1['unmatch_vol'] = df1.bid_cum_volume - df1.ask_cum_volume
	df1['match_vol_pct'] = group.match_volume.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).fillna(0)
	df1['match_volume_prop'] = df1.match_volume / df1.code.map(group.match_volume.sum())
	df1['ret'] = group.price.pct_change(fill_method=None).fillna(0)
	df1['ret2pre'] = (df1.price / df1.preclose - 1)
	
	df = pd.DataFrame(index=preclose.keys(), dtype='float64')
	bid_number = group.bid_order_number.sum() - group.bid_cancel_number.sum()
	ask_number = group.ask_order_number.sum() - group.ask_cancel_number.sum()
	df['imb_num'] = (bid_number - ask_number) / (bid_number + ask_number)
	bid_volume = group.bid_order_volume.sum() - group.bid_cancel_volume.sum()
	ask_volume = group.ask_order_volume.sum() - group.ask_cancel_volume.sum()
	df['imb_vol'] = (bid_volume - ask_volume) / (bid_volume + ask_volume)
	
	total_order_volume = group.order_volume.sum() - group.cancel_volume.sum()
	df['vol_ratio_15s'] = df1[df1.bins.eq(92500000)].set_index('code').order_volume / total_order_volume
	df['vol_ratio_30s'] = df1[df1.bins.ge(92445000)].groupby('code').order_volume.sum() / total_order_volume
	df['vol_ratio_1m'] = df1[df1.bins.ge(92415000)].groupby('code').order_volume.sum() / total_order_volume
	df['vol_ratio_5m'] = df1[df1.bins.ge(92015000)].groupby('code').order_volume.sum() / total_order_volume
	df['bid_vol_ratio_15s'] = df1[df1.bins.eq(92500000)].set_index('code').bid_order_volume / total_order_volume
	df['bid_vol_ratio_30s'] = df1[df1.bins.ge(92445000)].groupby('code').bid_order_volume.sum() / total_order_volume
	df['match_vol_ratio'] = df1[df1.bins.eq(92500000)].set_index('code').match_volume / total_order_volume
	
	df['ret_std'] = group.ret.std()
	df['ret_max_5m'] = df1[df1.bins.ge(92015000)].groupby('code').ret.max()
	df['ret_min_5m'] = df1[df1.bins.ge(92015000)].groupby('code').ret.min()
	df['avg_ret2pre'] = group.ret2pre.apply(lambda x: x.ewm(span=20, adjust=False).mean().iloc[-1])
	df['ret2pre_std'] = group.ret2pre.std()
	df['ret2pre_max'] = group.ret2pre.max()
	df['ret2pre_min'] = group.ret2pre.min()
	df['vwtd_ret'] = (df1.ret * df1.match_volume_prop).groupby(df1.code).sum()
	df['vwtd_ret2pre'] = (df1.ret2pre * df1.match_volume_prop).groupby(df1.code).sum()
	
	df['low2high'] = (group.price.max() - group.price.min()) / group.preclose.first()
	with warnings.catch_warnings():
		warnings.simplefilter('ignore', RuntimeWarning)
		df['pv_cor'] = group[['price', 'match_volume']].apply(lambda x: x.price.corr(x.match_volume))
		df['pt_cor'] = group[['price', 'timeTag']].apply(lambda x: x.price.corr(x.timeTag))
		df['vt_cor'] = group[['match_volume', 'timeTag']].apply(lambda x: x.match_volume.corr(x.timeTag))
		df['uvt_cor'] = group[['unmatch_vol', 'timeTag']].apply(lambda x: x.unmatch_vol.corr(x.timeTag))
		group1 = df1[df1.bins.ge(92015000)].groupby('code')
		df['vol_ret_cor'] = group1[['ret', 'match_vol_pct']].apply(lambda x: x.ret.corr(x.match_vol_pct))
		df['r2pt_cor_5m'] = group1[['ret2pre', 'timeTag']].apply(lambda x: x.ret2pre.corr(x.timeTag))
	df['vol_std'] = group.match_volume.std() / group.match_volume.mean()
	
	df['cncl_tot_num_ratio'] = group.cancel_number.sum() / group.order_number.sum()
	df['cncl_tot_vol_ratio'] = group.cancel_volume.sum() / group.order_volume.sum()
	bid_cancel_number, ask_cancel_number = group.bid_cancel_number.sum(), group.ask_cancel_number.sum()
	df['imb_cncl_num_ratio'] = (bid_cancel_number - ask_cancel_number) / (bid_cancel_number + ask_cancel_number)
	
	x1 = group.bid_order_volume.sum() - group.bid_cancel_volume.sum()
	x2 = group.bid_order_number.sum() - group.bid_cancel_number.sum()
	x3 = group.ask_order_volume.sum() - group.ask_cancel_volume.sum()
	x4 = group.ask_order_number.sum() - group.ask_cancel_number.sum()
	y1, y2 = x1 / x2, x3 / x4
	df['imb_vpn'] = (y1 - y2) / (y1 + y2)
	
	temp = df1[df1.bins.eq(92500000)].set_index('code')
	df['vol_ratio10'] = (temp.match_volume * temp.price) / temp.amount10
	df['imb_match_vol'] = temp.unmatch_vol / (temp.bid_cum_volume + temp.ask_cum_volume)
	df['bid_vol_prop_15s'] = temp.bid_order_volume / temp.ask_order_volume
	
	book = get_order_book(order, cancel)
	book['preclose'] = book.code.map(preclose)
	total_bid_order_volume = book.groupby('code').bid_volume.sum()
	total_bid_order_number = book.groupby('code').bid_number.sum()
	
	agg_bid_volume = book[book.price.ge(book.preclose)].groupby('code').bid_volume.sum()
	df['pos_bid_vol_prop_bp0'] = agg_bid_volume / total_bid_order_volume
	df['pos_bid_tot_vol_prop_bp0'] = agg_bid_volume / total_order_volume
	
	group2 = book[book.price.ge(book.preclose * 1.001)].groupby('code')
	df['pos_bid_vol_prop_bp10'] = group2.bid_volume.sum() / total_bid_order_volume
	df['pos_bid_num_prop_bp10'] = group2.bid_number.sum() / total_bid_order_number
	
	df = df.astype('float64').replace([np.inf, -np.inf], np.nan)
	makedirs(join(feature_path, date), exist_ok=True)
	df.to_parquet(join(feature_path, date, 'feature_auction.parquet'))


if __name__ == '__main__':
	trading_dates = [
		'20250102', '20250103', '20250106', '20250107', '20250108', '20250109', '20250110', '20250113', '20250114',
		'20250115', '20250116', '20250117', '20250120', '20250121', '20250122', '20250123', '20250124', '20250127',
		'20250205', '20250206', '20250207', '20250210', '20250211', '20250212', '20250213', '20250214', '20250217',
		'20250218', '20250219', '20250220', '20250221', '20250224', '20250225', '20250226', '20250227', '20250228',
	]
	perform_batch_task(get_auction_feature, trading_dates, n_worker=6)
