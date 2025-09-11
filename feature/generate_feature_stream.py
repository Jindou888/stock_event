# -*- coding = UTF-8 -*-
# Author : Kuang Fu
# Time   : 2025/8/23 14:02


import sys
sys.path.extend(['/home/kfu/stock_event/'])
import numpy as np
import pandas as pd
from os.path import join
from paths.paths import clean_data_path, event_data_path, feature_path, stock_resample_path, stock_split_data_path
from common_utils import perform_batch_task, time_to_milliseconds, timefn
from research_utils import get_quote_additional_columns


def read_event_data1(date: str):
	columns = ['code', 'bizIndex', 'serverTime', 'datetime']
	df = pd.read_parquet(join(event_data_path, date, 'event_data1.parquet'), columns=columns)
	df = df.drop_duplicates(['code', 'bizIndex']).sort_values(['code', 'bizIndex'], ignore_index=True)
	return df


@timefn
def get_section_feature(date: str):
	def get_section_feature_by_code(t):
		code, df = t
		columns = [
			'code', 'bizIndex', 'preClose', 'open', 'high', 'low', 'limitHigh', 'limitLow',
			'cumVolume', 'cumAmount', 'cumNumber', 'bidPx1', 'bidPx2', 'bidPx3', 'bidPx4',
			'bidPx5', 'bidPx6', 'bidPx7', 'bidPx8', 'bidPx9', 'bidPx10', 'bidVol1', 'bidVol2',
			'bidVol3', 'bidVol4', 'bidVol5', 'bidVol6', 'bidVol7', 'bidVol8', 'bidVol9', 'bidVol10',
			'bidNum1', 'bidNum2', 'bidNum3', 'bidNum4', 'bidNum5', 'bidNum6',  'bidNum7', 'bidNum8',
			'bidNum9', 'bidNum10', 'askPx1', 'askPx2', 'askPx3', 'askPx4', 'askPx5', 'askPx6',
			'askPx7', 'askPx8', 'askPx9', 'askPx10', 'askVol1', 'askVol2',  'askVol3', 'askVol4',
			'askVol5', 'askVol6', 'askVol7', 'askVol8', 'askVol9', 'askVol10',  'askNum1', 'askNum2',
			'askNum3', 'askNum4', 'askNum5', 'askNum6', 'askNum7', 'askNum8',  'askNum9', 'askNum10',
			'avgBidPx', 'avgAskPx', 'totalBidVol', 'totalAskVol', 'totalBidNum', 'totalAskNum',
		]
		df1 = pd.read_parquet(join(stock_split_data_path, date, str(code), 'quotes.parquet'), columns=columns)
		df1 = df1[df1.bizIndex.isin(df.bizIndex)].set_index('bizIndex')
		df1 = get_quote_additional_columns(df1)
		df1.loc[df1.limitLow.le(0.02), ['limitHigh', 'limitLow', 'avgBidPx', 'avgAskPx']] = np.nan
		
		df = df.set_index('bizIndex')
		df['midPx'] = df1.midPx
		df['imb1'] = (df1.bidVol1 - df1.askVol1) / (df1.bidVol1 + df1.askVol1)
		df['imb2'] = (df1.totalBidVol5 - df1.totalAskVol5) / (df1.totalBidVol5 + df1.totalAskVol5)
		df['imb3'] = (df1.totalBidVol10 - df1.totalAskVol10) / (df1.totalBidVol10 + df1.totalAskVol10)
		df['imb4'] = (df1.totalBidVol - df1.totalAskVol) / (df1.totalBidVol + df1.totalAskVol)
		df['imb5'] = (df1.bidNum1 - df1.askNum1) / (df1.bidNum1 + df1.askNum1)
		df['imb6'] = (df1.totalBidNum5 - df1.totalAskNum5) / (df1.totalBidNum5 + df1.totalAskNum5)
		df['imb7'] = (df1.totalBidNum10 - df1.totalAskNum10) / (df1.totalBidNum10 + df1.totalAskNum10)
		df['imb8'] = (df1.totalBidNum - df1.totalAskNum) / (df1.totalBidNum + df1.totalAskNum)
		df['imb9'] = df1.bidVol1 / df1.totalBidVol5 - df1.askVol1 / df1.totalAskVol5
		df['imb10'] = df1.totalBidVol5 / df1.totalBidVol10 - df1.totalAskVol5 / df1.totalAskVol10
		df['imb11'] = df1.totalBidVol10 / df1.totalBidVol - df1.totalAskVol10 / df1.totalAskVol
		df['imb12'] = df1.bidNum1 / df1.totalBidNum5 - df1.askNum1 / df1.totalAskNum5
		df['imb13'] = df1.totalBidNum5 / df1.totalBidNum10 - df1.totalAskNum5 / df1.totalAskNum10
		df['imb14'] = df1.totalBidNum10 / df1.totalBidNum - df1.totalAskNum10 / df1.totalAskNum
		x1 = df1.totalBidVol10 / df1.totalBidNum10
		x2 = df1.totalAskVol10 / df1.totalAskNum10
		df['imb15'] = (x1 - x2) / (x1 + x2)
		
		df['gap1'] = (df1.midPx - df1.avgBidPx5) / df1.midPx
		df['gap2'] = (df1.avgAskPx5 - df1.midPx) / df1.midPx
		df['gap3'] = (df1.midPx - df1.avgBidPx10) / df1.midPx
		df['gap4'] = (df1.avgAskPx10 - df1.midPx) / df1.midPx
		df['gap5'] = (df1.midPx - df1.avgBidPx) / df1.midPx
		df['gap6'] = (df1.avgAskPx - df1.midPx) / df1.midPx
		df['gap7'] = df.gap1 - df.gap2
		df['gap8'] = df.gap3 - df.gap4
		df['gap9'] = df.gap5 - df.gap6
		df['gap10'] = df.gap1 / df.gap3
		df['gap11'] = df.gap2 / df.gap4
		df['gap12'] = df.gap3 / df.gap5
		df['gap13'] = df.gap4 / df.gap6
		
		df['ret1'] = (df1.midPx - df1.open) / df1.midPx
		df['ret2'] = (df1.midPx - df1.preClose) / df1.midPx
		df['ret3'] = (df1.open - df1.preClose) / df1.open
		df['ret4'] = df.ret1 - df.ret3
		df['ret5'] = (df1.midPx - df1.high) / df1.midPx
		df['ret6'] = (df1.midPx - df1.low) / df1.midPx
		df['ret7'] = (df1.limitHigh - df1.midPx) / df1.midPx
		
		df['sprd_rat1'] = (df1.bidPx1 - df1.bidPx2) / df1.open
		df['sprd_rat2'] = (df1.askPx2 - df1.askPx1) / df1.open
		df['sprd_rat3'] = (df1.avgAskPx5 - df1.avgBidPx5) / df1.open
		return df
	
	event = read_event_data1(date)
	event.serverTime = time_to_milliseconds(event.serverTime)
	feature = pd.concat(list(map(get_section_feature_by_code, event.groupby('code'))))
	feature = feature.reset_index().set_index(['code', 'bizIndex']).drop(['serverTime', 'datetime'], axis=1)
	feature = feature.astype('float64').replace([np.inf, -np.inf], np.nan)
	feature.to_parquet(join(feature_path, date, 'feature_section.parquet'))


@timefn
def get_series_indus_feature(date: str):
	def tstack_select(feature: pd.DataFrame):
		feature = feature.T.stack(future_stack=True).loc[df.index]
		return feature
	
	columns = [
		'code', 'serverTime', 'bizIndex', 'preClose', 'open', 'high', 'low', 'cumVolume', 'cumAmount', 'cumNumber',
		'bidPx1', 'bidPx2', 'bidPx3', 'bidPx4', 'bidPx5', 'bidPx6', 'bidPx7', 'bidPx8', 'bidPx9', 'bidPx10',
		'bidVol1', 'bidVol2', 'bidVol3', 'bidVol4', 'bidVol5', 'bidVol6', 'bidVol7', 'bidVol8', 'bidVol9',
		'bidVol10', 'bidNum1', 'bidNum2', 'bidNum3', 'bidNum4', 'bidNum5', 'bidNum6', 'bidNum7', 'bidNum8',
		'bidNum9', 'bidNum10', 'askPx1', 'askPx2', 'askPx3', 'askPx4', 'askPx5', 'askPx6', 'askPx7', 'askPx8',
		'askPx9', 'askPx10', 'askVol1', 'askVol2', 'askVol3', 'askVol4', 'askVol5', 'askVol6', 'askVol7',
		'askVol8', 'askVol9', 'askVol10', 'askNum1', 'askNum2', 'askNum3', 'askNum4', 'askNum5', 'askNum6',
		'askNum7', 'askNum8', 'askNum9', 'askNum10', 'avgBidPx', 'avgAskPx', 'totalBidVol', 'totalAskVol',
		'totalBidNum', 'totalAskNum',
	]
	df1 = pd.read_parquet(join(stock_resample_path, date, 'quote.parquet'), columns=columns)
	df1 = get_quote_additional_columns(df1)
	df1['amount1'] = np.nan
	df1['amount10'] = np.nan
	df1['amount10_ratio'] = np.nan
	
	df2 = pd.read_parquet(join(clean_data_path, date, 'CSInfo.parquet'))
	df2 = df2[df2.code.isin(df1.index.get_level_values(0).unique())].set_index('code')
	
	columns = [
		'serverTime', 'preClose', 'open', 'high', 'low', 'cumVolume', 'cumAmount', 'bidPx1', 'bidVol1',
		'askPx1', 'askVol1', 'totalBidVol', 'totalAskVol', 'avgBidPx5', 'avgAskPx5', 'avgBidPx10',
		'avgAskPx10', 'volume', 'amount', 'midPx', 'amount1', 'amount10', 'amount10_ratio',
	]
	df1 = df1[columns].unstack(level=0)
	assert np.all(df1.open.columns == df2.index)
	
	df = read_event_data1(date)
	df = df.set_index(['code', 'datetime'])
	
	x3 = df1.bidVol1.rolling(200, min_periods=1).sum()
	x4 = df1.askVol1.rolling(200, min_periods=1).sum()
	df['imb16'] = tstack_select(((x3 - x4) / (x3 + x4)).rolling(200, min_periods=1).sum())
	x5 = df1.totalBidVol.rolling(200, min_periods=1).sum()
	x6 = df1.totalAskVol.rolling(200, min_periods=1).sum()
	df['imb17'] = tstack_select(((x5 - x6) / (x5 + x6)).rolling(200, min_periods=1).sum())
	
	df['ret8'] = tstack_select(1 - df1.midPx.shift(20).bfill() / df1.midPx)
	df['ret9'] = tstack_select(1 - df1.midPx.shift(100).bfill() / df1.midPx)
	df['ret10'] = tstack_select(1 - df1.midPx.shift(600).bfill() / df1.midPx)
	df['ret11'] = tstack_select((df1.midPx / df1.preClose - 1).rolling(100, min_periods=1).mean())
	
	ret = 1 - df1.midPx.shift(1).bfill() / df1.midPx
	df['rv1'] = tstack_select((ret ** 2).rolling(20, min_periods=1).sum())
	df['rv2'] = tstack_select((ret ** 2).rolling(100, min_periods=1).sum())
	df['rv3'] = tstack_select((df1.high - df1.low) / df1.midPx)
	
	ret_abs = ret.abs()
	df['trend1'] = df.ret8 / tstack_select(ret_abs.rolling(20, min_periods=1).sum())
	df['trend2'] = df.ret9 / tstack_select(ret_abs.rolling(100, min_periods=1).sum())
	df['trend3'] = df.ret10 / tstack_select(ret_abs.rolling(600, min_periods=1).sum())
	
	df['trend4'] = tstack_select((df1.totalBidVol + df1.totalAskVol).diff().rolling(100, min_periods=1).mean() / np.sqrt(df1.amount10))
	df['trend5'] = tstack_select((df1.totalBidVol + df1.totalAskVol).diff().rolling(200, min_periods=1).mean() / np.sqrt(df1.amount10))
	df['trend6'] = tstack_select((df1.avgBidPx5.pct_change(fill_method=None) + df1.avgAskPx5.pct_change(fill_method=None)).rolling(100, min_periods=1).mean())
	
	spread1 = df1.askPx1 - df1.bidPx1
	spread4 = df1.avgAskPx5 - df1.avgBidPx5
	df['sprd_std1'] = tstack_select(spread1.rolling(20, min_periods=1).std())
	df['sprd_std2'] = tstack_select(spread1.rolling(100, min_periods=1).std())
	df['sprd_std3'] = tstack_select(spread4.rolling(100, min_periods=1).std())
	
	midPx1 = df1.midPx.rolling(20, min_periods=1)
	midPx2 = df1.midPx.rolling(100, min_periods=1)
	df['pv1'] = tstack_select(np.sqrt(np.log(midPx1.max() - midPx1.min() + 1) ** 2 / 4 / np.log(2)))
	df['pv2'] = tstack_select(np.sqrt(np.log(midPx2.max() - midPx2.min() + 1) ** 2 / 4 / np.log(2)))
	
	volume1 = df1.volume.rolling(20, min_periods=1).sum()
	volume2 = df1.volume.rolling(100, min_periods=1).sum()
	volume3 = df1.volume.rolling(600, min_periods=1).sum()
	amount1 = df1.amount.rolling(20, min_periods=1).sum()
	amount2 = df1.amount.rolling(100, min_periods=1).sum()
	amount3 = df1.amount.rolling(600, min_periods=1).sum()
	msc = time_to_milliseconds(df1.serverTime)
	msc[:] = np.where(msc <= 45000000, msc - 34200000, msc - 39600000)
	df['time2'] = tstack_select(msc // 6e4 + 1)
	
	df['amt_rat1'] = tstack_select(amount1 / amount2)
	df['amt_rat2'] = tstack_select(amount2 / amount3)
	df['amt_rat3'] = tstack_select(df1.cumAmount / (msc // 1e3 + 1) / df1.amount10 * 14220)
	
	avgpx1 = amount1 / volume1
	avgpx2 = amount2 / volume2
	avgpx3 = amount3 / volume3
	avgpx4 = df1.cumAmount / df1.cumVolume
	
	df['mid2avg1'] = tstack_select(1 - avgpx1 / df1.midPx)
	df['mid2avg2'] = tstack_select(1 - avgpx2 / df1.midPx)
	df['mid2avg3'] = tstack_select(1 - avgpx3 / df1.midPx)
	df['mid2avg4'] = tstack_select(1 - avgpx4 / df1.midPx)
	df['mid2avg5'] = tstack_select(df1.midPx.rolling(20, min_periods=1).mean() / avgpx1 - 1)
	df['mid2avg6'] = tstack_select(df1.midPx.rolling(100, min_periods=1).mean() / avgpx2 - 1)
	df['mid2avg7'] = tstack_select(df1.midPx.rolling(600, min_periods=1).mean() / avgpx3 - 1)
	
	df['mid2high1'] = tstack_select(df1.midPx.rolling(20, min_periods=1).max() / df1.high - 1)
	df['mid2low1'] = tstack_select(df1.midPx / df1.midPx.rolling(200, min_periods=1).min() - 1)
	
	df['avg_ret1'] = tstack_select(1 - avgpx4.shift(20).bfill() / avgpx4)
	df['avg_ret2'] = tstack_select(1 - avgpx4.shift(100).bfill() / avgpx4)
	df['avg_ret3'] = tstack_select(1 - avgpx4.shift(600).bfill() / avgpx4)
	
	df['avg_amt1'] = tstack_select(df1.amount.rolling(20, min_periods=1).mean() / np.sqrt(df1.amount1))
	df['avg_amt2'] = tstack_select(df1.amount.rolling(100, min_periods=1).mean() / np.sqrt(df1.amount1))
	df['avg_amt3'] = tstack_select(df1.amount.rolling(20, min_periods=1).mean() / np.sqrt(df1.amount10))
	df['avg_amt4'] = tstack_select(df1.amount.rolling(100, min_periods=1).mean() / np.sqrt(df1.amount10))
	df['avg_amt5'] = tstack_select(df1.amount.rolling(600, min_periods=1).mean() / np.sqrt(df1.amount10))
	
	df['corr1'] = tstack_select(df1.totalBidVol.rolling(20, min_periods=1).corr(df1.totalAskVol))
	df['corr2'] = tstack_select(df1.totalBidVol.rolling(100, min_periods=1).corr(df1.totalAskVol))
	df['corr3'] = tstack_select(df1.totalBidVol.rolling(600, min_periods=1).corr(df1.totalAskVol))
	df['corr4'] = tstack_select(df1.bidPx1.rolling(200, min_periods=1).corr(df1.askPx1))
	df['corr5'] = tstack_select(df1.bidVol1.rolling(200, min_periods=1).corr(df1.askVol1))
	df['corr6'] = tstack_select(df1.avgBidPx5.rolling(200, min_periods=1).corr(df1.avgAskPx5))
	df['corr7'] = tstack_select(df1.avgBidPx10.rolling(200, min_periods=1).corr(df1.avgAskPx10))
	
	amount1 = df1.amount.rolling(20, min_periods=1).sum()
	amount2 = df1.amount.rolling(100, min_periods=1).sum()
	amount3 = df1.amount.rolling(600, min_periods=1).sum()
	total_amount = df1.cumAmount.sum(axis=1)
	indus_amount = df1.cumAmount.T.groupby(df2.indus2).transform('sum').T
	amount10_ratio = tstack_select(df1.amount10_ratio)
	
	df['amt_rat4'] = tstack_select(amount1.div(amount1.sum(axis=1), axis=0))
	df['amt_rat5'] = tstack_select(amount2.div(amount2.sum(axis=1), axis=0))
	df['amt_rat6'] = tstack_select(amount3.div(amount3.sum(axis=1), axis=0))
	df['amt_rat7'] = tstack_select(df1.cumAmount.div(total_amount, axis=0))
	df['amt_rat8'] = tstack_select(df1.cumAmount / indus_amount)
	df['amt_rat9'] = tstack_select(indus_amount.div(total_amount, axis=0))
	df['amt_rat10'] = df.amt_rat4 - amount10_ratio
	df['amt_rat11'] = df.amt_rat5 - amount10_ratio
	df['amt_rat12'] = df.amt_rat6 - amount10_ratio
	df['amt_rat13'] = df.amt_rat7 - amount10_ratio
	
	ret1 = (1 - df1.open / df1.midPx).fillna(0)
	ret2 = (1 - df1.preClose / df1.midPx).fillna(0)
	ret3 = (1 - df1.preClose / df1.open).fillna(0)
	ret8 = (1 - df1.midPx.shift(20).bfill() / df1.midPx).fillna(0)
	ret9 = (1 - df1.midPx.shift(100).bfill() / df1.midPx).fillna(0)
	ret10 = (1 - df1.midPx.shift(600).bfill() / df1.midPx).fillna(0)
	
	df['amt_ret1'] = df.amt_rat4 * tstack_select(ret8)
	df['amt_ret2'] = df.amt_rat5 * tstack_select(ret8)
	df['amt_ret3'] = df.amt_rat6 * tstack_select(ret10)
	df['amt_ret4'] = df.amt_rat7 * tstack_select(ret1)
	
	df['indus_ret1'] = tstack_select(ret1.T.groupby(df2.indus2).transform('mean').T)
	df['indus_ret2'] = tstack_select(ret2.T.groupby(df2.indus2).transform('mean').T)
	df['indus_ret3'] = tstack_select(ret3.T.groupby(df2.indus2).transform('mean').T)
	df['indus_ret4'] = tstack_select(ret8.T.groupby(df2.indus2).transform('mean').T)
	df['indus_ret5'] = tstack_select(ret9.T.groupby(df2.indus2).transform('mean').T)
	df['indus_ret6'] = tstack_select(ret10.T.groupby(df2.indus2).transform('mean').T)
	df['ret2indus1'] = tstack_select(ret1) - df.indus_ret1
	df['ret2indus2'] = tstack_select(ret2) - df.indus_ret2
	df['ret2indus3'] = tstack_select(ret3) - df.indus_ret3
	df['ret2indus4'] = tstack_select(ret8) - df.indus_ret4
	df['ret2indus5'] = tstack_select(ret9) - df.indus_ret5
	df['ret2indus6'] = tstack_select(ret10) - df.indus_ret6
	
	df['indus_ret7'] = tstack_select(ret1.T.groupby(df2.indus2).transform(lambda x: x.nlargest(10).mean()).T)
	df['indus_ret8'] = tstack_select(ret2.T.groupby(df2.indus2).transform(lambda x: x.nlargest(10).mean()).T)
	df['indus_ret9'] = tstack_select(ret3.T.groupby(df2.indus2).transform(lambda x: x.nlargest(10).mean()).T)
	df['indus_ret10'] = tstack_select(ret8.T.groupby(df2.indus2).transform(lambda x: x.nlargest(10).mean()).T)
	df['indus_ret11'] = tstack_select(ret9.T.groupby(df2.indus2).transform(lambda x: x.nlargest(10).mean()).T)
	df['indus_ret12'] = tstack_select(ret10.T.groupby(df2.indus2).transform(lambda x: x.nlargest(10).mean()).T)
	
	df['ret2indus7'] = tstack_select(ret1) - df.indus_ret7
	df['ret2indus8'] = tstack_select(ret2) - df.indus_ret8
	df['ret2indus9'] = tstack_select(ret3) - df.indus_ret9
	df['ret2indus10'] = tstack_select(ret8) - df.indus_ret10
	df['ret2indus11'] = tstack_select(ret9) - df.indus_ret11
	df['ret2indus12'] = tstack_select(ret10) - df.indus_ret12
	
	df['indus_std1'] = tstack_select(ret1.T.groupby(df2.indus2).transform('std').T)
	df['indus_std2'] = tstack_select(ret2.T.groupby(df2.indus2).transform('std').T)
	df['indus_std3'] = tstack_select(ret3.T.groupby(df2.indus2).transform('std').T)
	df['indus_std4'] = tstack_select(ret8.T.groupby(df2.indus2).transform('std').T)
	df['indus_std5'] = tstack_select(ret9.T.groupby(df2.indus2).transform('std').T)
	df['indus_std6'] = tstack_select(ret10.T.groupby(df2.indus2).transform('std').T)
	
	indus_count = df1.cumAmount.T.groupby(df2.indus2).transform('count').T
	df['rank1'] = tstack_select(ret1.T.groupby(df2.indus2).transform('rank').T / indus_count)
	df['rank2'] = tstack_select(ret2.T.groupby(df2.indus2).transform('rank').T / indus_count)
	df['rank3'] = tstack_select(ret3.T.groupby(df2.indus2).transform('rank').T / indus_count)
	df['rank4'] = tstack_select(ret8.T.groupby(df2.indus2).transform('rank').T / indus_count)
	df['rank5'] = tstack_select(ret9.T.groupby(df2.indus2).transform('rank').T / indus_count)
	df['rank6'] = tstack_select(ret10.T.groupby(df2.indus2).transform('rank').T / indus_count)
	
	df['indus_rise1'] = tstack_select(ret1.gt(0).T.groupby(df2.indus2).transform('sum').T / indus_count)
	df['indus_rise2'] = tstack_select(ret2.gt(0).T.groupby(df2.indus2).transform('sum').T / indus_count)
	df['indus_rise3'] = tstack_select(ret3.gt(0).T.groupby(df2.indus2).transform('sum').T / indus_count)
	df['indus_rise4'] = tstack_select(ret8.gt(0).T.groupby(df2.indus2).transform('sum').T / indus_count)
	df['indus_rise5'] = tstack_select(ret9.gt(0).T.groupby(df2.indus2).transform('sum').T / indus_count)
	df['indus_rise6'] = tstack_select(ret10.gt(0).T.groupby(df2.indus2).transform('sum').T / indus_count)
	
	df = df.reset_index().set_index(['code', 'bizIndex']).drop(['serverTime', 'datetime'], axis=1)
	df = df.astype('float64').replace([np.inf, -np.inf], np.nan)
	df.to_parquet(join(feature_path, date, 'feature_series_indus.parquet'))


def division(a: float, b: float):
	try:
		result = (a - b) / (a + b)
	except ZeroDivisionError:
		result = np.nan
	return result


@timefn
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
	df = pd.concat(list(map(get_trans_feature_by_code, df.groupby('code'))))
	df = df.set_index(['code', 'bizIndex']).drop(['serverTime', 'datetime'], axis=1)
	df = df.astype('float64').replace([np.inf, -np.inf], np.nan)
	df.to_parquet(join(feature_path, date, 'feature_trans.parquet'))


@timefn
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
	df = pd.concat(list(map(get_order_feature_by_code, df.groupby('code'))))
	df = df.set_index(['code', 'bizIndex']).drop(['serverTime', 'datetime'], axis=1)
	df = df.astype('float64').replace([np.inf, -np.inf], np.nan)
	df.to_parquet(join(feature_path, date, 'feature_order.parquet'))


def generate_feature_stream(date: str):
	get_section_feature(date)
	get_series_indus_feature(date)
	get_trans_feature(date)
	get_order_feature(date)


if __name__ == '__main__':
	trading_dates = [
		'20250102', '20250103', '20250106', '20250107', '20250108', '20250109', '20250110', '20250113', '20250114',
		'20250115', '20250116', '20250117', '20250120', '20250121', '20250122', '20250123', '20250124', '20250127',
		'20250205', '20250206', '20250207', '20250210', '20250211', '20250212', '20250213', '20250214', '20250217',
		'20250218', '20250219', '20250220', '20250221', '20250224', '20250225', '20250226', '20250227', '20250228',
	]
	perform_batch_task(generate_feature_stream, trading_dates, n_worker=6)
