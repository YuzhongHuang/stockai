# -*- coding: utf-8 -*-

import os
import time
import datetime
import pickle
import numpy as np
import pandas as pd

minute = 60
hour = minute*60
day = hour*24
week = day*7
month = day*30
year = day*365

def get_trainable_data(raw_data, market_param):
	unix_trading_period = int(market_param["trading_period"] * minute)
	unix_diff_period = int(market_param["diff_period"] * minute)
	feature_length = market_param["feature_length"]
	data = EMA(raw_data)

	trainable_dataset = process(data, unix_trading_period, feature_length)

	end = data["date"].iloc[0]
	start = data["date"].iloc[-1]

	for i in range(0, unix_trading_period-1, unix_diff_period):
		offset_data = data[data["date"] > (start + i)]
		offset_dataset = process(offset_data, unix_trading_period, feature_length)
		trainable_dataset = trainable_dataset.append(offset_dataset, ignore_index=True)

	return trainable_dataset

def EMA(raw_data, alpha=0.09):
	"""
	Converts raw data into EMA data 
	"""
	S = raw_data["rate"].iloc[0]

	for i, row in raw_data.iterrows():
		S = alpha*row["rate"] + (1-alpha)*S
		raw_data.set_value(i, "rate", S) 

	return raw_data


def process(data, unix_trading_period, feature_length):
	"""
	Process raw data to trainable (features,target) dataframe

	Args:
		data (dataframe): dataframe that records data chronologically
		trading_period (int): in minutes
	"""
	end = int(data["date"].iloc[0])
	start = int(data["date"].iloc[-1])

	period_price_features = []
	period_volumn_features = []
	prev_closing_price = None
	prev_volumns = [None, None]

	for i in range(start, end, unix_trading_period):
		df = data[(data["date"] >= i) & (data["date"] < (i+unix_trading_period))]
		if len(df.axes[0]) != 0:
			period_price_features.append(get_price_features(df, prev_closing_price))
			prev_closing_price = df["rate"].iloc[0]
			
			volumn_feature, prev_volumns = get_volumn_features(df, prev_volumns)
			period_volumn_features.append(volumn_feature)
		else:
			period_price_features.append([0.0,0.0,0.0])
			period_volumn_features.append([0.0,0.0])

	trainable_data = []

	for i in range(len(period_price_features)-feature_length):
		single_data = {}
		single_data["price_feature"] = np.array(period_price_features[i:i+feature_length]).reshape(3,feature_length)
		single_data["volumn_feature"] = np.array(period_volumn_features[i:i+feature_length]).reshape(2,feature_length)
		single_data["target"] = np.array(period_price_features[i+feature_length][0])
		trainable_data.append(single_data)

	return pd.DataFrame(trainable_data)

	
def get_price_features(period_data, prev_closing_price):
	"""
	get features from data collected from one trading period

	Args:
		period_data (dataframe): dataframe that records raw data chronologically in one trading period

	"""
	if prev_closing_price == None:
		prev_closing_price = period_data["rate"].iloc[-1]
	closing_price = period_data["rate"].iloc[0]

	high_price = period_data["rate"].max()
	low_price = period_data["rate"].min()

	features = [closing_price, high_price, low_price]

	return [(i/prev_closing_price - 1) for i in features]

def get_volumn_features(period_data, prev_volumns):
	buy_df = period_data[period_data["type"]=="buy"]
	sell_df = period_data[period_data["type"]=="sell"]

	buy_volumn = buy_df["total"].sum()
	sell_volumn = sell_df["total"].sum()

	buy_delta = 0
	sell_delta = 0

	if (prev_volumns[0] != None) and (prev_volumns[0] != 0):
		buy_delta = buy_volumn/prev_volumns[0] - 1

	if (prev_volumns[1] != None) and (prev_volumns[1] != 0):
		sell_delta = sell_volumn/prev_volumns[1] - 1

	return [buy_delta, sell_delta], [buy_volumn, sell_volumn]

