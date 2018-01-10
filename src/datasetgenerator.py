# -*- coding: utf-8 -*-

import os
import time
import datetime
import numpy as np
import pandas as pd

minute = 60
hour = minute*60
day = hour*24
week = day*7
month = day*30
year = day*365

class DatasetGenerator(object):
	"""Single Coin dataset"""

	def __init__(self, input_file, output_file, trading_period=30, diff_period=5, feature_length=50):
		"""
		Args: 
			input_file (string): Path to the csv file of coin's raw data.
            transform (callable, optional): Optional transform to be applied
	            on a sample.
		"""
		self.raw_data = pd.read_csv(input_file, sep='\t')
		self.output_file = output_file
		self.unix_trading_period = trading_period * minute
		self.unix_diff_period = diff_period * minute
		self.feature_length = feature_length

	def len(self):
		return 

	def save_trainable_data(self):
		trainable_dataset = self.process(self.raw_data, self.unix_trading_period, self.feature_length)

		end = self.raw_data["date"].iloc[0]

		for i in range(self.unix_diff_period, self.unix_trading_period, self.unix_diff_period):
			offset_raw_data = self.raw_data[self.raw_data["date"] < (end - i)]
			offset_dataset = self.process(offset_raw_data, self.unix_trading_period, self.feature_length)
			trainable_dataset.append(offset_dataset)

		trainable_dataset.to_csv(self.output_file, sep='\t')

	def process(self, raw_data, unix_trading_period, feature_length):
		"""
		Process raw data to trainable (features,target) dataframe

		Args:
			raw_data (dataframe): dataframe that records raw data chronologically
			trading_period (int): in minutes
		"""
		end = raw_data["date"].iloc[0]
		start = raw_data["date"].iloc[-1]

		prev_closing_price = None
		period_features = []

		for i in range(end, start, -unix_trading_period):
			df = raw_data[(raw_data["date"]<=i) & (raw_data["date"]>(i-unix_trading_period))]
			period_features.append(self.get_features(df, prev_closing_price))
			prev_closing_price = df["rate"].iloc[-1]

		trainable_data = []

		for i in range(len(period_features)-feature_length):
			single_data = {}
			single_data["feature"] = np.array(period_features[i:i+feature_length])
			single_data["target"] = np.array(period_features[i+feature_length])
			trainable_data.append(single_data)

		return pd.DataFrame(trainable_data)

		
	def get_features(self, period_data, prev_closing_price=None):
		"""
		get features from data collected from one trading period

		Args:
			period_data (dataframe): dataframe that records raw data chronologically in one trading period

		"""
		closing_price = period_data["rate"].iloc[-1]

		if not prev_closing_price:
			prev_closing_price = closing_price

		high_price = period_data["rate"].max()
		low_price = period_data["rate"].min()

		features = [closing_price, high_price, low_price]

		return [i/prev_closing_price for i in features]


