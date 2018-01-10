# -*- coding: utf-8 -*-

import os
import time
import datetime
import pandas as pd

from poloniex import Poloniex

minute = 60
hour = minute*60
day = hour*24
week = day*7
month = day*30
year = day*365

class HistoryManager(object):
	"""
	Manages files of historical data of coins of interest.  

	Attributes:
		data_dir (string): path to data dir
		last_update_time (string): unix timestamp of last update
		track_time (int): number of days to track coins history from now


	"""
	def __init__(self, data_dir, time_path, track_days):
		"""
		Args:
			data_dir (string): path to data dir
			time_path (string): path to pickle file recording last update time
			track_days (int): number of days to track coins history from now
		"""
		self.data_dir = data_dir
		self.last_update_time = '0'	#!!! work on read pickle file to load last update time
		self.track_time = track_days * day

		self.polo = Poloniex()

	def update(self):
		for file in os.listdir(self.data_dir):
			coin_name = file.split('.')[0]
			print("updating " + coin_name)

			args = {}
			args["currencyPair"] = "BTC_" + coin_name
			command = "returnTradeHistory"

			self.last_update_time = int(time.time())
			args["end"] = self.last_update_time
			args["start"] = self.last_update_time - self.track_time

			df = self.get_data_until_success(command, args)
			df.to_csv(self.data_dir+coin_name+'.csv', sep='\t')

	def get_data_until_success(self, command, args):
		"""
		Due to Poloniex's limit an api call to 50000 trading records
		get_data_utill_success() send api calls until reaching the end time
		"""
		js0 = self.polo.api(command, args)
		df0 = pd.DataFrame(js0)
		# !!! timezone conversion fix -18000
		unix_time_convert = lambda x: int(time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timetuple())-18000)
		df0["date"] = df0["date"].apply(unix_time_convert)

		while (df0["date"].iloc[-1] > args["start"]) and (df0["tradeID"].iloc[-1] != df0["tradeID"].iloc[-2]):
			args["end"] = df0["date"].iloc[-1] + 1

			print(args)
			js = self.polo.api(command, args)
			df = pd.DataFrame(js)

			if not df.empty:
				df["date"] = df["date"].apply(unix_time_convert)
				df0 = df0.append(df, ignore_index=True)

		return df0

	def track_coin(self, name):
		"""
		Add a coin to the track list

		Args:
			name (string): name of a coin
		"""

		open(self.data_dir+name+'.csv', 'a').close()
		self.update()