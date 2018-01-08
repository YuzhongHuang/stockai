# -*- coding: utf-8 -*-

import os
import time
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
		data_path (string): path to data dir
		last_update_time (string): unix timestamp of last update
		track_time (int): number of days to track coins history from now


	"""
	def __init__(self, data_path, time_path, track_days):
		"""
		Args:
			data_path (string): path to data dir
			time_path (string): path to pickle file recording last update time
			track_days (int): number of days to track coins history from now
		"""
		self.data_path = data_path
		self.last_update_time = '0'	#!!! work on read pickle file to load last update time
		self.track_time = track_days * day

		self.polo = Poloniex()

	def update(self):
		for file in os.listdir(self.data_path):
			coin_name = file.split('.')[0]
			print("updating " + coin_name)

			args = {}
			args["currencyPair"] = "BTC_" + coin_name
			command = "returnTradeHistory"

			self.last_update_time = int(time.time())
			args["end"] = self.last_update_time
			args["start"] = self.last_update_time - self.track_time

			js = self.polo.api(command, args)
			df = pd.DataFrame(js)
			df.to_csv(self.data_path+coin_name+'.csv', sep='\t')

	def track_coin(self, name):
		"""
		Add a coin to the track list

		Args:
			name (string): name of a coin
		"""

		open(self.data_path+name, 'a').close()
		self.update()