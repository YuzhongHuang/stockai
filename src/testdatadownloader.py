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

def get_test_data(coin_name, end_time, trading_period, feature_length):
	print("Downloading " + coin_name + " test data")

	args = {}
	args["currencyPair"] = "USDT_" + coin_name
	command = "returnTradeHistory"

	start_time = end_time - trading_period * feature_length * minute
	args["start"] = start_time
	args["end"] = end_time

	df = get_data_until_success(command, args)
	return df

def get_data_until_success(command, args):
	"""
	Due to Poloniex's limit an api call to 50000 trading records
	get_data_utill_success() send api calls until reaching the end time
	"""
	polo = Poloniex()
	js0 = polo.api(command, args)
	df0 = pd.DataFrame(js0)
	# !!! timezone conversion fix -18000
	unix_time_convert = lambda x: int(time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timetuple())-18000)
	df0["date"] = df0["date"].apply(unix_time_convert)

	if len(df0) == 50000:
		while (df0["date"].iloc[-1] > args["start"]):
			args["end"] = df0["date"].iloc[-1] + 1

			print(args)
			js = polo.api(command, args)
			df = pd.DataFrame(js)

			if not df.empty:
				df["date"] = df["date"].apply(unix_time_convert)
				df0 = df0.append(df, ignore_index=True)
				if len(df) < 50000:
					df0.loc[-1,"date"] = args["start"]

	df0["rate"] = df0["rate"].apply(float)
	df0["total"] = df0["total"].apply(float)
	return df0

