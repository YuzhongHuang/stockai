# -*- coding: utf-8 -*-

import os
import time
import datetime
import numpy as np
import pandas as pd

from poloniex import Poloniex

minute = 60
hour = minute*60
day = hour*24
week = day*7
month = day*30
year = day*365

def get_train_data(coin_name, end_time, track_days):
	print("Downloading " + coin_name + " training data")

	args = {}
	args["currencyPair"] = "USDT_" + coin_name
	command = "returnTradeHistory"

	start_time = end_time - track_days * day
	period = track_days * day / 30

	args["end"] = end_time
	args["start"] = end_time - period

	df0 = get_data_until_success(command, args)

	for i in range(end_time-period, start_time, -period):
		args["end"] = i
		args["start"] = i - period
		df0 = df0.append(get_data_until_success(command, args), ignore_index=True)

	return df0

# def get_data_total(command, args, remain_time):
# 	if remain_time < 30*day:
# 		args["start"] = args["end"] - remain_time
# 		return get_data_until_success(command, args)

# 	args["start"] = args["end"] - 30*day
# 	df0 = get_data_until_success(command, args)

# 	args["end"] = args["start"]
# 	return df0.append(get_data_month(command, args, remain_time - 30*day))

def get_data_until_success(command, args):
	"""
	Due to Poloniex's limit an api call to 50000 trading records
	get_data_utill_success() send api calls until reaching the end time
	"""
	polo = Poloniex()
	# print(args)
	js0 = polo.api(command, args)
	df0 = pd.DataFrame(js0)
	# !!! timezone conversion fix -18000
	unix_time_convert = lambda x: int(time.mktime(datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S").timetuple())-18000)
	df0["date"] = df0["date"].apply(unix_time_convert)

	if len(df0) == 50000:
		while (df0["date"].iloc[-1] > args["start"]):
			args["end"] = df0["date"].iloc[-1] + 1

			# print(args)
			js = polo.api(command, args)
			df = pd.DataFrame(js)

			if not df.empty:
				df["date"] = df["date"].apply(unix_time_convert)
				df0 = df0.append(df, ignore_index=True)
				if len(df) < 50000:
					df0.loc[-1,"date"] = args["start"]

	df0["rate"] = df0["rate"].apply(float)
	df0["total"] = df0["total"].apply(float)
	df0 = df0[np.isfinite(df0['rate'])]
	return df0