# -*- coding: utf-8 -*-

import time
import math
import random
import numpy as np
import pandas as pd

from poloniex import Poloniex
from priceindicator import get_price_indicator

def test_accuracy(coin_name, start_time, end_time, size):
	""" Return the test accuracy of the algorithm given a test time period in linux timestamp
	and a given coin

	coin_name: abbr of a coin, e.x. LTC
	start_time, end_time: time in linux timestamp
	size: number of tests in total
	"""
	correct = 0	# keep track of number of correct predictions

	for i in range(size):
		print("Iteration "+str(i))
		# pick a random time from the given time period to conduct a test
		# note that result in target is sigmoid(price_t/price_t-1 - 1)
		time_chosen = random.randint(start_time, end_time)
		target = get_target(coin_name, time_chosen)
		pred = get_price_indicator(coin_name, time_chosen)
		print("target: " + str(target))
		print("prediction: " + str(pred))

		# prediction is considered as correct if the predicted indicator is align with the target
		# indicator above 0.5 indicates a rise and indicator below 0.5 indicates a fall
		if ((target>0.5) and (pred>0.5)) or ((target<0.5) and (pred<0.5)):
			correct += 1

		print(str(correct)+" correct out of "+str(i+1))

	return (correct/size)

def get_target(coin_name, time_chosen):
	# download actual trading data of the test time
	polo = Poloniex()

	args = {}
	args["currencyPair"] = "BTC_" + coin_name
	command = "returnTradeHistory"

	args["start"] = time_chosen
	args["end"] = time_chosen + 1800

	js = polo.api(command, args)
	df = pd.DataFrame(js)

	df["rate"] = df["rate"].apply(float)
	df = df[np.isfinite(df['rate'])]

	delta = df["rate"].iloc[0]/df["rate"].iloc[-1]-1
	# sigmoid function with an multiply factor to magnify delta
	###############################
	# need to make the multiply factor an argument
	###############################
	target = 1/(1+math.e**(-150*delta))

	return target

print("Total accuracy: ")
print(test_accuracy("LTC", 1523493950, 1523993950, 100))
