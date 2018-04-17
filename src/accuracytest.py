# -*- coding: utf-8 -*-

import time
import math
import random
import numpy as np
import pandas as pd

from poloniex import Poloniex
from priceindicator import get_price_indicator

def test_accuracy(coin_name, start_time, end_time, size):
	correct = 0

	for i in range(size):
		print("Iteration "+str(i)+"")
		time_chosen = random.randint(start_time, end_time)
		target = get_target(coin_name, time_chosen)
		pred = get_price_indicator(coin_name, time_chosen)
		print("target: " + str(target))
		print("prediction: " + str(pred))

		if ((target>0.5) and (pred>0.5)) or ((target<0.5) and (pred<0.5)):
			correct += 1

		print(str(correct)+" correct out of "+str(i+1))

	return (correct/size)

def get_target(coin_name, time_chosen):
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
	target = 1/(1+math.e**(-150*delta))

	# print("delta: "+str(delta))
	# print("target: "+str(target))

	return target

print("Total accuracy: ")
print(test_accuracy("LTC", 1523281952, 1523881952, 100))
