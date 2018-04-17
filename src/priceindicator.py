import time
import math
import pandas as pd
import torch

from train import get_trained_net
from traindatadownloader import get_train_data
from trainsetgenerator import get_trainable_data
from testdatadownloader import get_test_data
from testfeaturegenerator import get_test_feature

def get_price_indicator(coin_name, start_time, cuda=False):
    hyper_param = {"bsize":50,"lr":0.4,"momentum":0.3, "epoch":150}
    market_param = {"trading_period":30, "diff_period":5, "feature_length":50, "track_days":18}

    # train_start_time = start_time
    train_start_time = start_time
    print(train_start_time)

    raw_train_data = get_train_data(coin_name, train_start_time, market_param["track_days"])
    trainable_dataset = get_trainable_data(raw_train_data, market_param)

    # while int(time.time()) - train_start_time < 300:
    #     wait = 300 + train_start_time - int(time.time())
    #     print("Wait for " + str(wait) + " seconds")
    #     time.sleep(wait) 
    # test_start_time = start_time
    test_start_time = train_start_time
    # test_start_time = 1516895133
    print(test_start_time)

    raw_test_data = get_test_data(coin_name, test_start_time, 
                                    market_param["trading_period"], market_param["feature_length"])
    test_inputs = get_test_feature(raw_test_data, market_param, cuda)

    net = get_trained_net(trainable_dataset, hyper_param, market_param, cuda)
    # torch.save(net, "../trainedmodels/pgpnet.pt")
    indicator = net(test_inputs)
    # print(indicator)

    return float(indicator)

# get_price_indicator("LTC")