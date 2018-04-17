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
    """Trains a model and run the model on the given time to get an price indicator

    Indicator is predicting sigmoid(price_t/price_t-1 - 1)
    Indicator above 0.5 indicates a rise and indicator below 0.5 indicates a fall
    The larger the indicator the higher the rise and vice versa

    training data comes from trading data before the given start time
    test data comes from trading data after the given start time

    coin_name: abbr of a coin, e.x. LTC
    start_time: start time in linux timestamp
    cuda: GPU option
    """
    hyper_param = {"bsize":50,"lr":0.4,"momentum":0.3, "epoch":150}
    market_param = {"trading_period":30, "diff_period":5, "feature_length":50, "track_days":18}

    # train_start_time = start_time
    train_start_time = start_time
    test_start_time = train_start_time
    print("Start Time: " + str(train_start_time))

    # Download raw data for training and parse into a trainable dataset
    raw_train_data = get_train_data(coin_name, train_start_time, market_param["track_days"])
    trainable_dataset = get_trainable_data(raw_train_data, market_param)

    # Download raw data for testing and parse into a test input
    raw_test_data = get_test_data(coin_name, test_start_time, 
                                    market_param["trading_period"], market_param["feature_length"])
    test_inputs = get_test_feature(raw_test_data, market_param, cuda)

    # train a model and get an indicator
    net = get_trained_net(trainable_dataset, hyper_param, market_param, cuda)
    indicator = net(test_inputs)

    # torch.save(net, "../trainedmodels/pgpnet.pt")

    return float(indicator)

# get_price_indicator("LTC")