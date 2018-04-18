from net import PGPNet
import pickle
import torch
import torch.optim as optim
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from singlecoindataset import SingleCoinDataset

def get_trained_net(dataset, hyper_param, market_param, cuda=False):
    net = PGPNet(market_param["feature_length"]).double()
    if cuda:
        net = net.cuda() 

    trainset = SingleCoinDataset(dataset)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=hyper_param["bsize"],
    shuffle=True)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(net.parameters(), lr=hyper_param["lr"], momentum=hyper_param["momentum"]) 

    print("Start training for " +str(hyper_param["epoch"])+ " epochs")

    for epoch in range(hyper_param["epoch"]):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            price_inputs = data["price_feature"]
            targets =  data["target"]

            if cuda:
                price_inputs = price_inputs.cuda()
                targets = targets.cuda()

            # wrap them in Variable
            price_inputs, targets = Variable(price_inputs), Variable(targets)
            # price_inputs, volumn_inputs, targets = Variable(price_inputs), Variable(volumn_inputs), Variable(targets)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(price_inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            # if (i%50 == 49):    # print every 50 mini-batches
            #     print('[%d, %5d] loss: %.3f' %
            #           (epoch + 1, i + 1, running_loss / 50))
            #     # print(outputs)
            #     # print("targets")
            #     # print(targets)
            #     running_loss = 0.0

    print('Finished Training')
    return net