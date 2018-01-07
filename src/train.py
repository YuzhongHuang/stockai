from net import PGPNet
import torch.optim as optim

net = PGPNet()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 