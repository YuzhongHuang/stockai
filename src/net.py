import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class PGPNet(nn.Module):

	def __init__(self):
		super(PGPNet, self).__init__()
		self.conv1 = nn.Conv1d(3, 2, 3)
		self.conv2 = nn.Conv1d(2, 20, 48)
		self.conv3 = nn.Conv1d(20, 1, 1)

	def forward(self, input):
		x = self.conv3(F.relu(self.conv2(F.relu(self.conv1(input)))))
		x = x.view(x.size(0), -1)
		return x

net = PGPNet()
input = Variable(torch.randn(1, 3, 50))
out = net(input)
print(out)