import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

# class PGPNet(nn.Module):

# 	def __init__(self, feature_length):
# 		super(PGPNet, self).__init__()
# 		self.price_conv1 = nn.Conv1d(3, 2, 3)
# 		self.price_conv2 = nn.Conv1d(2, 20, feature_length-2)
# 		self.volumn_conv1 = nn.Conv1d(2, 2, 3)
# 		self.volumn_conv2 = nn.Conv1d(2, 20, feature_length-2)
# 		self.fc = nn.Linear(40, 1)
# 		self.sigmoid = nn.Sigmoid()

# 	def forward(self, inputs):
# 		price_input = inputs[0]
# 		volumn_input = inputs[1]
# 		price_output = F.relu(self.price_conv2(F.relu(self.price_conv1(self.sigmoid(price_input)))))
# 		volumn_output = F.relu(self.volumn_conv2(F.relu(self.volumn_conv1(self.sigmoid(volumn_input)))))
# 		price_output = price_output.view(-1, 20)
# 		volumn_output = volumn_output.view(-1, 20)
# 		output_combined = torch.cat((price_output, volumn_output), 1)
# 		output = self.sigmoid(self.fc(output_combined))
# 		return output

class PGPNet(nn.Module):

	def __init__(self, feature_length):
		super(PGPNet, self).__init__()
		self.price_conv1 = nn.Conv1d(3, 2, 3)
		self.price_conv2 = nn.Conv1d(2, 20, feature_length-2)
		self.price_conv3 = nn.Conv1d(20, 1, 1)
		self.sigmoid = nn.Sigmoid()

	def forward(self, inputs):
		price_output = self.price_conv3(F.relu(self.price_conv2(F.relu(self.price_conv1(self.sigmoid(inputs))))))
		output = self.sigmoid(price_output)
		return output

# net = PGPNet()
# inputs = Variable(torch.randn(15, 3, 50))
# print(inputs.size())
# out = net(inputs)
# print(out)