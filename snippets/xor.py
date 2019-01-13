import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

EPOCH_TO_TRAIN = 5000
MAX_LOSS = 0.0001


class XorNet(nn.Module):

    def __init__(self):
        super(XorNet, self).__init__()
        self.fc1 = nn.Linear(2, 3, True)
        self.fc2 = nn.Linear(3, 1, True)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        # print(x.data.numpy())
        x = self.fc2(x)
        # print(x.data.numpy())
        return x


net = XorNet()

inputs = (
    torch.Tensor([0, 0]),
    torch.Tensor([0, 1]),
    torch.Tensor([1, 0]),
    torch.Tensor([1, 1])
)

targets = (
    torch.Tensor([0]),
    torch.Tensor([1]),
    torch.Tensor([1]),
    torch.Tensor([0])
)

criterion = nn.MSELoss()  # Mean squared error los
optimizer = optim.SGD(net.parameters(), lr=0.01)
# optim.

for name, param in net.named_parameters():
    print("PARAM {}".format(name), param.data.numpy(),
          param.size())

print("Training Loop:")

for idx in range(0, EPOCH_TO_TRAIN):
    for input, target in zip(inputs, targets):
        optimizer.zero_grad()
        output = net(inputs[0])
        loss = criterion(output, targets[0])
        loss.backward()
        optimizer.step()
    print("Epoch {:<5} Loss: {}".format(idx, loss.data))
    if loss.data <= MAX_LOSS:
        print("Max Loss {} reached".format(MAX_LOSS))
        break
