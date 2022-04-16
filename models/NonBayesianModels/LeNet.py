import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import ceil

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        #nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
        nn.init.normal_(m.weight, mean=0, std=1)
        nn.init.constant(m.bias, 0)

class LeNet(nn.Module):
    def __init__(self, num_classes, inputs=3, wide=2):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(inputs, ceil(6*wide), 5)
        self.conv2 = nn.Conv2d(ceil(6*wide), ceil(16*wide), 5)
        self.fc1   = nn.Linear(ceil(16*5*5*wide), ceil(120*wide))
        self.fc2   = nn.Linear(ceil(120*wide), 84)
        self.fc3   = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        #print(out.size())
        out = F.max_pool2d(out, 2)
        #print(out.size())
        out = F.relu(self.conv2(out))
        #print(out.size())
        out = F.max_pool2d(out, 2)
        #print(out.size())
        out = out.view(out.size(0), -1)
        #print(out.size())
        out = F.relu(self.fc1(out))
        #print(out.size())
        out = F.relu(self.fc2(out))
        #print(out.size())
        out = self.fc3(out)
        #print(out.size())
        #print("END")
        return(out)
