import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import gc
import time

device = 'cuda'
gc.collect()
torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()
torch.cuda.synchronize()
start_time = None

trainset = torchvision.datasets.MNIST(root='mnist/',train=True,download=True,transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))]))
trainloader = torch.utils.data.DataLoader(trainset,batch_size=512,shuffle=True,num_workers=2)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.4)
        self.dropout2 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

net = Net()
net.to(device)
loss_fn = nn.CrossEntropyLoss().cuda()
optimizer = optim.Adadelta(net.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=1, gamma=0.8)

start_time = time.time()

for epoch in range(20):
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader, 0):
        optimizer.zero_grad()

        outputs = net(inputs.to(device))
        loss = loss_fn(outputs, labels.to(device))
        
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    scheduler.step()

torch.cuda.synchronize()
end_time = time.time()
print("Total execution time = {:.3f} sec".format(end_time - start_time))
print("Max memory used by tensors = {} bytes".format(torch.cuda.max_memory_allocated()))