import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
# -----------------------------------dataset---------------------------------------------------------------------------
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),  
    transforms.Normalize((0.1307, ), (0.3081, )) 
])
train = datasets.MNIST(root='../dataset/mnist',
                       train=True,
                       download=True,
                       transform=transform)
test = datasets.MNIST(root='../dataset/mnist',
                      train=False,
                      download=True,
                      transform=transform)
train_loader = DataLoader(train,
                          shuffle=True,
                          batch_size=batch_size)
test_loader = DataLoader(test,
                         shuffle=False,
                         batch_size=batch_size)
# -----------------------------------model---------------------------------------------------------------------------
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, 5)
        self.conv2 = torch.nn.Conv2d(10, 20, 5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.l1 = torch.nn.Linear(320, 160)
        self.l2 = torch.nn.Linear(160, 80)
        self.l3 = torch.nn.Linear(80, 40)
        self.l4 = torch.nn.Linear(40, 20)
        self.l5 = torch.nn.Linear(20, 10)

    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
# -----------------------------------loss optimizer--------------------------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)  # 更好的优化算法带动量的，动量梯度下降
# ----------------------------------------train test--------------------------------------------------------------------
def train(epoch):
    for batch_index, data in enumerate(train_loader, 0):
        inputs, target = data 
        inputs, target = inputs.to(device), target.to(device)
        outputs = model(inputs)  # 64 * 10的矩阵
        optimizer.zero_grad()
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

def test():
    total = 0
    correct = 0
    with torch.no_grad():  # 只是测试
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, outputs = torch.max(outputs.data, dim=1)    
            total += target.size(0)
            correct += (outputs == target).sum().item() 
    print('Accuracy on test set: %d %%' % (100*correct/total))

if __name__=='__main__':
    for epoch in range(10):
        train(epoch)
        test()
