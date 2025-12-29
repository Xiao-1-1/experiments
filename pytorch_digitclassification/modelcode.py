import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_set=datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
test_set=datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)
train_loader=DataLoader(dataset=train_set,batch_size=64,shuffle=True)
test_loader=DataLoader(dataset=test_set,batch_size=64,shuffle=False)

#me defining the lameass model

class MYLAMEAHHMODEL(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1=nn.Linear(28*28,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)
    def forward(self,x):
        x=x.view(-1,28*28)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x

model=MYLAMEAHHMODEL()
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=0.001)

epochs=5
for epoch in range(epochs):
    model.train()
    running_loss=0.0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs=model(images)
        loss=criterion(outputs,labels)
        loss.backward()
        optimizer.step()


        running_loss+=loss.item()
    
    print(f"epoch [{epoch+1}/{epochs}]-loss:{running_loss:.4f}")

model.eval()
correct=0
total=0

with torch.no_grad():
    for images,labels in test_loader:
        outputs=model(images)
        _,predicted=torch.max(outputs.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
accuracy=100*correct/total
print(f"Test Accuracy: {accuracy:.2f}%")

torch.save(model.state_dict(),'digitclassifier_torch.pth')
print("model saved as digitclassifier_torch.pth")