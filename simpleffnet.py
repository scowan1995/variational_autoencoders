import torch
import pdb
import torchmetrics
import torch.nn as nn
from torchvision import datasets, transforms

device = 'cuda:0'

class SimpleFF(nn.Module):

    def __init__(self, input_size=28*28, hidden_size=256):
        super(SimpleFF, self).__init__()
        out_size = 10
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, 10)
        self.act = nn.ReLU()
        self.sf = nn.Softmax()

    def forward(self, image):
        batch_size = image.shape[0]
        flattened = image.view(batch_size, 784)
        x = self.l1(flattened)
        x = self.act(x)
        raw = self.l2(x)
        logits = self.sf(raw)
        return logits, raw


def train_epoch(model, optimiser, loss_fun, dataloader):
    accuracy = torchmetrics.Accuracy().to(device)
    for batch_idx, batch in enumerate(dataloader):
        images, targets = batch
        images = images.to(device)
        targets = targets.to(device)
        logits, raw = model(images)
        accuracy(logits, targets)

        optimiser.zero_grad()
        loss = loss_fun(raw, targets)
        loss.backward()
        optimiser.step()
    print(f"train accuracy {accuracy.compute().item()}")

def test_model(model, dl):
    accuracy = torchmetrics.Accuracy().to(device)
    for batch in dl:
        images, targets = batch
        images = images.to(device)
        targets = targets.to(device)
        logits, raw = model(images)
        accuracy(logits, targets)
    
    print(f"test accuracy {accuracy.compute().item()}")



def train():
    model = SimpleFF().to(device)
    optimiser = torch.optim.Adam(model.parameters())
    bsize = 64
    kwargs = {'num_workers': 1, 'pin_memory': False}
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=True, download=True,
                    transform=transforms.ToTensor()),
        batch_size=bsize, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=bsize, shuffle=True, **kwargs)
    for epoch in range(20):
        train_epoch(model, optimiser, torch.nn.CrossEntropyLoss(), train_loader)
        if epoch % 5 == 0:
            test_model(model, test_loader)


        
if __name__ == "__main__":
    train()
    # acc after 20 epochs : 0.9395
