import torch
import torchmetrics
import torch.nn as nn
from collections import namedtuple
from vae import VAE
   
from torch.nn import functional as F
from torchvision import datasets, transforms


bsize = 64
kwargs = {'num_workers': 1, 'pin_memory': False}
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=bsize, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=bsize, shuffle=True, **kwargs)

latent_space = 64
encoder_hidden = 256
B = 2
decoder_hidden_size = 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TopModel(nn.Module):

    def __init__(self, input_size=64, hidden_size=16,  num_classes=10) -> None:
        super(TopModel, self).__init__()
        self.encoder = self.load_encoder_model().cuda()
        self.l1 = nn.Linear(input_size, 64)
        self.l2 = nn.Linear(64, num_classes)
        self.act = nn.ReLU()
        self.sf = nn.Sigmoid()
        self.ReturnValue = namedtuple("ReturnValue", ['logits', 'raw'])

    def load_encoder_model(self):
        vae = VAE(encoder_hidden, decoder_hidden_size, latent_space, 28*28)
        vae.load_state_dict(torch.load("outputs/vae_B=2_encoder_hidden=256_decoder_hidden_size=256.pt"))
        encoder = vae.encoder
        del vae.decoder
        for param in encoder.parameters():
            param.requires_grad = False
        return encoder


    def forward(self, input_data):
        flattened = input_data.view(-1, 28 * 28)
        mu, _ = self.encoder(flattened)
        x = self.l1(mu)
        x = self.act(x)
        x = self.l2(x)
        logits = self.sf(x)
        return self.ReturnValue(logits, x)


def train(dataloader, model, optimiser, loss_func):
    accuracy = torchmetrics.Accuracy().to(device)
    for batchidx, batch in enumerate(dataloader):
        x, y = batch
        x = x.to(device)
        y = y.to(device)

        pred = model(x)

        optimiser.zero_grad()
        loss = loss_func(pred.raw, y)
        loss.backward()
        optimiser.step()
        accuracy.update(pred.logits, y)
    print(f"accuracy = {accuracy.compute()}")


def main():
    model = TopModel().to(device)
    for param in model.parameters():
        print(f"{param.shape} {param.requires_grad}")
    # optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters())) # dont want to train the vae layer
    optim = torch.optim.Adam(model.parameters())
    loss_func = nn.CrossEntropyLoss()
    for epoch in range(20):
        print(f"epoch {epoch}")
        train(train_loader, model, optim, loss_func)
    # train for 5 epochs
    # test

main()