import torch
import os
from datetime import datetime
import pdb
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torchvision
from torch.utils.tensorboard import SummaryWriter
import optuna
from functools import partial

bsize = 64
kwargs = {'num_workers': 1, 'pin_memory': False}
train_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=True, download=True,
                   transform=transforms.ToTensor()),
    batch_size=bsize, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=False, transform=transforms.ToTensor()),
    batch_size=bsize, shuffle=True, **kwargs)



mnist_size = 28 * 28

class Encoder(nn.Module):

    def __init__(self, encoder_hidden_size, latent_space_size, input_size):
        super(Encoder, self).__init__()
        self.l1 = nn.Linear(input_size, 768)
        self.l2 = nn.Linear(768, encoder_hidden_size)
        self.l3 = nn.Linear(encoder_hidden_size, encoder_hidden_size)
        self.activation = nn.ReLU()
        self.mean_encoder = nn.Linear(encoder_hidden_size, latent_space_size)
        self.logvar_encoder = nn.Linear(encoder_hidden_size, latent_space_size)

    def forward(self, x):
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)
        x = self.activation(x)
        x = self.l3(x)
        x = self.activation(x)
        mu = self.mean_encoder(x)
        logvar = self.logvar_encoder(x)
        return mu, logvar


class Decoder(nn.Module):

    def __init__(self, decoder_hidden_size, latent_space_size) -> None:
        super(Decoder, self).__init__()
        self.l1 = nn.Linear(latent_space_size, decoder_hidden_size)
        self.l2 = nn.Linear(decoder_hidden_size, mnist_size)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, mu, logvariance, sampled_normal):
        latent_space = mu + logvariance * sampled_normal
        x = self.l1(latent_space)
        x = self.activation(x)
        raw = self.l2(x)
        return self.sigmoid(raw), raw


class VAE(nn.Module):

    def __init__(self, encoder_size, decoder_size, latent_size, input_size) -> None:
        super(VAE, self).__init__()
        self.input_size = input_size
        self.encoder = Encoder(encoder_size, latent_size, input_size)
        self.decoder = Decoder(decoder_size, latent_size)

    def forward(self, x):
        batch_size = x.shape[0]
        flattened = x.view(batch_size, mnist_size)
        mu, logvar = self.encoder(flattened)
        z = torch.randn_like(mu)
        standard_deviation = torch.exp(0.5 * logvar)
        reconstruction, raw = self.decoder(mu, standard_deviation, z)
        return reconstruction, raw, mu, logvar
        
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(disentangling_param, raw_recon, x, mu, logvar):
    # using the reconnstructionw ithout the sigmoid so I can use BCE with logits loss
    # this is mosre numerically stable and stops issues with recon[i] = 1
    l = torch.nn.BCEWithLogitsLoss(reduction='sum')
    BCE = l(raw_recon, x.view(-1, 784))

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + (disentangling_param * KLD)

def train(num_epochs: int, dataloader, model, loss_fn, optimizer, comment=""):
    writer = SummaryWriter(comment=comment)
    unflatten = nn.Unflatten(1, (28, 28))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mel = 0.
    for epoch in tqdm(range(num_epochs)):
        dataloader_size = len(dataloader)
        running_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            images, targets = batch
            if batch_idx == 1:
                grid = torchvision.utils.make_grid(images)
                writer.add_image(f'{epoch}_original', grid, global_step=epoch)
            if len(targets) < 2:
                raise ValueError("Batch size must be at least 2")
            images = torch.squeeze(images)
            images = images.to(device)
            targets = targets.to(device)

            reconstruction, raw, mu, logvar = model(images)
            if batch_idx == 1:
                recon = unflatten(reconstruction)
                # add colour channel
                recon = recon.unsqueeze(1)
                recon_grid = torchvision.utils.make_grid(recon)
                writer.add_image(f'{epoch}_recon', recon_grid, global_step=epoch)
            loss = loss_fn(raw, images, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_val = loss.detach().cpu().item()
            running_loss += loss_val
            global_step = (epoch * len(dataloader) + batch_idx)
            writer.add_scalar("loss", loss_val, global_step)
        mel = running_loss / (dataloader_size * bsize)
        writer.add_scalar("mean epoch loss", mel, epoch)
    return mel


def objective(trial):
    lr = trial.suggest_float("learning_rate", 0.00001, 0.1)
    encoder_hidden: int = trial.suggest_categorical("Encoder_hidden_size_1", [32, 64, 128, 256])
    divisor: int = trial.suggest_categorical("latent_size_divisor", [1, 2, 4, 8])
    latent_space = encoder_hidden // divisor
    decoder_hidden_size: int = trial.suggest_categorical("decoder_hidden_size_1", [32, 64, 128, 256, 512])
    print(f"{encoder_hidden=}\n{divisor=}\n{decoder_hidden_size=}\n{lr=}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(encoder_hidden, decoder_hidden_size, latent_space, 28*28).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    last_mean_loss = train(10, train_loader, model, loss_function, optimizer)
    print(f"{last_mean_loss=}\n")
    return last_mean_loss

def tune():
    # I have a feeling the loss is not what I want to tune for
    # since the loss may be larger for a larger latent space
    # TODO check this
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    fig = optuna.visualization.plot_optimization_history(study)
    fig.show()

def train_full():
    latent_space = 64
    encoder_hidden = 256
    B = 2
    decoder_hidden_size = 256
    os.makedirs("outputs", exist_ok = True)
    modelpath = "outputs"
    lr = 0.00254
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(encoder_hidden, decoder_hidden_size, latent_space, 28*28).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    last_mean_loss = train(20, train_loader, model, partial(loss_function,B), optimizer, comment=f"{B}_{encoder_hidden}_{decoder_hidden_size}")
    print(f"{last_mean_loss=}\n")
    torch.save(model.state_dict(), f"{modelpath}/vae_{B=}_{encoder_hidden=}_{decoder_hidden_size=}.pt")


            
if __name__ == "__main__":
    train_full()




