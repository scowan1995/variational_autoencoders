import torch
import torch.nn.functional as F
import torch.nn as nn

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

    def __init__(self, decoder_hidden_size, latent_space_size, output_size) -> None:
        super(Decoder, self).__init__()
        self.l1 = nn.Linear(latent_space_size, decoder_hidden_size)
        self.l2 = nn.Linear(decoder_hidden_size, output_size)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, mu, logvariance, sampled_normal):
        latent_space = mu + logvariance * sampled_normal
        x = self.l1(latent_space)
        x = self.activation(x)
        raw = self.l2(x)
        return self.sigmoid(raw), raw


class VAE(nn.Module):

    def __init__(self, encoder_size: int, decoder_size: int, latent_size: int, input_size: int) -> None:
        super(VAE, self).__init__()
        self.input_size = input_size
        self.encoder = Encoder(encoder_size, latent_size, input_size)
        self.decoder = Decoder(decoder_size, latent_size, input_size)

    def forward(self, x):
        batch_size = x.shape[0]
        flattened = x.view(batch_size, self.input_size) # create shape batch_size X sample_length
        mu, logvar = self.encoder(flattened)
        z = torch.randn_like(mu)
        standard_deviation = torch.exp(0.5 * logvar)
        reconstruction, raw = self.decoder(mu, standard_deviation, z)
        return reconstruction, raw, mu, logvar
        
# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(disentangling_param, raw_recon, x, mu, logvar, writer, epoch_num, input_size):
    # using the reconnstructionw ithout the sigmoid so I can use BCE with logits loss
    # this is mosre numerically stable and stops issues with recon[i] = 1
    BCE = F.binary_cross_entropy(raw_recon, x.view(-1, input_size))

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    writer.add_scalar("BCE", BCE, epoch_num)
    writer.add_scalar("KLD", disentangling_param * KLD, epoch_num)
    return BCE + (disentangling_param * KLD)



