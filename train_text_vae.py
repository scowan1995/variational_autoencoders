from vae import VAE, loss_function
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils import WordCountVectorizer, VocabularyGenerator, TorchTextDataset, get_string_representation_from_wordcounts
import torchtext
from tqdm import tqdm
from functools import partial
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

def train(model, dataloader, optimizer, lossfunc, num_epochs):
    for ep in range(num_epochs):
        print(f"On epoch {ep}")
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            labels, lines = batch

            optimizer.zero_grad()
            reconstruction, raw_recon, mu, logvar = model(lines)
            loss = lossfunc(raw_recon=reconstruction, x=lines, mu=mu, logvar=logvar, epoch_num=ep)
            if batch_idx % 200 == 0:
                print(f"{loss.item()}")


            loss.backward()
            optimizer.step()


def validate(model, val_dl, vocab, print_batch):
    mse = 0.0
    num_batches = 0
    for batch_idx, batch in enumerate(val_dl):
        with torch.zero_grad():
            _, lines = batch
            recon, _, _, _ = model(lines)
            mse += torch.nn.functional.mse_loss(recon, lines)
            if print_batch and batch_idx == 1:
                print(f"{mse.item()=}")
                print("lines:")
                print(f"{get_string_representation_from_wordcounts(lines, vocab)}")

                print("recon:")
                print(f"{get_string_representation_from_wordcounts(recon, vocab)}")
    return mse / num_batches




def main():
    summarywriter = SummaryWriter()
    print("in main")
    m = 3
    vocab_obj = VocabularyGenerator(split="train", tokenizer=torchtext.data.utils.get_tokenizer("basic_english"), min_freq=m)
    vocab = vocab_obj.vocab
    print(f"With minimum freq {m} vocab has {len(vocab)} tokens")
    
    vectorizer = WordCountVectorizer(vocab, torchtext.data.utils.get_tokenizer("basic_english"))
    print("creating dataset")
    dataset = TorchTextDataset(split="train", vectorizer=vectorizer, device=device)
    for i in range(4):
        print(dataset[i])

    dataloader = DataLoader(dataset, batch_size=16)

    model = VAE(encoder_size=512, decoder_size=512, latent_size=128, input_size = len(vocab))
    model.to(device)

    
    lossfunc = partial(loss_function, disentangling_param=1, input_size=len(vocab), writer=summarywriter)
    sgd = optim.SGD(model.parameters(), lr=0.00001, momentum = 0.9)
    # possibly pass in metrics to compute
    train(model, dataloader, sgd, lossfunc, num_epochs=5)
    dataset = TorchTextDataset(split="test", vectorizer=vectorizer, device=device)
    dataloader = DataLoader(dataset, batch_size=16)
    validate(model, dataloader, vocab, True)

    # train and log each epoch
main()
