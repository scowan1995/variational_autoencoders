from vae import VAE
from torchtext.datasets import IMDB
from torch.utils.data import Dataset, DataLoader

class TorchTextDataset:

    def __init__(split: str, vectorizer):
        _data = IMDB(split=split)
        self.data = [(label, line) for label, line in _data]
        self.vectorizer = vectorizer
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        preprocced = self.vectorizer(self.data[idx][1])
        return self.data[idx][0], preprocced

    def __len__(self):
        return len(self.data)

    @property
    def vocab(self):
        return self._vocab


class Vocabulary:

    def __init__(self, split, tokenizer):
        dataset = IMDB(split=split)
        self.vocab = self._create_vocab(dataset)

    def _create_vocab(self):
        vocab_dict = {}
        next_idx = 0
        for _, line in self.data:
            for token in self.tokenizer(line):
                if token not in vocab_dict:
                    vocab_dict[token] = next_idx
                    next_idx += 1
        return vocab_dict

class WordCountVectorizer:

    def __init__(self, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __call__(self, data: str):
        wordcounts = torch.zeros(len(vocab))
        for token in self.tokenizer(data):
            if token in self.vocab:
                wordcounts[self.vocab[token]] += 1
        return wordcounts


def tokenizer(data: str):
    return  data.lower().split()



def main():
    print("in main")
    vocab = Vocabulary(split="train", tokenizer=tokenizer)
    vectorizer = WordCountVectorizer(vocab, tookenizer)
    dataset = TorchTextDataset(split="train", vectorizer=vectorizer)
    for i in range(4):
        print(dataset[i])


    # create vae

    # train and log each epoch

