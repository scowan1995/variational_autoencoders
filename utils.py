from vae import VAE
import torch
from torchtext.datasets import IMDB
from torch.utils.data import Dataset, DataLoader
from torchtext.vocab import vocab as torchtextvocab

class TorchTextDataset:

    def __init__(self, split, vectorizer, device):
        _data = IMDB(split=split)
        self.data = [(label, line) for label, line in _data]
        self.vectorizer = vectorizer
        self.device = device

    def __getitem__(self, idx):
        preprocced = self.vectorizer(self.data[idx][1])
        return self.data[idx][0], preprocced.to(self.device)

    def __len__(self):
        return len(self.data)

    @property
    def vocab(self):
        return self._vocab


class VocabularyGenerator:

    def __init__(self, split, tokenizer, min_freq):
        dataset = IMDB(split=split)
        self.tokenizer = tokenizer
        self.vocab = torchtextvocab(self._create_vocab(dataset), min_freq=min_freq)

    def _create_vocab(self, data):
        token_frequency_dict = {}
        for _, line in data:
            for token in self.tokenizer(line):
                if token not in token_frequency_dict:
                    token_frequency_dict[token] = 1
                else:
                    token_frequency_dict[token] += 1
        return token_frequency_dict

class WordCountVectorizer:

    def __init__(self, vocab, tokenizer):
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __call__(self, data: str):
        wordcounts = torch.zeros(len(self.vocab))
        for token in self.tokenizer(data):
            if token in self.vocab:
                wordcounts[self.vocab[token]] += 1
        return wordcounts


class Tokenizer:

    def __init__(self):
        from spacy.lang.en import English
        nlp = English()
        self.tokenizer = nlp.tokenizer

    def __call__(self, x):
        return self.tokenizer(x)

def get_string_representation_from_wordcounts(vector: torch.Tensor, vocab):
    wordcounts = torch.round(vector)
    string_repr = ""
    for idx, count in enumerate(wordcouts):
        if count > 0:
            token = vocab.lookup_token(idx)
            for i in range(count):
                string_repr += f"{token} "
    return string_repr.strip()


        

