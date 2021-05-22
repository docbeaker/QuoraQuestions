#!/usr/bin/env python
import argparse
import logging
import torch

from random import shuffle
from typing import NamedTuple
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from numpy import round

from util import csv_file_loader
from util.vocab import Vocabulary


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, datefmt='%d-%m-%Y %H:%M:%S')
logger = logging.getLogger(__name__)


def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", "-t", required=True)
    parser.add_argument("--test", "-e", required=True)
    parser.add_argument("--batch-size", "-b", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    return parser.parse_args()


class QuestionPair(NamedTuple):
    q1: torch.Tensor
    q2: torch.Tensor
    label: int = None

    def __str__(self):
        return f"{self.q1.tolist()}\t{self.q2.tolist()}\t{self.label}"

    def __len__(self):
        return max(len(self.q1), len(self.q2))


class QuoraQuestionsDataset(Dataset):
    def __init__(self, train_filepath: str, vocab: Vocabulary):
        self.max_length = 0
        self.train_samples = self._load_data(train_filepath, vocab)

    def _load_data(self, train_filepath, vocab):
        logger.info(f"Loading data from {train_filepath}")
        train_samples = []
        positives = 0
        for q1, q2, lab in csv_file_loader(train_filepath):
            positives += lab
            q1_idx = torch.tensor(vocab.convert_tokens_to_idx(q1))
            q2_idx = torch.tensor(vocab.convert_tokens_to_idx(q2))
            train_samples.append(QuestionPair(q1_idx, q2_idx, lab))
            self.max_length = max(len(train_samples[-1]), self.max_length)
        logger.info(f"Loaded {len(train_samples)} question pairs, {positives} duplicates "
                    f"({100.0 * positives / len(train_samples):.2f}%)")
        return train_samples

    def __getitem__(self, item):
        return self.train_samples[item]

    def __len__(self):
        return len(self.train_samples)

    def shuffle(self):
        shuffle(self.train_samples)

    @classmethod
    def pad_batches(cls, batch):
        q1s, q2s = [], []
        q1_lengths, q2_lengths = [], []
        labels = []

        for x in batch:
            if not len(x.q1) or not len(x.q2):
                continue
            q1s.append(x.q1)
            q1_lengths.append(len(x.q1))
            q2s.append(x.q2)
            q2_lengths.append(len(x.q2))
            labels.append(x.label)

        labels = torch.tensor(labels)
        q1s = nn.utils.rnn.pad_sequence(q1s, batch_first=True)
        q2s = nn.utils.rnn.pad_sequence(q2s, batch_first=True)

        return q1s, q1_lengths, q2s, q2_lengths, labels


class SiameseLSTM(nn.Module):
    def __init__(self, vocab_size, pad_idx, embedding_dim=128, hidden_size=256, bilstm=True):
        super(SiameseLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.directions = 1 + int(bilstm)

        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim,
                                      padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=1,
            bias=False,
            batch_first=True,
            bidirectional=bilstm
        )

        self.register_parameter(name='bias', param=nn.Parameter(torch.randn(1), requires_grad=True))
        self.register_parameter(name='scale', param=nn.Parameter(torch.randn(1), requires_grad=True))
        self.cos = nn.CosineSimilarity()
        self.sigmoid = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h0 = torch.zeros(self.directions, batch_size, self.hidden_size, requires_grad=True)
        c0 = torch.zeros(self.directions, batch_size, self.hidden_size, requires_grad=True)
        return h0, c0

    def encoder(self, queries, query_lengths):
        x = self.embedding(queries)
        batch_size = x.size()[0]
        h0, c0 = self.init_hidden(batch_size)
        x = nn.utils.rnn.pack_padded_sequence(
            x,
            query_lengths,
            batch_first=True,
            enforce_sorted=False
        )
        x, (hn, cn) = self.lstm(x, (h0, c0))
        hn = torch.cat((hn[0], hn[1]), dim=1)
        # x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        return hn

    def forward(self, q1s, q1_l, q2s, q2_l):
        q1h = self.encoder(q1s, q1_l)
        q2h = self.encoder(q2s, q2_l)
        cos = self.scale * self.cos(q1h, q2h) + self.bias
        return self.sigmoid(cos)

    def predict_on_batch(self, sampled_batch):
        return self.forward(*sampled_batch[:-1]).to(torch.float32)


def train_model(model: SiameseLSTM, n_epochs, train_data_loader, eval_data_loader,
                val_steps=5):
    logger.info(f"Training Siamese LSTM")
    model.train()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epi in range(n_epochs):
        cum_loss, samples = 0.0, 0
        for i_batch, sampled_batch in enumerate(train_data_loader):
            predictions = model.forward(*sampled_batch[:-1]).to(torch.float32)
            labels = sampled_batch[-1].to(torch.float32)

            loss = loss_fn(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cum_loss += loss.item()
            samples += labels.size()[0]

            if i_batch % val_steps == 0 and i_batch > 0:
                acc, p, r, f1 = evaluate_model(model, eval_data_loader)
                for n in [acc, p, r, f1]:
                    print(type(n))
                logger.info(f"Epoch {epi}, batch {i_batch}, "
                            f"loss = {cum_loss / samples:.4f}, "
                            f"validation accuracy = {100 * acc:.2f}%, "
                            f"precision = {p:.2f}, "
                            f"recall = {r:.2f}, "
                            f"f1 = {f1:.2f}")
                cum_loss, samples = 0.0, 0


def evaluate_model(model: SiameseLSTM, data_loader: DataLoader):
    logger.info(f"Evaluating model")
    y_true, y_pred = [], []
    for sampled_batch in data_loader:
        labels = sampled_batch[-1].to(torch.float32)
        predictions = model.predict_on_batch(sampled_batch)

        y_true.extend(labels.tolist())
        y_pred.extend(round(predictions.detach()).tolist())

    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    return acc, p, r, f1


def main():
    args = parse_cli_args()

    logger.info("Building vocabulary")
    vocab = Vocabulary()
    vocab.construct_vocab_from_files([args.train, args.test])
    logger.info(f"Created vocab of size {len(vocab)}")

    # Create the datasets
    train_data = QuoraQuestionsDataset(args.train, vocab)
    test_data = QuoraQuestionsDataset(args.test, vocab)

    train_data.shuffle()
    split_idx = int(0.9 * len(train_data))
    train_data, eval_data = train_data[:split_idx], train_data[split_idx:]

    # Create the data loaders
    data_loaders = []
    for data_type in [train_data, eval_data, test_data]:
        data_loaders.append(DataLoader(
            data_type,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=QuoraQuestionsDataset.pad_batches
        ))

    # Initialize the network
    slstm = SiameseLSTM(len(vocab), vocab.PAD_IDX)

    # Train the model
    train_model(slstm, args.epochs, data_loaders[0], data_loaders[1])


if __name__ == "__main__":
    main()
