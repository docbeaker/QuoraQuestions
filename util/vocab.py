from . import TextNormalizer, csv_file_loader
from collections import Counter


class Vocabulary(object):
    PAD, PAD_IDX = "[PAD]", 0
    UNK, UNK_IDX = "[UNK]", 1
    SEP, SEP_IDX = "[SEP]", 2

    def __init__(self):
        self.word2idx = {
            self.PAD: self.PAD_IDX, self.UNK: self.UNK_IDX, self.SEP: self.SEP_IDX
        }

    def construct_vocab_from_files(self, filepaths, min_count=0):
        word_counter = Counter()
        for fp in filepaths:
            for q1, q2, _ in csv_file_loader(fp):
                for q in [q1, q2]:
                    for tok in TextNormalizer.normalize(q).split():
                        if not min_count and tok not in self.word2idx:
                            self.word2idx[tok] = len(self.word2idx)
                        elif min_count:
                            word_counter[tok] += 1
                        else:
                            pass

        if word_counter:
            for word, cnt in word_counter.most_common():
                if cnt < min_count:
                    return
                self.word2idx[word] = len(self.word2idx)

    def __getitem__(self, item):
        return self.word2idx.get(item, self.UNK_IDX)

    def __len__(self):
        return len(self.word2idx)

    def convert_tokens_to_idx(self, sentence):
        sentence = TextNormalizer.normalize(sentence)
        return [self[tok] for tok in sentence.split()]
