import io

from string import punctuation
from csv import reader as csv_reader


class TextNormalizer(object):

    @classmethod
    def normalize(cls, text):
        return text.lower().translate(str.maketrans('', '', punctuation))


def csv_file_loader(filepath, header=True, include_ids=False):
    with io.open(filepath, "r", encoding="utf-8") as fin:
        data_reader = csv_reader(fin)
        for row in data_reader:
            if header:
                header = False
                continue
            pid, qid1, qid2, q1, q2, label = row
            pid = int(pid)
            qid1 = int(qid1)
            qid2 = int(qid2)
            label = int(label)
            if include_ids:
                yield pid, qid1, qid2, q1, q2, label
            else:
                yield q1, q2, label,