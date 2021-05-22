#!/usr/bin/env python
import argparse
import io
from random import random
from csv import reader as csv_reader, writer as csv_writer
from collections import defaultdict
from pathlib import Path


def parse_cli_args():
    parser = argparse.ArgumentParser(description="divide labeled dataset for experimentation")
    parser.add_argument("--train", "-t", required=True)
    parser.add_argument("--output-prefix", "-o", required=True)
    parser.add_argument("--split", "-s", type=float, default=0.8)
    return parser.parse_args()


def main():
    args = parse_cli_args()

    partners = defaultdict(set)

    header = True
    with io.open(args.train, "r", encoding="utf-8") as fin:
        data_reader = csv_reader(fin)
        for row in data_reader:
            if header:
                header = False
                continue
            pid, qid1, qid2, q1, q2, lb = row
            partners[qid1].add(qid2)
            partners[qid2].add(qid1)

    print(f"Read a total of {len(partners)} qids")

    train_qids, test_qids = set(), set()
    for qid in partners:
        if qid in train_qids or qid in test_qids:
            continue

        if random() < args.split:
            target_set = train_qids
        else:
            target_set = test_qids

        target_set.add(qid)
        partner_set = partners[qid]
        while partner_set - target_set:
            for pqid in partner_set - target_set:
                target_set.add(pqid)
                partner_set.update(partners[pqid])

    print(f"Train qids = {len(train_qids)}, test qids = {len(test_qids)}")
    print(f"Overlap = {len(train_qids & test_qids)}")

    if not Path(args.output_prefix).parent.exists():
        Path(args.output_prefix).parent.mkdir()

    header = True
    n_tr, n_te = 0, 0
    with io.open(args.train, "r", encoding="utf-8") as fin, \
            io.open(f"{args.output_prefix}.train", "w", encoding="utf-8") as ftr, \
            io.open(f"{args.output_prefix}.test", "w", encoding="utf-8") as fte:

        data_reader, train_writer, test_writer = csv_reader(fin), csv_writer(ftr), csv_writer(fte)

        for line in data_reader:

            if header:
                train_writer.writerow(line)
                test_writer.writerow(line)
                header = False
                continue

            qid1 = line[1]
            if qid1 in train_qids:
                train_writer.writerow(line)
                n_tr += 1
            elif qid1 in test_qids:
                test_writer.writerow(line)
                n_te += 1
            else:
                raise ValueError(f"Could not find partition for QID {qid1}!")

    print(f"Split base dataset into {n_tr} train queries and {n_te} test queries "
          f"({100.0 * n_tr / (n_tr + n_te):.2f}% train)")


if __name__ == "__main__":
    main()