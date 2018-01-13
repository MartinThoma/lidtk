#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a CSV file which contains only an identifier and the truth.

The first value is the identifier, the second is the index of the true class.
"""

# core modules
import csv

# internal modules
from lidtk.data import wili


def generate_truth_csv(out_filepath):
    labels_all = sorted(wili.labels_s)
    data = wili.load_data()
    # Read CSV file
    ys = [(index, labels_all.index(row))
          for index, row in enumerate(data['y_test'])]

    # Write CSV file
    with open(out_filepath, 'w') as fp:
        writer = csv.writer(fp, delimiter=';')
        writer.writerows(ys)


def get_parser():
    """Get parser object for script xy.py."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("--out",
                        dest="out_filepath",
                        required=True,
                        help="write report to FILE",
                        metavar="FILE")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    generate_truth_csv(args.out_filepath)
