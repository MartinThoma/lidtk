#!/usr/bin/env python

"""Create sharable dataset from downloaded texts."""

# Core Library modules
import glob
import logging
import os
import random
import re
import sys
import unicodedata

# Third party modules
import click

# First party modules
import lidtk
from lidtk.data import language_utils

random.seed(0)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.DEBUG,
    stream=sys.stdout,
)


def normalize_data(paragraph):
    """
    Bring unicode in one form.

    Some symbols can be written in multiple ways.
    """
    paragraph = unicodedata.normalize("NFC", paragraph)
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    return paragraph


@click.command(name="create-dataset", help=__doc__)
@click.option("--nb_elements", default=1000, show_default=True)
@click.option(
    "--source_path",
    default="~/.lidtk/lang/*.pickle",
    show_default=True,
    help="Path to directory where the source pickle files are.",
)
@click.option(
    "--data_path",
    default="~/.lidtk/data",
    show_default=True,
    help="Path to directory where the created dataset gets stored.",
)
def main(nb_elements, source_path, data_path):
    """
    Create dataset.

    Takes all pickle files with the `lang` directory and creates five files:

    * x_train.txt
    * y_train.txt
    * x_test.txt
    * y_test.txt
    * urls.txt
    """
    set_names = ["train", "test"]
    xs = {"train": [], "test": []}
    ys = {"train": [], "test": []}
    urls = []
    lang_path = lidtk.utils.make_path_absolute(source_path)
    files = sorted(glob.glob(lang_path))
    for filepath in files:
        filename = os.path.split(filepath)[1]
        wiki_code = os.path.splitext(filename)[0]
        label = language_utils.get_label(wiki_code)
        lang_data_p = language_utils.read_language_file(filepath)

        lang_data = lang_data_p["paragraphs"]
        used_pages = lang_data_p["used_pages"]
        for page_id in used_pages:
            urls.append(
                "https://{lang}.wikipedia.org/w/index.php?oldid={id}".format(
                    lang=wiki_code, id=page_id
                )
            )

        # normalize
        lang_data = [normalize_data(el) for el in lang_data]

        # Define permutation and apply it to data
        indices = list(range(nb_elements))
        nb_train = int(nb_elements / 2)
        random.shuffle(indices)
        indices = {"train": indices[:nb_train], "test": indices[nb_train:]}
        for set_name in set_names:
            for i in indices[set_name]:
                xs[set_name].append(lang_data[i])
                ys[set_name].append(label)
                # urls[set_name].append()
        # print("{:>3}: {}".format(label, filepath))
    for set_name in set_names:
        # Prevent languages from building blocks:
        perm = list(range(len(xs[set_name])))
        random.shuffle(perm)
        xs[set_name] = [xs[set_name][i] for i in perm]
        ys[set_name] = [ys[set_name][i] for i in perm]
        # urls[set_name] = [urls[set_name][i] for i in perm]

        # Write data
        data_path = lidtk.utils.make_path_absolute(data_path)
        dataset_filepath = os.path.join(data_path, "x_{}.txt".format(set_name))
        logging.debug("Write dataset_filepath={}".format(dataset_filepath))
        with open(dataset_filepath, "w") as f:
            for el in xs[set_name]:
                f.write(el + "\n")

        labels_filepath = os.path.join(data_path, "y_{}.txt".format(set_name))
        with open(labels_filepath, "w") as f:
            for el in ys[set_name]:
                f.write(el + "\n")

        urls_filepath = os.path.join(data_path, "urls_{}.txt".format(set_name))
        with open(urls_filepath, "w") as f:
            for el in urls:  # [set_name]
                f.write(el + "\n")
    logging.info("Done writing files to {}".format(data_path))
