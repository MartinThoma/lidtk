#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create tfidf.pickle."""

# Core Library modules
import logging
import pickle

# Third party modules
import click
from sklearn.feature_extraction.text import TfidfVectorizer

# First party modules
from lidtk.data import wili
from lidtk.utils import load_cfg


@click.command(name="vectorizer", help="Train Tfidf vectorizer")
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True),
    help="Path to a YAML configuration file",
)
def main(config_file):
    config = load_cfg(config_file)
    data = wili.load_data()
    ret = get_features(config, data)
    analyze_vocabulary(ret)
    print("First 20 samplex of x_train:")
    print(ret["xs"]["x_train"][0])
    filepath = config["feature-extraction"]["serialization_path"]
    with open(filepath, "wb") as handle:
        logging.info("Store model to '{}'".format(filepath))
        pickle.dump(ret["vectorizer"], handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_features(config, data):
    """
    Get tf-idf features based on characters.

    Parameters
    ----------
    config : dict
    data : dict
    """
    if config is None:
        config = {}
    if "feature-extraction" not in config:
        config["feature-extraction"] = {}
    if "min_df" not in config["feature-extraction"]:
        config["feature-extraction"]["min_df"] = 50
    make_lowercase = config["feature-extraction"]["lowercase"]
    vectorizer = TfidfVectorizer(
        analyzer="char",
        min_df=config["feature-extraction"]["min_df"],
        lowercase=make_lowercase,
        norm=config["feature-extraction"]["norm"],
    )
    xs = {}
    vectorizer.fit(data["x_train"])
    # Serialize trained vectorizer
    logging.info(
        "Serialize vectorizer to '{}'".format(
            config["feature-extraction"]["serialization_path"]
        )
    )
    with open(config["feature-extraction"]["serialization_path"], "wb") as fin:
        pickle.dump(vectorizer, fin)
    for set_name in ["x_train", "x_test", "x_val"]:
        xs[set_name] = vectorizer.transform(data[set_name]).toarray()
    return {"vectorizer": vectorizer, "xs": xs}


def analyze_vocabulary(ret):
    """Show which vocabulary is used by the vectorizer."""
    voc = sorted([key for key, _ in ret["vectorizer"].vocabulary_.items()])
    print(",".join(voc))
    print("Vocabulary: {}".format(len(voc)))


def load_feature_extractor(config):
    filepath = config["feature-extraction"]["serialization_path"]
    with open(filepath, "rb") as handle:
        vectorizer = pickle.load(handle)
    return vectorizer
