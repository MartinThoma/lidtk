#!/usr/bin/env python

"""Create tfidf.pickle."""

# Core Library modules
import logging
import pickle
from typing import Any, Dict

# Third party modules
import click
from sklearn.feature_extraction.text import TfidfVectorizer

# First party modules
from lidtk.data import wili
from lidtk.utils import load_cfg

logger = logging.getLogger(__name__)


@click.command(name="vectorizer", help="Train Tfidf vectorizer")
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=True),
    help="Path to a YAML configuration file",
)
def main(config_file: str) -> None:
    config = load_cfg(config_file)
    data = wili.load_data()
    ret = get_features(config, data)
    analyze_vocabulary(ret)
    print("First 20 samplex of x_train:")
    print(ret["xs"]["x_train"][0])
    filepath = config["feature-extraction"]["serialization_path"]
    with open(filepath, "wb") as handle:
        logger.info(f"Store model to '{filepath}'")
        pickle.dump(ret["vectorizer"], handle, protocol=pickle.HIGHEST_PROTOCOL)


def get_features(config: Dict[str, Any], data: Dict[Any, Any]) -> Dict[str, Any]:
    """
    Get tf-idf features based on characters.

    Parameters
    ----------
    config : Dict[str, Any]
    data : Dict[Any, Any]
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
    logger.info(
        "Serialize vectorizer to "
        f"'{config['feature-extraction']['serialization_path']}'"
    )
    with open(config["feature-extraction"]["serialization_path"], "wb") as fin:
        pickle.dump(vectorizer, fin)
    for set_name in ["x_train", "x_test", "x_val"]:
        xs[set_name] = vectorizer.transform(data[set_name]).toarray()
    return {"vectorizer": vectorizer, "xs": xs}


def analyze_vocabulary(ret) -> None:
    """Show which vocabulary is used by the vectorizer."""
    voc = sorted(key for key, _ in ret["vectorizer"].vocabulary_.items())
    print(",".join(voc))
    print(f"Vocabulary: {len(voc)}")


def load_feature_extractor(config: Dict[str, Any]):
    filepath = config["feature-extraction"]["serialization_path"]
    with open(filepath, "rb") as handle:
        vectorizer = pickle.load(handle)
    return vectorizer
