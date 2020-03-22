#!/usr/bin/env python

"""Feature Extraction module."""

# Core Library modules
import pickle

# Third party modules
from sklearn.feature_extraction.text import TfidfVectorizer


def extract(cfg, text):
    """
    Extract features.

    Parameters
    ----------
    cfg : dict

    Returns
    -------
    features : object
    """
    if cfg["features"]["type"] == "raw":
        return text
    elif cfg["features"]["type"] == "tfidf":
        return get_tfidif_features(cfg, [text])[0]
    else:
        raise NotImplementedError(f"Feature: {cfg['features']['type']}")


def get_dim(cfg):
    """
    Get the dimension of the extracted features.

    Parameters
    ----------
    cfg : dict

    Returns
    -------
    feature_dim : int
    """
    if cfg["features"]["type"] == "raw":
        raise NotImplementedError(f"Feature: {cfg['features']['type']}")
    elif cfg["features"]["type"] == "tfidf":
        pass  # TODO
    else:
        raise NotImplementedError(f"Feature: {cfg['features']['type']}")


def train_tfidf_features(config, data):
    """
    Get tf-idf features based on characters.

    Parameters
    ----------
    config : dict
    data : dict
    """
    if config is None:
        config = {}
    if "features" not in config:
        config["features"] = {}
    if "min_df" not in config["features"]:
        config["features"]["min_df"] = 50
    vectorizer = TfidfVectorizer(
        analyzer="char",
        min_df=config["features"]["min_df"],
        lowercase=config["features"]["lowercase"],
        norm=config["features"]["norm"],
    )
    xs = {}
    vectorizer.fit(data["x_train"])
    # Serialize trained vectorizer
    with open(config["features"]["name"], "wb") as fin:
        pickle.dump(vectorizer, fin)
    for set_name in ["x_train", "x_test", "x_val"]:
        xs[set_name] = vectorizer.transform(data[set_name]).toarray()
    return {"vectorizer": vectorizer, "xs": xs}


def get_tfidif_features(cfg, samples):
    """
    Get Tf-idf features for samples.

    Parameters
    ----------
    cfg : dict
    samples : ndarray

    Returns
    -------
    tfidf_features : ndarray
    """
    if "vectorizer" not in cfg["features"]:
        # Load data (deserialize)
        with open(cfg["features"]["vectorizer_path"], "rb") as handle:
            vectorizer = pickle.load(handle)
        cfg["features"]["vectorizer"] = vectorizer
    return vectorizer.transform(cfg["features"]["vectorizer"])
