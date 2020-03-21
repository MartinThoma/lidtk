#!/usr/bin/env python

"""
Run classification with tfidf-features and Neural Network classifier.

tfidf = text frequency, inverse document frequency
"""

# Core Library modules
import os
import pickle

# Third party modules
import click
import numpy as np
import pkg_resources

# First party modules
import lidtk.classifiers.mlp
import lidtk.classifiers.tfidf_features
from lidtk.data import wili

classifier_name = "tfidf_nn"
classifier = None


def load_classifier(filepath):
    """
    Load a TfidfNNClassifier.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    classifier : TfidfNNClassifier object
    """
    if filepath is None:
        filepath = "classifiers/config/tfidf_nn.yaml"
        filepath = pkg_resources.resource_filename("lidtk", filepath)
    classifier = TfidfNNClassifier(filepath)
    classifier.load(
        classifier.cfg["feature-extraction"]["serialization_path"],
        classifier.cfg["classification"]["artifacts_path"],
    )
    globals()["classifier"] = classifier
    return classifier


class TfidfNNClassifier(lidtk.classifiers.LIDClassifier):
    """LID with the TfidfNNClassifier."""

    def __init__(self, filepath):
        super().__init__(filepath)
        self.labels = wili.labels

    def load(self, vectorizer_filename, classifier_filename):
        from keras.models import load_model

        with open(vectorizer_filename, "rb") as handle:
            self.vectorizer = pickle.load(handle)
        self.model = load_model(classifier_filename)

    def predict(self, text):
        """
        Predicting the language of a text.

        Parameters
        ----------
        text : str
        """
        features = self.vectorizer.transform([text]).toarray()
        prediction = self.model.predict(features)
        most_likely = np.argmax(prediction, axis=1)
        most_likely = [self.map2wili(index) for index in most_likely]
        return most_likely[0]


###############################################################################
# CLI                                                                         #
###############################################################################
@click.group(name=classifier_name)
def entry_point():
    """Use the TfidfNNClassifier classifier."""


@entry_point.group(name="train")
def train_entry_point():
    """Train the TfidfNNClassifier classifier."""


train_entry_point.add_command(lidtk.classifiers.tfidf_features.main)
train_entry_point.add_command(lidtk.classifiers.mlp.main)


@entry_point.command(name="predict")
@click.option("--text")
@click.option(
    "--config",
    "config_filepath",
    type=click.Path(exists=True),
    help="Path to a YAML configuration file",
)
def predict_cli(text, config_filepath):
    """
    Command line interface function for predicting the language of a text.

    Parameters
    ----------
    text : str
    config_filepath : str
        Path to a YAML configuration file.
    """
    load_classifier(config_filepath)
    print(classifier.predict(text))


@entry_point.command(name="get_languages")
@click.option(
    "--config",
    "config_filepath",
    type=click.Path(exists=True),
    help="Path to a YAML configuration file",
)
def get_languages(config_filepath):
    """
    Get all predicted languages of for the WiLI dataset.

    Parameters
    ----------
    config_filepath : str
        Path to a YAML configuration file.
    """
    load_classifier(config_filepath)
    print(classifier.get_languages())


@entry_point.command(name="print_languages")
@click.option(
    "--label_filepath",
    required=True,
    type=click.Path(exists=True),
    help="CSV file with delimiter ;",
)
@click.option(
    "--config",
    "config_filepath",
    type=click.Path(exists=True),
    help="Path to a YAML configuration file",
)
def print_languages(config_filepath, label_filepath):
    """
    Print supported languages of classifier.

    Parameters
    ----------
    config_filepath : str
        Path to a YAML configuration file.
    label_filepath : str
    """
    load_classifier(config_filepath)
    label_filepath = os.path.abspath(label_filepath)
    wili_labels = wili.get_language_data(label_filepath)
    iso2name = {el["ISO 369-3"]: el["English"] for el in wili_labels}
    print(
        ", ".join(
            sorted(
                [
                    iso2name[iso]
                    for iso in classifier.get_mapping_languages()
                    if iso != "UNK"
                ]
            )
        )
    )


@entry_point.command(name="wili")
@click.option(
    "--result_file",
    default="{}_results.txt".format(classifier_name),
    show_default=True,
    help="Where to store the predictions",
)
@click.option(
    "--config",
    "config_filepath",
    type=click.Path(exists=True),
    help="Path to a YAML configuration file",
)
def eval_wili(config_filepath, result_file):
    """
    CLI function evaluating the classifier on WiLI.

    Parameters
    ----------
    config_filepath : str
        Path to a YAML configuration file.
    result_file : str
        Path to a file where the results will be stored
    """
    load_classifier(config_filepath)
    classifier.eval_wili(result_file)
