#!/usr/bin/env python

"""Neural Network Classifier."""

# Core Library modules
import imp
import logging

# Third party modules
import click

# First party modules
import lidtk.classifiers
import lidtk.features
import lidtk.utils
from lidtk.data import wili

config = None


###############################################################################
# CLI                                                                         #
###############################################################################
@click.group(name="nn")
def entry_point():
    """Use a neural network classifier."""
    pass


@entry_point.command(name="predict")
@click.option("--text")
def predict_cli(text):
    """
    Command line interface function for predicting the language of a text.

    Parameters
    ----------
    text : str
    """
    init_nn(config)
    print(predict(text))


@entry_point.command(name="wili")
@click.option(
    "--result_file",
    default="char_dist_metric_results.txt",
    show_default=True,
    help="Where to store the predictions",
)
@click.option(
    "--config",
    default="config.yaml",
    show_default=True,
    type=click.Path(exists=True),
    help="configuration file for the classifier",
)
def eval_wili(result_file, config):
    """
    CLI function evaluating the classifier on WiLI.

    Parameters
    ----------
    result_file : str
        Path to a file where the results will be stored
    """
    globals()["config"] = lidtk.utils.load_cfg(config)
    init_nn(config)
    lidtk.classifiers.eval_wili(result_file, predict)


###############################################################################
# Logic                                                                       #
###############################################################################
def predict(text):
    """
    Predict the language of a text.

    Parameters
    ----------
    text : str

    Returns
    -------
    language : str
    """
    features = lidtk.features.extract(config, text)
    prediction = globals()["nn"].predict(features)
    return prediction


def init_nn(config):
    """
    Initialize a neural network.

    Parameters
    ----------
    config : dict
    """
    import keras.models

    weigths = config["classifier"]["weight_path"]
    globals()["nn"] = keras.models.load_model(weigths)


@entry_point.command(name="train")
@click.option(
    "--config", default="config.yaml", show_default=True, type=click.Path(exists=True)
)
def train(config, data=None):
    """
    Train a neural network.

    Parameters
    ----------
    config : dict
    data : dict, optional (default: wili)
    """
    cfg = lidtk.utils.load_cfg(config)
    if data is None:
        # Read data
        data = wili.load_data()
        logging.info("Finished loading data")
    nn_module = imp.load_source("nn_module", cfg["classifier"]["script_path"])
    model = nn_module.create_model(
        lidtk.features.get_dim(cfg), len(set(data["y_train"]))
    )
    print(model.summary())
