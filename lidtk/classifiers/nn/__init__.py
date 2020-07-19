#!/usr/bin/env python

"""Neural Network Classifier."""

# Core Library modules
import imp
import logging
from typing import Any, Dict, Optional

# Third party modules
import click

# First party modules
import lidtk.classifiers
import lidtk.features
import lidtk.utils
from lidtk.data import wili

logger = logging.getLogger(__name__)
config = None  # type: Optional[Dict[str, Any]]


###############################################################################
# CLI                                                                         #
###############################################################################
@click.group(name="nn")
def entry_point() -> None:
    """Use a neural network classifier."""


@entry_point.command(name="predict")
@click.option("--text")
def predict_cli(text: str) -> None:
    """
    Command line interface function for predicting the language of a text.

    Parameters
    ----------
    text : str
    """
    assert config is not None, "Run lidtk.utils.load_cfg(config)"
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
def eval_wili(result_file: str, config: str) -> None:
    """
    CLI function evaluating the classifier on WiLI.

    Parameters
    ----------
    result_file : str
        Path to a file where the results will be stored
    config : str
        Path to a configuration file for the classifier
    """
    globals()["config"] = lidtk.utils.load_cfg(config)
    init_nn(globals()["config"])
    lidtk.classifiers.eval_wili(result_file, predict)  # type: ignore


###############################################################################
# Logic                                                                       #
###############################################################################
def predict(text: str) -> str:
    """
    Predict the language of a text.

    Parameters
    ----------
    text : str

    Returns
    -------
    language : str
    """
    assert config is not None, "Run lidtk.utils.load_cfg(config)"
    features = lidtk.features.extract(config, text)
    prediction = globals()["nn"].predict(features)
    return prediction


def init_nn(config: Dict[str, Any]) -> None:
    """
    Initialize a neural network.

    Parameters
    ----------
    config : Dict[str, Any]
    """
    # Third party modules
    import keras.models

    weigths = config["classifier"]["weight_path"]
    globals()["nn"] = keras.models.load_model(weigths)


@entry_point.command(name="train")
@click.option(
    "--config", default="config.yaml", show_default=True, type=click.Path(exists=True)
)
def train(config: str, data: Optional[Dict[Any, Any]] = None) -> None:
    """
    Train a neural network.

    Parameters
    ----------
    config : str
    data : Optional[Dict[Any, Any]], optional (default: wili)
    """
    assert config is not None, "Run lidtk.utils.load_cfg(config)"
    cfg = lidtk.utils.load_cfg(config)
    if data is None:
        # Read data
        data = wili.load_data()
        logger.info("Finished loading data")
    nn_module = imp.load_source("nn_module", cfg["classifier"]["script_path"])
    model = nn_module.create_model(  # type: ignore
        lidtk.features.get_dim(cfg), len(set(data["y_train"]))  # type: ignore
    )
    print(model.summary())
