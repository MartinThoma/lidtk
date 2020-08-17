"""Train and evaluate a MLP."""

# Core Library modules
import logging
import time
from typing import TYPE_CHECKING, Any, Dict, Optional

# Third party modules
import click
import numpy as np
from sklearn.metrics import accuracy_score

# First party modules
from lidtk.classifiers import tfidf_features as feature_extractor_module
from lidtk.data import wili
from lidtk.utils import load_cfg

if TYPE_CHECKING:
    # Third party modules
    from keras.models import Model

logger = logging.getLogger(__name__)
model_name = "mlp-3layer-tfidf-50"
model = None  # type: Optional[Model]


###############################################################################
# CLI                                                                         #
###############################################################################
@click.command(name="mlp", help=__doc__)
@click.option("--config", "config_filepath", help="Load YAML configuration file")
def main(config_filepath: str) -> None:
    """Load data, train model and evaluate it."""
    config = load_cfg(config_filepath)
    main_loaded(config, wili, feature_extractor_module)


def load_model(config: Dict[str, Any], shape) -> "Model":
    """Load a model."""
    model = create_model(wili.n_classes, shape)
    print(model.summary())
    return model


def main_loaded(config: Dict[str, Any], data_module, feature_extractor_module) -> None:
    """
    Load data, train model and evaluate it.

    Parameters
    ----------
    config : Dict[str, Any]
    data_module : Python module
    feature_extractor_module : Python module
    """
    data = data_module.load_data()
    # First party modules
    from lidtk.classifiers import tfidf_features

    vectorizer = tfidf_features.load_feature_extractor(config)
    for set_name in ["x_train", "x_val", "x_test"]:
        data[set_name] = vectorizer.transform(data[set_name]).toarray()
    for set_name in ["y_train", "y_val", "y_test"]:
        data[set_name] = wili.lang_codes_to_one_hot(data[set_name], wili.labels_s)
    optimizer = get_optimizer(config)
    logger.debug(data["x_train"][0])
    model = load_model(config, data["x_train"][0].shape)
    assert model is not None, "for mypy"
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    t0 = time.time()
    model.fit(
        data["x_train"],
        data["y_train"],
        batch_size=config["classification"]["optimizer"]["batch_size"],
        epochs=config["classification"]["optimizer"]["epochs"],
        validation_data=(data["x_val"], data["y_val"]),
        shuffle=True,
    )
    t1 = time.time()
    model.save(config["classification"]["artifacts_path"])
    logger.info(f"Save model to '{config['classification']['artifacts_path']}'")
    preds = model.predict(data["x_test"])
    y_pred = np.argmax(preds, axis=1)
    y_true = np.argmax(data["y_test"], axis=1)
    print(
        "{clf_name:<30}: {acc:>4.2f}% in {train_time:0.2f}s train".format(
            clf_name="MLP",
            acc=(accuracy_score(y_true=y_true, y_pred=y_pred) * 100),
            train_time=(t1 - t0),
        )
    )


def predict(text: str):
    """Predict the language of a text."""
    assert model is not None, "Call 'load_model' first"
    preds = model.predict(text)
    return preds


def create_model(nb_classes: int, input_shape) -> "Model":
    """Create a MLP model."""
    # Third party modules
    from keras.layers import Dense, Input
    from keras.models import Model

    input_ = Input(shape=input_shape)
    x = input_
    x = Dense(512, activation="relu")(x)
    x = Dense(nb_classes, activation="softmax")(x)
    model = Model(inputs=input_, outputs=x)
    return model


def get_optimizer(config: Dict[str, Any]):
    """Return an optimizer."""
    # Third party modules
    from keras.optimizers import Adam

    lr = config["classification"]["optimizer"]["initial_lr"]
    optimizer = Adam(lr=lr)  # Using Adam instead of SGD to speed up training
    return optimizer
