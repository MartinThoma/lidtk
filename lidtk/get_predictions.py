#!/usr/bin/env python

"""Get predictions."""

# Core Library modules
import imp
import os
import pprint
import time

# Third party modules
import yaml

# First party modules
from lidtk.utils import make_paths_absolute


def get_predictions(experiment_meta, data_module, model_module):
    """
    Get predictions for one experiment.

    Parameters
    ----------
    experiment_meta : dict
    data_module : Python module
    model_module : Python module
    """
    # Get data
    data = data_module.load_data()
    model = model_module.create_model(data_module.n_classes, None)

    n_elements = 20

    t0 = time.time()
    y_pred = model.predict(data["x_test"][:n_elements])
    print(y_pred)
    t1 = time.time()
    elapsed_time = t1 - t0
    print("time: {}s".format(elapsed_time))
    print("time per element: {}s".format(elapsed_time / n_elements))

    os.makedirs(experiment_meta["artifacts_path"])
    pred_txt = os.path.join(
        experiment_meta["artifacts_path"], "predictions_langdetect.txt"
    )
    with open(pred_txt, "w") as f:
        for pred in y_pred:
            f.write("{}\n".format(pred))


def get_parser():
    """Get parser object."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "-f",
        "--file",
        dest="experiment_filepath",
        required=True,
        help="experiment file",
        metavar="FILE",
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    # Read YAML experiment definition file
    with open(args.experiment_filepath, "r") as stream:
        meta = yaml.load(stream)
    # Make paths absolute
    meta = make_paths_absolute(os.path.dirname(args.experiment_filepath), meta)

    data_module = imp.load_source("data_module", meta["dataset"]["script_path"])
    model_module = imp.load_source("data_module", meta["model"]["script_path"])

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(meta)
    get_predictions(meta, data_module, model_module)
