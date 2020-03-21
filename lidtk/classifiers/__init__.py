#!/usr/bin/env python

"""Language classifiers."""

# Core Library modules
import datetime
import io
import json
import logging
import os
import time
from abc import ABC, abstractmethod

# Third party modules
import click
import numpy as np
import progressbar

# First party modules
import lidtk.utils
from lidtk.data import wili


class LIDClassifier(ABC):
    """
    A classifier for identifying languages.

    Parameters
    ----------
    cfg : dict
    """

    def __init__(self, cfg_path):
        """Constructor."""
        cfg_path = os.path.abspath(cfg_path)
        self.cfg = lidtk.utils.load_cfg(cfg_path)

    def map2wili(self, services_code):
        """
        Map the classifiers code to ISO 369-3 code.

        Parameters
        ----------
        services_code : str

        Returns
        -------
        iso_369_3 : str
        """
        return self.cfg["mapping"].get(services_code, "UNK")

    @abstractmethod
    def predict(self, text):
        """
        Predict the language of the given text.

        Parameters
        ----------
        text : str

        Returns
        -------
        language : str
            ISO 369-3 code
        """

    def predict_bulk(self, texts):
        """
        Predict the language of a list of texts.

        Parameters
        ----------
        texts : list of str

        Returns
        -------
        languages : list of str
            List of ISO 369-3 codes or UNK
        """
        return [self.predict(text) for text in texts]

    def get_languages(self):
        """
        Find which languages are predicted for the WiLI dataset.

        Returns
        -------
        languages : list of str
            Each str is a ISO 369-3 code
        """
        languages = set()
        # Read data
        data = wili.load_data()
        logging.info("Finished loading data")
        for set_name in ["test", "train"]:
            x_set_name = "x_{}".format(set_name)
            bar = progressbar.ProgressBar(
                redirect_stdout=True, max_value=len(data[x_set_name])
            )
            for i, el in enumerate(data[x_set_name]):
                try:
                    predicted = self.predict(el)
                except Exception as e:
                    predicted = "UNK"
                    logging.error({"message": "Exception in get_languages", "error": e})
                languages.add(predicted)
                bar.update(i + 1)
            bar.finish()
        return sorted(list(languages))

    def get_mapping_languages(self):
        """
        Get the languages supported by th classifier and supported by WiLI.

        Returns
        -------
        languages : list of str
            Each str is a ISO 369-3 code
        """
        return sorted([lang for _, lang in self.cfg["mapping"].items()])

    def eval_wili(self, result_file, languages=None, eval_unk=False):
        """
        Evaluate the classifier on WiLI.

        Parameters
        ----------
        result_file : str
            Path to a file where the results will be stored
        languages : list, optional (default: All languages)
            Filter languages by this list
        """
        # Read data
        data = wili.load_data()
        logging.info("Finished loading data")
        times = []
        bar = progressbar.ProgressBar(
            redirect_stdout=True, max_value=len(data["x_test"])
        )
        result_filepath = os.path.abspath(result_file)
        logging.info("Write results to {}".format(result_filepath))
        results = {"meta": {}}
        now = datetime.datetime.now()
        results["meta"]["experiment_start"] = "{:%Y-%m-%d %H:%M:%S}".format(now)
        cl_results = {}
        if languages is None:
            eval_unk = False
        with open(result_filepath, "w") as filepointer:
            for i, (el, label_t) in enumerate(zip(data["x_test"], data["y_test"])):
                if languages is not None:
                    if label_t not in languages:
                        if eval_unk:
                            print("UNK")
                        else:
                            continue
                    else:
                        print(label_t)
                try:
                    t0 = time.time()
                    predicted = self.predict(el)
                    t1 = time.time()
                    times.append(t1 - t0)
                    bar.update(i + 1)
                    if label_t != predicted:
                        if label_t not in cl_results:
                            cl_results[label_t] = {}
                        if predicted not in cl_results[label_t]:
                            cl_results[label_t][predicted] = []
                        identifier = "test_{}".format(i)
                        cl_results[label_t][predicted].append([identifier, el])
                except Exception as e:  # catch them all
                    logging.error({"message": "Exception in eval_wili", "error": e})
                    predicted = "UNK-exception"
                filepointer.write(predicted + "\n")
        bar.finish()
        results["cl_results"] = cl_results
        times = np.array(times)
        print("Average time per 10**6 elements: {:.2f}s".format(times.mean() * 10 ** 6))
        results["time_per_10*6"] = times.mean() * 10 ** 6
        logfile = result_filepath + ".json"
        results["meta"]["hardware"] = lidtk.utils.get_hardware_info()
        results["meta"]["software"] = lidtk.utils.get_software_info()
        with io.open(logfile, "w", encoding="utf8") as f:
            f.write(json.dumps(results, indent=4, sort_keys=True, ensure_ascii=False))


def classifier_cli_factor(classifier):
    """
    Create the CLI for a classifier.

    Parameters
    ----------
    classifier : lidtk.classifiers.LIDClassifier object

    Returns
    -------
    entry_point : function
    """

    @click.group(name=classifier.cfg["name"])
    def entry_point():
        """Use this language classifier."""

    @entry_point.command(name="predict")
    @click.option("--text")
    def predict_cli(text):
        """
        Command line interface function for predicting the language of a text.

        Parameters
        ----------
        text : str
        """
        print(classifier.predict(text))

    @entry_point.command(name="get_languages")
    def get_languages():
        """Get all predicted languages of for the WiLI dataset."""
        print(classifier.get_languages())

    @entry_point.command(name="print_languages")
    @click.option(
        "--label_filepath",
        required=True,
        type=click.Path(exists=True),
        help="CSV file with delimiter ;",
    )
    def print_languages(label_filepath):
        """
        Print supported languages of classifier.

        Parameters
        ----------
        label_filepath : str
        """
        label_filepath = os.path.abspath(label_filepath)
        wili_labels = wili.get_language_data(label_filepath)
        iso2name = {el["ISO 369-3"]: el["English"] for el in wili_labels}
        print(
            ", ".join(
                sorted(
                    [
                        iso2name.get(iso, iso)
                        for iso in classifier.get_mapping_languages()
                        if iso != "UNK"
                    ]
                )
            )
        )

    @entry_point.command(name="wili")
    @click.option(
        "--result_file",
        default="{}_results.txt".format(classifier.cfg["name"]),
        show_default=True,
        help="Where to store the predictions",
    )
    def eval_wili(result_file):
        """
        CLI function evaluating the classifier on WiLI.

        Parameters
        ----------
        result_file : str
            Path to a file where the results will be stored
        """
        classifier.eval_wili(result_file)

    @entry_point.command(name="wili_k")
    @click.option(
        "--result_file",
        default=("{}_results_known.txt".format(classifier.cfg["name"])),
        show_default=True,
        help="Where to store the predictions",
    )
    def eval_wili_known(result_file):
        """
        CLI function evaluating the classifier on WiLI.

        Parameters
        ----------
        result_file : str
            Path to a file where the results will be stored
        """
        classifier.eval_wili(result_file, classifier.get_mapping_languages())

    @entry_point.command(name="wili_unk")
    @click.option(
        "--result_file",
        default=("{}_results_unknown.txt".format(classifier.cfg["name"])),
        show_default=True,
        help="Where to store the predictions",
    )
    def eval_wili_unknown(result_file):
        """
        CLI function evaluating the classifier on WiLI.

        Parameters
        ----------
        result_file : str
            Path to a file where the results will be stored
        """
        classifier.eval_wili(
            result_file, classifier.get_mapping_languages(), eval_unk=True
        )

    return entry_point
