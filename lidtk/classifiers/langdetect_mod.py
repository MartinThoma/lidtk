#!/usr/bin/env python

"""
Run classification with langdetect.

Notes
-----
* Python wrapper: https://pypi.python.org/pypi/langdetect
* Based on: https://github.com/shuyo/language-detection
"""

# core modules
import pkg_resources

# 3rd party modules
from langdetect import detect_langs, detect
from langdetect import DetectorFactory
import click

# internal modules
import lidtk.classifiers


DetectorFactory.seed = 0  # Make sure we get consistent results


class LangdetectClassifier(lidtk.classifiers.LIDClassifier):
    """LID with the Langdetect classifier."""

    def predict(self, text):
        """
        Predicting the language of a text.

        Parameters
        ----------
        text : str
        """
        return self.map2wili(detect(text))

    def predict_proba(self, text):
        """
        Predicting probability of languages of a text.

        Parameters
        ----------
        text : str
        """
        probabilities = detect_langs(text)
        converted = []
        for el in probabilities:
            converted.append({"lang": self.map2wili(el.lang), "prob": el.prob})
        return converted


path = "classifiers/config/langdetect.yaml"
filepath = pkg_resources.resource_filename("lidtk", path)
classifier = LangdetectClassifier(filepath)


###############################################################################
# CLI                                                                         #
###############################################################################
entry_point = lidtk.classifiers.classifier_cli_factor(classifier)


@entry_point.command(name="predict_proba")
@click.option("--text")
def predict_proba_cli(text):
    """
    CLI function for predicting the probability of a language of a text.

    Parameters
    ----------
    text : str
    """
    print(classifier.predict_proba(text))
