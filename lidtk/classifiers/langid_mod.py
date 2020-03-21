#!/usr/bin/env python

"""
Run classification with Langid.py.

Notes
-----
* https://github.com/saffsd/langid.py
"""

# Third party modules
import langid
import pkg_resources

# First party modules
import lidtk.classifiers


class LangidClassifier(lidtk.classifiers.LIDClassifier):
    """LID with the Langid classifier."""

    def predict(self, text):
        """
        Predicting the language of a text.

        Parameters
        ----------
        text : str
        """
        language_code, score = langid.classify(text)
        return self.map2wili(language_code)


path = "classifiers/config/langid.yaml"
filepath = pkg_resources.resource_filename("lidtk", path)
classifier = LangidClassifier(filepath)


###############################################################################
# CLI                                                                         #
###############################################################################
entry_point = lidtk.classifiers.classifier_cli_factor(classifier)
