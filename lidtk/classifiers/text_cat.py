#!/usr/bin/env python

"""
Run classification with text_cat.

Notes
-----
* https://github.com/CLD2Owners/cld2
* https://pypi.python.org/pypi/cld2-cffi
"""

# Third party modules
import nltk.classify.textcat
import pkg_resources

# First party modules
import lidtk.classifiers


class TextCatClassifier(lidtk.classifiers.LIDClassifier):
    """LID Classifier which uses TextCat."""

    def predict(self, text):
        """
        Predicting the language of a text.

        Parameters
        ----------
        text : str
        """
        o = nltk.classify.textcat.TextCat()
        language_code = o.guess_language(text)
        return language_code


path = "classifiers/config/textcat.yaml"
filepath = pkg_resources.resource_filename("lidtk", path)
classifier = TextCatClassifier(filepath)


###############################################################################
# CLI                                                                         #
###############################################################################
entry_point = lidtk.classifiers.classifier_cli_factor(classifier)
