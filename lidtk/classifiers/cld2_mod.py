#!/usr/bin/env python

"""
Run classification with CLD2.

Notes
-----
* https://github.com/CLD2Owners/cld2
* https://pypi.python.org/pypi/cld2-cffi
"""

# Third party modules
import cld2
import pkg_resources

# First party modules
import lidtk.classifiers


class CLD2Classifier(lidtk.classifiers.LIDClassifier):
    """LID with the CLD-2 classifier."""

    def predict(self, text):
        """
        Predicting the language of a text.

        Parameters
        ----------
        text : str
        """
        is_reliable, text_bytes_found, details = cld2.detect(text, bestEffort=True)
        return self.map2wili(details[0].language_code)


path = "classifiers/config/cld2.yaml"
filepath = pkg_resources.resource_filename("lidtk", path)
classifier = CLD2Classifier(filepath)


###############################################################################
# CLI                                                                         #
###############################################################################
entry_point = lidtk.classifiers.classifier_cli_factor(classifier)
