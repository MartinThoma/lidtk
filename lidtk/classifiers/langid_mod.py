#!/usr/bin/env python

"""
Run classification with Langid.py.

Notes
-----
* https://github.com/saffsd/langid.py
"""

# core modules
import os
import pkg_resources

# 3rd party modules
import click
import langid

# internal modules
import lidtk.classifiers
import lidtk.data.wili as wili


class LangidClassifier(lidtk.classifiers.LIDClassifier):
    def predict(self, text):
        """
        Predicting the language of a text.

        Parameters
        ----------
        text : str
        """
        language_code, score = langid.classify(text)
        return self.map2wili(language_code)

path = 'classifiers/config/langid.yaml'
filepath = pkg_resources.resource_filename('lidtk', path)
classifier = LangidClassifier(filepath)


###############################################################################
# CLI                                                                         #
###############################################################################
@click.group(name='langid')
def entry_point():
    """Use the langid language classifier."""


@entry_point.command(name='predict')
@click.option('--text')
def predict_cli(text):
    """
    Command line interface function for predicting the language of a text.

    Parameters
    ----------
    text : str
    """
    print(classifier.predict(text))


@entry_point.command(name='get_languages')
def get_languages():
    print(classifier.get_languages())


@entry_point.command(name='print_languages')
@click.option('--label_filepath',
              required=True,
              type=click.Path(exists=True),
              help='CSV file with delimiter ;')
def print_languages(label_filepath):
    label_filepath = os.path.abspath(label_filepath)
    wili_labels = wili.get_language_data(label_filepath)
    iso2name = dict([(el['ISO 369-3'], el['English'])
                     for el in wili_labels])
    print(', '.join(sorted([iso2name[iso]
                            for iso in classifier.get_mapping_languages()
                            if iso != 'UNK'])))


@entry_point.command(name='wili')
@click.option('--result_file',
              default='langid_results.txt', show_default=True,
              help='Where to store the predictions')
def eval_wili(result_file):
    """
    CLI function evaluating the classifier on WiLI.

    Parameters
    ----------
    result_file : str
        Path to a file where the results will be stored
    """
    classifier.eval_wili(result_file)


@entry_point.command(name='wili_k')
@click.option('--result_file',
              default='langdetect_results_known.txt', show_default=True,
              help='Where to store the predictions')
def eval_wili_known(result_file):
    """
    CLI function evaluating the classifier on WiLI.

    Parameters
    ----------
    result_file : str
        Path to a file where the results will be stored
    """
    classifier.eval_wili(result_file, classifier.get_mapping_languages())
