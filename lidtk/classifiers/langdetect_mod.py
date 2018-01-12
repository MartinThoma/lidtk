#!/usr/bin/env python

"""
Run classification with langdetect.

Notes
-----
* Python wrapper: https://pypi.python.org/pypi/langdetect
* Based on: https://github.com/shuyo/language-detection
"""

# core modules
import os
import pkg_resources

# 3rd party modules
from langdetect import detect_langs, detect
from langdetect import DetectorFactory
import click

# internal modules
import lidtk.classifiers
import lidtk.data.wili as wili


DetectorFactory.seed = 0  # Make sure we get consistent results


class LangdetectClassifier(lidtk.classifiers.LIDClassifier):

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
            converted.append({'lang': self.map2wili(el.lang),
                              'prob': el.prob})
        return converted


path = 'classifiers/config/langdetect.yaml'
filepath = pkg_resources.resource_filename('lidtk', path)
classifier = LangdetectClassifier(filepath)


###############################################################################
# CLI                                                                         #
###############################################################################
@click.group(name=classifier.cfg['name'])
def entry_point():
    """Use the langdetect language classifier."""


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


@entry_point.command(name='predict_proba')
@click.option('--text')
def predict_proba_cli(text):
    """
    CLI function for predicting the probability of a language of a text.

    Parameters
    ----------
    text : str
    """
    print(classifier.predict_proba(text))


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
              default='{}_results.txt'.format(classifier.cfg['name']),
              show_default=True,
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
              default='{}_results_known.txt'.format(classifier.cfg['name']),
              show_default=True,
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


@entry_point.command(name='wili_unk')
@click.option('--result_file',
              default='{}_results_unknown.txt'.format(classifier.cfg['name']),
              show_default=True,
              help='Where to store the predictions')
def eval_wili_unknown(result_file):
    """
    CLI function evaluating the classifier on WiLI.

    Parameters
    ----------
    result_file : str
        Path to a file where the results will be stored
    """
    classifier.eval_wili(result_file,
                         classifier.get_mapping_languages(),
                         eval_unk=True)
