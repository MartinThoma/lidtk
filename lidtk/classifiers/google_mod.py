#!/usr/bin/env python

"""
Run classification with Google Cloud.

Notes
-----
* Install Google Cloud SDK first: https://cloud.google.com/sdk/downloads?hl=de
* See https://cloud.google.com/translate/docs/detecting-language
"""

# core modules
import os
import pkg_resources

# 3rd party modules
import click

# internal modules
import lidtk.classifiers
import lidtk.data.wili as wili


class GCClassifier(lidtk.classifiers.LIDClassifier):
    """LID with the Google Cloud classifier."""

    def predict(self, text):
        """
        Predicting the language of a text.

        Parameters
        ----------
        text : str
        """
        from google.cloud import translate
        translate_client = translate.Client()
        result = translate_client.detect_language(text)
        # print('Confidence: {}'.format(result['confidence']))
        # print('Language: {}'.format(result['language']))
        return result['language']  # self.map2wili(details[0].language_code)

path = 'classifiers/config/google-cloud.yaml'
filepath = pkg_resources.resource_filename('lidtk', path)
classifier = GCClassifier(filepath)


###############################################################################
# CLI                                                                         #
###############################################################################
@click.group(name=classifier.cfg['name'])
def entry_point():
    """Use the CLD-2 language classifier."""


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
    """Get all predicted languages of for the WiLI dataset."""
    print(classifier.get_languages())


@entry_point.command(name='print_languages')
@click.option('--label_filepath',
              required=True,
              type=click.Path(exists=True),
              help='CSV file with delimiter ;')
def print_languages(label_filepath):
    """
    Print supported languages of classifier.

    Parameters
    ----------
    label_filepath : str
    """
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


@entry_point.command(name='list-languages')
def list_languages():
    """List all available languages."""
    from google.cloud import translate
    translate_client = translate.Client()

    results = translate_client.get_languages()

    for language in results:
        print(u'{name} ({language})'.format(**language))