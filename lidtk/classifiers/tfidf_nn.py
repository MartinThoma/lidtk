#!/usr/bin/env python

"""
Run classification with tfidf-features and Neural Network classifier.

"""

# core modules
import pickle
import pkg_resources

# 3rd party modules
import numpy as np
import click

# local modules
import lidtk.classifiers
from lidtk.data import wili


class TfidfNNClassifier(lidtk.classifiers.LIDClassifier):
    """LID with the TfidfNNClassifier."""

    def __init__(self, filepath):
        super().__init__(filepath)
        self.labels = wili.labels

    def load(self, vectorizer_filename, classifier_filename):
        from keras.models import load_model
        with open(vectorizer_filename, 'rb') as handle:
            self.vectorizer = pickle.load(handle)
        self.model = load_model(classifier_filename)

    def predict(self, text):
        """
        Predicting the language of a text.

        Parameters
        ----------
        text : str
        """
        features = self.vectorizer.transform([text]).toarray()
        prediction = self.model.predict(features)
        most_likely = np.argmax(prediction, axis=1)
        most_likely = [self.map2wili(index) for index in most_likely]
        return most_likely[0]


path = 'classifiers/config/tfidf_nn.yaml'
filepath = pkg_resources.resource_filename('lidtk', path)
classifier = TfidfNNClassifier(filepath)
classifier.load(classifier.cfg['vectorizer_src_path'],
                classifier.cfg['model_src_path'])


###############################################################################
# CLI                                                                         #
###############################################################################
@click.group(name=classifier.cfg['name'])
def entry_point():
    """Use the TfidfNNClassifier classifier."""


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
