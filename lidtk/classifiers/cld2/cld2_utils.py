#!/usr/bin/env python

"""
Run classification with CLD2.

Notes
-----
* https://github.com/CLD2Owners/cld2
* https://pypi.python.org/pypi/cld2-cffi
"""

# core modules
import logging

# 3rd party modules
import click
import cld2

# internal modules
from lidtk.classifiers.cld2 import cld2wili
import lidtk.classifiers


###############################################################################
# CLI                                                                         #
###############################################################################
@click.group(name='cld2')
def entry_point():
    """Use the CLD-2 language classifier."""
    pass


@entry_point.command(name='predict')
@click.option('--text')
def predict_cli(text):
    """
    Command line interface function for predicting the language of a text.

    Parameters
    ----------
    text : str
    """
    print(predict(text))


@entry_point.command(name='wili')
@click.option('--result_file',
              default='cld2_results.txt', show_default=True,
              help='Where to store the predictions')
def eval_wili(result_file):
    """
    CLI function evaluating the classifier on WiLI.

    Parameters
    ----------
    result_file : str
        Path to a file where the results will be stored
    """
    lidtk.classifiers.eval_wili(result_file, predict)


###############################################################################
# Logic                                                                       #
###############################################################################
def code2wili(lang_code):
    """
    Convert a language code to a WiLI language code.

    Parameters
    ----------
    lang_code : str

    Returns
    -------
    wili_lang_code : str
    """
    if lang_code not in cld2wili.servicecode2label:
        logging.warning('Could not find "{}"'.format(lang_code))
    return cld2wili.servicecode2label.get(lang_code, 'UNK')


def predict(text):
    """
    Predicting the language of a text.

    Parameters
    ----------
    text : str
    """
    is_reliable, text_bytes_found, details = cld2.detect(text,
                                                         bestEffort=True)
    return code2wili(details[0].language_code)
