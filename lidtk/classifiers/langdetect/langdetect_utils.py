#!/usr/bin/env python

"""
Run classification with langdetect.

Notes
-----
* Python wrapper: https://pypi.python.org/pypi/langdetect
* Based on: https://github.com/shuyo/language-detection
"""

# 3rd party modules
from langdetect import detect_langs, detect
from langdetect import DetectorFactory
import click

# internal modules
import lidtk.classifiers


DetectorFactory.seed = 0  # Make sure we get consistent results


mapping = {'af': 'afr',
           'ar': 'ara',
           'bg': 'bul',
           'bn': 'ben',
           'ca': 'cat',
           'cs': 'ces',
           'cy': 'cym',
           'da': 'dan',
           'de': 'deu',
           'el': 'ell',
           'en': 'eng',
           'es': 'spa',
           'et': 'est',
           'fa': 'fas',
           'fi': 'fin',
           'fr': 'fra',
           'gu': 'guj',
           'he': 'heb',
           'hi': 'hin',
           'hr': 'hrv',
           'hu': 'hun',
           'id': 'ind',
           'it': 'ita',
           'ja': 'jpn',
           'kn': 'kan',
           'ko': 'kor',
           'lt': 'lit',
           'lv': 'lav',
           'mk': 'mkd',
           'ml': 'mal',
           'mr': 'mar',
           'ne': 'nep',
           'nl': 'nld',
           'no': 'nob',
           'pa': 'pan',
           'pl': 'pol',
           'pt': 'por',
           'ro': 'ron',
           'ru': 'rus',
           'sk': 'slk',
           'sl': 'slv',
           'so': 'som',
           'sq': 'sqi',
           'sv': 'swe',
           'sw': 'swa',
           'ta': 'tam',
           'te': 'tel',
           'th': 'tha',
           'tl': 'tgl',
           'tr': 'tur',
           'uk': 'ukr',
           'ur': 'urd',
           'vi': 'vie',
           'zh-cn': 'zho',  # ?
           'zh-tw': 'lzh'}  # ?


###############################################################################
# CLI                                                                         #
###############################################################################
@click.group(name='langdetect')
def entry_point():
    """Use the langdetect language classifier."""
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


@entry_point.command(name='predict_proba')
@click.option('--text')
def predict_proba_cli(text):
    """
    CLI function for predicting the probability of a language of a text.

    Parameters
    ----------
    text : str
    """
    print(predict_proba(text))


@entry_point.command(name='wili')
@click.option('--result_file',
              default='langdetect_results.txt', show_default=True,
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
def langdetect2wili(lang_code):
    """
    Convert a lagdetect language code to a WiLI language code.

    Parameters
    ----------
    lang_code : str

    Returns
    -------
    wili_lang_code : str
    """
    return mapping.get(lang_code, 'UNK')


def predict(text):
    """
    Predicting the language of a text.

    Parameters
    ----------
    text : str
    """
    return langdetect2wili(detect(text))


def predict_proba(text):
    """
    Predicting probability of languages of a text.

    Parameters
    ----------
    text : str
    """
    probabilities = detect_langs(text)
    converted = []
    for el in probabilities:
        converted.append({'lang': langdetect2wili(el.lang),
                          'prob': el.prob})
    return converted
