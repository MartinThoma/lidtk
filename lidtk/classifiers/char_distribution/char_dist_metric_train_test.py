#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
Train and test a character frequenceys/earthmovers distance model.

The idea is to take the most common characters which make up a proportion of
\alpha (e.g. 0.9) of the complete data. For language i, let this set of
characters be C_i. Do this for all languages, build the super set C =
\bigcup_{i} C_i. Add an "other" character.

Count the frequencies of C for each language. This way, you get a language
character distribution.

When a new text comes, count the frequencies of the characters C in the text.
Compare this distribution to all language distributions. Predict the language
which has the most similar distribution. Similarity of distributions can
be calculated by the earthmovers distance.
"""

# core moduels
from collections import defaultdict, Counter
import datetime
import io
import json
import logging
import os
import pickle
import random
import sys
import time
random.seed(0)

# 3rd party modules
from scipy.spatial import distance
from sklearn.metrics import classification_report
import numpy as np
import progressbar
import scipy.stats
import click

# local modules
import lidtk.classifiers
from lidtk.analysis import manual_error_analysis
# from lidtk.classifiers import char_features
from lidtk.data import wili
from lidtk.utils import make_path_absolute

# Needed to import for pickle
from lidtk.classifiers.char_features import FeatureExtractor  # noqa


logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


language_models = None
language_models_chars = None


###############################################################################
# CLI                                                                         #
###############################################################################
@click.group(name='char-distrib')
def entry_point():
    """Use the character distribution language classifier."""
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
              default='char_dist_metric_results.txt', show_default=True,
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
@entry_point.command(name='train')
@click.option('--coverage', default=0.8, show_default=True)
@click.option('--metric', default=0, show_default=True)
@click.option('--set_name',
              default='train',
              show_default=True,
              type=click.Choice(['train', 'test', 'val']))
@click.option('--unicode_cutoff', default=10**6, show_default=True)
def main(coverage, metric, unicode_cutoff, set_name='train'):
    """
    Main function.

    Parameters
    ----------
    coverage : float
    metric : function
    unicode_cutoff : int
    set_name : str
        Define on which set to evaluate
    """
    metrics = [ido,  # 0
               distance.braycurtis,  # 1
               distance.canberra,  # 2
               distance.chebyshev,  # 3
               distance.cityblock,  # 4
               distance.correlation,  # 5
               distance.cosine,  # 6
               distance.euclidean,  # 7
               distance.sqeuclidean,  # 8
               scipy.stats.entropy,  # 9
               ]
    metric = metrics[metric]

    config = {'coverage': coverage}

    # Read data
    data = wili.load_data()
    logging.info("Finished loading data")

    # Train
    trained = train(data, unicode_cutoff, coverage, metric)

    # Create model for each language and store it
    out_tmp = get_counts_by_lang(trained['common_chars'],
                                 trained['char_counter_by_lang'])
    language_models, chars = out_tmp
    model_filename = ('~/.lidtk/models/char_dist_{metric}_{cutoff}.pickle'
                      .format(metric=metric.__name__,
                              cutoff=unicode_cutoff))
    model_filename = os.path.expanduser(model_filename)
    with open(model_filename, 'wb') as handle:
        model_info = {'language_models': language_models,
                      'chars': chars}
        pickle.dump(model_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Evaluate
    cm_filepath = ("char_{metric}_{coverage}_{cutoff}_{set_name}.cm.csv"
                   .format(metric=metric.__name__,
                           coverage=coverage,
                           cutoff=unicode_cutoff,
                           set_name=set_name))
    cm_filepath = make_path_absolute(os.path.join('~/.lidtk/artifacts',
                                                  cm_filepath))
    logging.info('Start evaluation on "{}"'.format(set_name))
    results = evaluate_model(data,
                             language_models,
                             metric,
                             cm_filepath,
                             config,
                             chars,
                             set_name=set_name)
    logfile = ('errors-char-dist-metric-{:%Y-%m-%d-%H-%M}.log'
               .format(datetime.datetime.now()))
    results['report'] = classification_report(data['y_{}'.format(set_name)],
                                              results['y_pred'])
    with io.open(logfile, 'w', encoding='utf8') as f:
        f.write(json.dumps(results,
                           indent=4,
                           sort_keys=True,
                           ensure_ascii=False))
    manual_error_analysis(results['errors'], ['eng', 'deu', 'fra'])
    logging.info('Logged to "{}"'.format(logfile))


def train(data, unicode_cutoff, coverage, metric):
    """
    Train a model which is purely based on character distributions.

    Parameters
    ----------
    data : dict
    unicode_cutoff : int
    coverage : float
    metric : function

    Returns
    -------
    results : dict
    """
    char_counter_by_lang = defaultdict(Counter)
    for x, y in zip(data['x_train'], data['y_train']):
        char_counter_by_lang[y] += Counter(preprocess(x, unicode_cutoff))

    common_chars_by_lang = {}
    for key, character_counter in char_counter_by_lang.items():
        common_chars_by_lang[key] = get_common_characters(character_counter,
                                                          coverage=coverage)
    common_chars = set()
    for lang, char_list in common_chars_by_lang.items():
        common_chars = common_chars.union(char_list)
    common_chars = list(common_chars)
    logging.info("|{metric} & {coverage}% & {cutoff} &  {characters} chars"
                 .format(metric=metric.__name__,
                         coverage=(coverage * 100),
                         cutoff=unicode_cutoff,
                         characters=len(common_chars)))
    results = {}
    results['common_chars'] = common_chars
    results['char_counter_by_lang'] = char_counter_by_lang
    return results


def get_counts_by_lang(common_chars, char_counter_by_lang):
    """
    Get a language model for each language.

    Parameters
    ----------
    common_chars : list of str
    char_counter_by_lang : dict
        Maps language to list of int. This list has the same length as
        common_chars and represents the number of times the character was
        present in the corpus for the given language.

    Returns
    -------
    language_models, chars : (dict, list of str)
        maps (code => ndarray)
    """
    language_model = {}
    for lang, char_counter in char_counter_by_lang.items():
        total_count = sum(count for count in char_counter.values())
        other_count = sum(count for char, count in char_counter.items()
                          if char not in common_chars)
        language_model[lang] = {'other': (float(other_count) /
                                          float(total_count))}
        for char in common_chars:
            language_model[lang][char] = (float(char_counter[char]) /
                                          total_count)

    logging.info("Language model ready")
    for lang in ['deu', 'eng', 'fra']:
        print(u"{lang}: {model}".format(lang=lang,
                                        model=model_repr(language_model,
                                                         lang)))

    # should be the same for all languages
    chars = sorted(language_model['eng'].keys())
    print(chars)

    language_models = {}
    for code, model in language_model.items():
        language_models[code] = np.array([model[char] for char in chars])
    return language_models, chars


def preprocess(x, unicode_cutoff, cut_off_char=u'澳'):
    """
    Preprocess the string x.

    Parameters
    ----------
    x : str
    unicode_cutoff : int

    Returns
    -------
    preprocessed_str : str
        Some characters are replaced by a 'cut off' parameter
    """
    y = u""
    for el in x:
        if unicode_cutoff is not None and ord(el) > unicode_cutoff:
            y += cut_off_char
        else:
            y += el
    return y


def get_common_characters(character_counter, coverage=1.0):
    """
    Get the most common characters of a language.

    Parameters
    ----------
    character_counter : collections.Counter
    coverage : float, optional (default: 1.0)
        Take the most common characters that make up `coverage` of the dataset

    Returns
    -------
    common_characters : list of most common characters that cover `coverage`
        of all character occurences, ordered by count (most common first).
    """
    assert coverage > 0.0
    counts = sorted(character_counter.items(),
                    key=lambda n: (n[1], n[0]),
                    reverse=True)
    chars = []
    count_sum = sum([el[1] for el in counts])
    count_sum_min = coverage * count_sum
    count = 0
    for char, char_count in counts:
        chars.append(char)
        count += char_count
        if count >= count_sum_min:
            break
    return chars


def model_repr(models, key):
    """Get a model representation."""
    m = [(char, proba)
         for char, proba in models[key].items() if proba > 0.0001]
    m = sorted(m, key=lambda n: n[1], reverse=True)
    s = ""
    for char, prob in m:
        s += u"{}={:4.2f}%  ".format(char, prob * 100)
    return s


def get_distribution(x, chars):
    """
    Get distribution of characters in sample.

    Parameters
    ----------
    x : str
    chars : iterable
        e.g. a list of str

    Returns
    -------
    distribution : ndarray
        Has the same length as chars
    """
    dist = np.zeros(len(chars), dtype=np.float32)
    other_index = chars.index('other')
    for el in x:
        if el in chars:
            dist[chars.index(el)] += 1
        else:
            dist[other_index] += 1
    # Normalize
    for i in range(len(dist)):
        dist[i] /= len(x)
    return dist


def predict(text):
    """
    Predict the language of a text.

    Parameters
    ----------
    text : str

    Returns
    -------
    language_code : str
    """
    comp_metric = distance.cityblock
    if language_models is None:
        init_language_models(comp_metric, unicode_cutoff=10**6)
    pass  # TODO
    x_distribution = get_distribution(text, language_models_chars)
    return predict_param(language_models,
                         comp_metric,
                         x_distribution,
                         best_only=True)


def init_language_models(metric, unicode_cutoff):
    """Initialize the language_models global variable."""
    model_filename = ('~/.lidtk/models/char_dist_{metric}_{cutoff}.pickle'
                      .format(metric=metric.__name__,
                              cutoff=unicode_cutoff))
    model_filename = os.path.expanduser(model_filename)
    # Load data (deserialize)
    with open(model_filename, 'rb') as handle:
        data = pickle.load(handle)
    globals()['language_models'] = data['language_models']
    globals()['language_models_chars'] = data['chars']


def predict_param(language_models,
                  comp_metric,
                  x_distribution,
                  best_only=True):
    """
    Predict the language of x.

    Parameters
    ----------
    language_models : dict
        language => model
    comp_metric : function with two parameters (model_dist, x_dist)
    x_distribution : numpy array of dtype float
    best_only : bool, optional (default: True)
        If this is True, then only the most likely language code is be returned
        Otherwise, the list of tuples is returned.

    Returns
    -------
    distances : list of tuples or str
        If best_only, then [(distance, 'language'), ...]
        Otherwise, 'language'
    """
    distances = []  # tuples (distance, language)
    for lang, model_distribution in language_models.items():
        distance = comp_metric(model_distribution, x_distribution)
        distances.append((distance, lang))
    if best_only:
        return min(distances)[1]
    else:
        return distances


def ido(x, y):
    """The ido metric."""
    return 1 - np.sum(np.minimum(x, y))


def evaluate_model(data,
                   language_models,
                   comp_metric,
                   cm_filepath,
                   config,
                   chars,
                   complete_dump=False,
                   set_name='test'):
    """
    Evaluate the model and write the predictions to cm_filepath.

    Parameters
    ----------
    data : dict
        With keys 'x_test', 'y_test'
    language_models : dict
        lang => model
    comp_metric : function with two parameters (model_dist, x_dist)
    cm_filepath : str
        Classification matrix file path
    config : dict
    complete_dump : boolean, optional (default: False)
        If this is true, then the probability distribution of the prediction
        will be stored to cm_filepath

    Returns
    -------
    errors : dict of dict of lists
        Has the form 'errors[true][predicted] = [sample 1, sample 2, ...]'
    """
    errors = {}
    bar = progressbar.ProgressBar(redirect_stdout=True,
                                  max_value=len(data['y_' + set_name]))
    done = 0
    # cm = [[0 for _ in range(len(wili.labels_s))]
    #       for _ in range(len(wili.labels_s))]
    total_time = 0
    # features = char_features.get_features(config, data)
    # features = features['xs']['x_' + set_name]
    y_preds = []
    i = 0
    logging.info("Write data to '{}'".format(cm_filepath))
    with open(cm_filepath, "w") as f:
        for ident, (x, y) in enumerate(zip(data['x_' + set_name],
                                           data['y_' + set_name])):
            # x_prep = features[i]
            # x = preprocess(x, unicode_cutoff)
            x_prep = get_distribution(x, chars)
            t0 = time.time()
            y_pred = predict_param(language_models,
                                   comp_metric,
                                   x_prep,
                                   not complete_dump)
            t1 = time.time()
            if complete_dump:
                predictions = ";".join([str(el) for el in y_pred])
                f.write("{ident};{predictions}\n"
                        .format(ident=ident, predictions=predictions))
            else:
                true_lang_index = wili.labels_s.index(y_pred)
                f.write("{ident};{prediction}\n"
                        .format(ident=ident, prediction=true_lang_index))
                if y != y_pred:
                    if y not in errors:
                        errors[y] = {}
                    if y_pred not in errors[y]:
                        errors[y][y_pred] = []
                    errors[y][y_pred].append(('{}_{}'.format(set_name, ident),
                                             x))
            total_time += t1 - t0
            # pred_lang_index = wili.labels_s.index(y_pred)
            # cm[true_lang_index][pred_lang_index] += 1
            y_preds.append(y_pred)
            done += 1
            bar.update(done)
            i += 1
    bar.finish()

    logging.info(r"Time per example: {:4.2f}\milli\second"
                 .format((total_time / len(data['y_' + set_name])) * 1000))
    results = {}
    results['errors'] = errors
    results['y_pred'] = y_preds
    return results


def get_parser():
    """Get parser object."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--coverage",
                        dest="coverage",
                        default=0.99,
                        type=float,
                        help="How much of the dataset should be covered?")
    parser.add_argument("--max-unicode",
                        dest="unicode_cutoff",
                        type=int,
                        help="don't print status messages to stdout")
    parser.add_argument("-m", "--metric",
                        dest="metric",
                        type=int,
                        required=True,
                        help="metric to use")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(coverage=args.coverage,
         metric=args.metric,
         unicode_cutoff=args.unicode_cutoff)
