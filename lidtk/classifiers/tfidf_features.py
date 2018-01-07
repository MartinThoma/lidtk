#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Create tfidf.pickle."""

# core modules
import pickle

# 3rd party modules
from sklearn.feature_extraction.text import TfidfVectorizer

# local modules
from lidtk.data import wili


def get_features(config, data):
    """
    Get tf-idf features based on characters.

    Parameters
    ----------
    config : dict
    data : dict
    """
    if config is None:
        config = {}
    if 'feature-extraction' not in config:
        config['feature-extraction'] = {}
    if 'min_df' not in config['feature-extraction']:
        config['feature-extraction']['min_df'] = 50
    vectorizer = TfidfVectorizer(analyzer='char',
                                 min_df=config['feature-extraction']['min_df'],
                                 lowercase=config['feature-extraction']['lowercase'],
                                 norm=config['feature-extraction']['norm'])
    xs = {}
    vectorizer.fit(data['x_train'])
    # Serialize trained vectorizer
    with open(config['feature-extraction']['name'], 'wb') as fin:
        pickle.dump(vectorizer, fin)
    for set_name in ['x_train', 'x_test', 'x_val']:
        xs[set_name] = vectorizer.transform(data[set_name]).toarray()
    return {'vectorizer': vectorizer, 'xs': xs}


def analyze_vocabulary(vectorizer):
    """Show which vocabulary is used by the vectorizer."""
    voc = sorted([unicode(key)
                  for key, _ in ret['vectorizer'].vocabulary_.items()])
    print(u','.join(voc))
    print("Vocabulary: {}".format(len(voc)))


def get_parser():
    """Get parser object."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    # parser.add_argument("--config",
    #                     dest="filename",
    #                     help="Read configuration file",
    #                     metavar="FILE",
    #                     required=True)
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    data = wili.load_data()
    config = {'feature-extraction': {'min_df': 50,
                                     'name': 'tfidf-50.pickle',
                                     'lowercase': True,
                                     'norm': 'l2'}}
    ret = get_features(config, data)
    analyze_vocabulary(data)
    print("First 20 samplex of x_train:")
    print(ret['xs']['x_train'][:20])
