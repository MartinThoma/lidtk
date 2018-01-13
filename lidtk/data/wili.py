#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Load WiLI-Dataset."""

# core modules
import codecs
import csv
import logging

# 3rd party modules
from sklearn.model_selection import train_test_split
import numpy as np

# internal modules
from lidtk.utils import make_path_absolute

isodict = None


def indices_to_one_hot(data, nb_classes):
    """
    Convert an iterable of indices to one-hot encoded labels.

    Examples
    --------
    >>> indices_to_one_hot([0, 1, 2, 0, 0, 1], 3)
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.],
           [1., 0., 0.],
           [1., 0., 0.],
           [0., 1., 0.]])
    """
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]


def get_language_data(csv_filepath='~/.lidtk/data/labels.csv'):
    """
    Get language data.

    Parameters
    ----------
    csv_filepath : str

    Returns
    -------
    wili_labels : list of dicts

    Example
    -------
    >>> wiki = get_language_data()
    >>> sorted(wiki[0].keys())[:6]
    ['English', 'German', 'ISO 369-3', 'Label', 'Language family', 'Remarks']
    >>> wiki[0]['ISO 369-3']
    'ace'
    """
    csv_filepath = make_path_absolute(csv_filepath)
    with open(csv_filepath, 'r') as fp:
        wiki = [{k: v for k, v in row.items()}
                for row in csv.DictReader(fp,
                                          skipinitialspace=True,
                                          delimiter=';', quotechar='"')]
    return wiki


def load_data(config=None):
    """
    Load the WID dataset.

    Parameters
    ----------
    config : dict

    Returns
    -------
    dict with 'x_train', 'y_train', 'x_val', 'y_val', 'x_test', 'y_test',
         'labels'

    Examples
    --------
    >>> data = load_data({'target_type': 'one_hot'})
    >>> len(data['x_train'])
    94000
    >>> len(data['x_val'])
    23500
    >>> len(data['x_test'])
    117500
    >>> data['y_train'].shape
    (94000, 235)
    >>> len(data['labels'])
    235
    >>> data['labels'][0]['ISO 369-3']
    'ace'
    """
    if config is None:
        config = {}
    x_train_path = make_path_absolute('~/.lidtk/data/x_train.txt')
    logging.info("wili.load_data uses x_train_path='{}'"
                 .format(x_train_path))
    with codecs.open(x_train_path, 'r', 'utf-8') as f:
        x_train = f.read().strip().split("\n")
    y_train_path = make_path_absolute('~/.lidtk/data/y_train.txt')
    logging.info("wili.load_data uses y_train_path='{}'"
                 .format(y_train_path))
    with codecs.open(y_train_path, 'r', 'utf-8') as f:
        y_train = f.read().strip().split("\n")
    x_train, x_val, y_train, y_val = train_test_split(x_train,
                                                      y_train,
                                                      stratify=y_train,
                                                      test_size=0.2,
                                                      random_state=0)
    x_test_path = make_path_absolute('~/.lidtk/data/x_test.txt')
    logging.info("wili.load_data uses x_test_path='{}'"
                 .format(x_test_path))
    with codecs.open(x_test_path, 'r', 'utf-8') as f:
        x_test = f.read().strip().split("\n")
    y_test_path = make_path_absolute('~/.lidtk/data/y_test.txt')
    logging.info("wili.load_data uses y_test_path='{}'"
                 .format(y_test_path))
    with codecs.open(y_test_path, 'r', 'utf-8') as f:
        y_test = f.read().strip().split("\n")
    ys = {'y_train': y_train, 'y_val': y_val, 'y_test': y_test}
    label_list = [el['Label'] for el in globals()["labels"]]
    for set_name in ['y_train', 'y_val', 'y_test']:
        if 'target_type' in config and config['target_type'] == 'one_hot':
            ys[set_name] = np.array([label_list.index(y)
                                     for y in ys[set_name]])
            ys[set_name] = indices_to_one_hot(ys[set_name],
                                              globals()['n_classes'])
    data = {'x_train': x_train, 'y_train': ys['y_train'],
            'x_val': x_val, 'y_val': ys['y_val'],
            'x_test': x_test, 'y_test': ys['y_test'],
            'labels': globals()["labels"]}
    return data


labels = get_language_data(make_path_absolute('~/.lidtk/data/labels.csv'))
labels_s = [el['Label'] for el in labels]
n_classes = len(labels)
