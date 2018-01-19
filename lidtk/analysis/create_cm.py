#!/usr/bin/env python

"""Analyze a given model."""

# core modules
import pickle
# Make it work for Python 2+3 and with Unicode
import io
try:
    to_unicode = unicode
except NameError:
    to_unicode = str
import json

# 3rd party modules
from sklearn.metrics import confusion_matrix
import numpy as np


def data_preprocessing(vectorizer_filename='tfidf-100.pickle'):
    """
    Calculate preprocessed data.

    Returns
    -------
    data : dict
    """
    from lidtk.data import wili
    data = wili.load_data({'target_type': 'one_hot'})

    englsh_classnames = [el['English'] for el in wili.labels]
    # Write JSON file
    with io.open('wili-labels.json', 'w', encoding='utf8') as outfile:
        str_ = json.dumps(englsh_classnames, sort_keys=True,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(to_unicode(str_))

    # Load vectorizer and preprocess data
    with open(vectorizer_filename, 'rb') as handle:
        vectorizer = pickle.load(handle)
    data['x_test'] = vectorizer.transform(data['x_test']).toarray()
    return {'data': data, 'labels': englsh_classnames}


def generate_confusion_matrix_report(model, data, labels):
    """Generate the confusion matrix and visualize it."""
    y_true = data['y_test']
    y_true = np.argmax(y_true, axis=1)
    y_pred = np.argmax(model.predict(data['x_test']), axis=1)
    cm = confusion_matrix(y_true, y_pred)

    import sklearn.metrics
    print(sklearn.metrics.classification_report(y_true, y_pred, target_names=labels))

    # Write JSON file
    with io.open('cm-test.json', 'w', encoding='utf8') as outfile:
        str_ = json.dumps(cm.tolist(), sort_keys=True,
                          separators=(',', ': '), ensure_ascii=False)
        outfile.write(to_unicode(str_))


def get_parser():
    """Get parser object for model analysis."""
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description=__doc__,
                            formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--classifier",
                        dest="classifier_filename",
                        help="Keras model filename",
                        required=True,
                        metavar="FILE")
    parser.add_argument("--vectorizer",
                        dest="vectorizer_filename",
                        help="tfidf vectorizer",
                        required=True,
                        metavar="FILE")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    from keras.models import load_model
    model = load_model(args.classifier_filename)
    ret = data_preprocessing(args.vectorizer_filename)
    data = ret['data']
    labels = ret['labels']
    generate_confusion_matrix_report(model, data, labels=labels)
