#!/usr/bin/env python

# core modules
import io
import json
import logging
import os
import time

# 3rd party modules
import progressbar
import numpy as np

# internal modules
from lidtk.data import wili


def eval_wili(result_file, predict):
    """
    Evaluate the classifier on WiLI.

    Parameters
    ----------
    result_file : str
        Path to a file where the results will be stored
    predict : function
        (str -> str),
        Takes a text and returns the WiLI code of a language or UNK
    """
    # Read data
    data = wili.load_data()
    logging.info("Finished loading data")
    times = []
    bar = progressbar.ProgressBar(redirect_stdout=True,
                                  max_value=len(data['x_test']))
    result_filepath = os.path.abspath(result_file)
    logging.info("Write results to {}".format(result_filepath))
    results = {}
    cl_results = {}
    with open(result_filepath, 'w') as filepointer:
        for i, (el, label_t) in enumerate(zip(data['x_test'], data['y_test'])):
            try:
                t0 = time.time()
                predicted = predict(el)
                t1 = time.time()
                times.append(t1 - t0)
                bar.update(i + 1)
                if label_t == predicted:
                    continue
                if label_t not in cl_results:
                    cl_results[label_t] = {}
                if predicted not in cl_results[label_t]:
                    cl_results[label_t][predicted] = []
                cl_results[label_t][predicted].append(('test_'.format(i), el))
            except:  # catch them all
                predicted = 'UNK'
            filepointer.write(predicted + '\n')
    bar.finish()
    results['cl_results'] = cl_results
    times = np.array(times)
    print("Average time per 10**6 elements: {:.2f}s"
          .format(times.mean() * 10**6))
    results['time_per_10*6'] = times.mean() * 10**6
    logfile = result_filepath + '.json'
    with io.open(logfile, 'w', encoding='utf8') as f:
        f.write(json.dumps(results,
                           indent=4,
                           sort_keys=True,
                           ensure_ascii=False))
