#!/usr/bin/env python

# core modules
from abc import ABC, abstractmethod
import datetime
import io
import json
import logging
import os
import time
import yaml

# 3rd party modules
import progressbar
import numpy as np

# internal modules
from lidtk.data import wili
import lidtk.utils


class LIDClassifier(ABC):
    """
    A classifier for identifying languages.

    Parameters
    ----------
    cfg : dict
    mapping : dict
        Maps the code of the service to the code used by WiLI.
        If the service knows a language which is not in WiLI, it is mapped to
        UNK.
    """

    def __init__(self, cfg_path):
        cfg_path = os.path.abspath(cfg_path)
        with open(cfg_path, 'r') as stream:
            cfg = yaml.load(stream)
        self.cfg = cfg

    def map2wili(self, services_code):
        return self.cfg['mapping'].get(services_code, 'UNK')

    @abstractmethod
    def predict(self, text):
        """
        Predict the language of the given text.

        Parameters
        ----------
        text : str

        Returns
        -------
        language : str
            ISO 369-3 code
        """

    def predict_bulk(self, texts):
        """
        Predict the language of a list of texts.

        Parameters
        ----------
        texts : list of str

        Returns
        -------
        languages : list of str
            List of ISO 369-3 codes or UNK
        """
        return [self.predict(text) for text in texts]

    def get_languages(self):
        """
        Find which languages are predicted for the WiLI dataset.

        Returns
        -------
        languages : list of str
            Each str is a ISO 369-3 code
        """
        languages = set()
        # Read data
        data = wili.load_data()
        logging.info("Finished loading data")
        for set_name in ['test', 'train']:
            x_set_name = 'x_{}'.format(set_name)
            bar = progressbar.ProgressBar(redirect_stdout=True,
                                          max_value=len(data[x_set_name]))
            for i, el in enumerate(data[x_set_name]):
                try:
                    predicted = self.predict(el)
                except Exception as e:
                    predicted = 'UNK'
                    logging.error({'message': 'Exception in get_languages',
                                   'error': e})
                languages.add(predicted)
                bar.update(i + 1)
            bar.finish()
        return sorted(list(languages))

    def get_mapping_languages(self):
        return sorted([lang for _, lang in self.cfg['mapping'].items()])

    def eval_wili(self, result_file, languages=None):
        """
        Evaluate the classifier on WiLI.

        Parameters
        ----------
        result_file : str
            Path to a file where the results will be stored
        languages : list, optional (default: All languages)
            Filter languages by this list
        """
        # Read data
        data = wili.load_data()
        logging.info("Finished loading data")
        times = []
        bar = progressbar.ProgressBar(redirect_stdout=True,
                                      max_value=len(data['x_test']))
        result_filepath = os.path.abspath(result_file)
        logging.info("Write results to {}".format(result_filepath))
        results = {'meta': {}}
        now = datetime.datetime.now()
        results['meta']['experiment_start'] = ('{:%Y-%m-%d %H:%M:%S}'
                                               .format(now))
        cl_results = {}
        with open(result_filepath, 'w') as filepointer:
            for i, (el, label_t) in enumerate(zip(data['x_test'],
                                                  data['y_test'])):
                if languages is not None:
                    if label_t not in languages:
                        continue
                    else:
                        print(label_t)
                try:
                    t0 = time.time()
                    predicted = self.predict(el)
                    t1 = time.time()
                    times.append(t1 - t0)
                    bar.update(i + 1)
                    if label_t != predicted:
                        if label_t not in cl_results:
                            cl_results[label_t] = {}
                        if predicted not in cl_results[label_t]:
                            cl_results[label_t][predicted] = []
                        identifier = 'test_{}'.format(i)
                        cl_results[label_t][predicted].append([identifier,
                                                               el])
                except Exception as e:  # catch them all
                    logging.error({'message': 'Exception in eval_wili',
                                   'error': e})
                    predicted = 'UNK'
                filepointer.write(predicted + '\n')
        bar.finish()
        results['cl_results'] = cl_results
        times = np.array(times)
        print("Average time per 10**6 elements: {:.2f}s"
              .format(times.mean() * 10**6))
        results['time_per_10*6'] = times.mean() * 10**6
        logfile = result_filepath + '.json'
        results['meta']['hardware'] = lidtk.utils.get_hardware_info()
        results['meta']['software'] = lidtk.utils.get_software_info()
        with io.open(logfile, 'w', encoding='utf8') as f:
            f.write(json.dumps(results,
                               indent=4,
                               sort_keys=True,
                               ensure_ascii=False))
