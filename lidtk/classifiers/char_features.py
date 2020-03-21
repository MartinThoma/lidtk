#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Extract character features."""

# Core Library modules
import logging
import os
import pickle
from collections import Counter, defaultdict

# Third party modules
import numpy as np
import progressbar

# First party modules
import lidtk.utils


class FeatureExtractor(object):
    """Character feature extractor."""

    def __init__(self, xs, ys, coverage=0.8):
        """
        Initialize the feature extractor.

        Parameters
        ----------
        coverage : float in (0, 1]
        """
        self.chars = []
        self.char2index = {}
        self.coverage = coverage
        self.fit(xs, ys)

    def fit(self, xs, ys):
        """
        Fit the feature extractor to the data.

        Parameters
        ----------
        xs : list of str
        ys : list of str

        Returns
        -------
        feature extractor
        """
        logging.info("count characters")
        char_counter_by_lang = defaultdict(Counter)
        for x, y in zip(xs, ys):
            char_counter_by_lang[y] += Counter(x)

        logging.info(
            "get common characters to get coverage of {}".format(self.coverage)
        )
        common_chars_by_lang = {}
        for key, character_counter in char_counter_by_lang.items():
            common_chars_by_lang[key] = self._get_common_characters(
                character_counter, coverage=self.coverage
            )

        logging.info("unify set of common characters")
        common_chars = set()
        for lang, char_list in common_chars_by_lang.items():
            common_chars = common_chars.union(char_list)
        common_chars.add("other")
        self.chars = list(common_chars)
        for index, char in enumerate(self.chars):
            self.char2index[char] = index
        return self

    def transform(self, xs):
        """Get distribution of characters in sample."""
        dist = None
        if isinstance(xs, (list, np.ndarray, np.generic)):
            dist = self.transform_multiple(xs)
        else:
            dist = self.transform_single(xs)
        return dist

    def transform_single(self, x):
        """
        Get distribution of characters in sample.

        Parameters
        ----------
        x : str

        Returns
        -------
        distribution : numpy array of dtype float
            Frequency of characters
        """
        dist = np.zeros(len(self.chars), dtype=np.float32)
        for el in x:
            if el in self.chars:
                dist[self.char2index[el]] += 1.0
            else:
                dist[self.char2index["other"]] += 1.0
        # Normalize
        dist /= dist.sum()
        return dist

    def transform_multiple(self, xs, bar=False):
        """
        TODO.

        Parameters
        ----------
        xs : TODO
        bar : boolean, optional (default: False)

        Returns
        -------
        dists : ndarray
            TODO
        """
        target_shape = (len(xs), len(self.chars))
        logging.info("transform_multiple to target_shape={}".format(target_shape))
        dists = np.zeros(target_shape)
        if bar:
            bar = progressbar.ProgressBar(redirect_stdout=True, max_value=len(xs))
        for i in range(len(xs)):
            dists[i] = self.transform_single(xs[i])
            if bar:
                bar.update(i + 1)
        if bar:
            bar.finish()
        return dists

    def get_xs_set(self, data, set_name):
        """
        Get featureset.

        Parameters
        ----------
        data : dict
        set_name : str

        Returns
        -------
        xs : ndarray
        """
        cfg = lidtk.utils.load_cfg()
        train_xs_pickle = cfg["train_xs_pickle_path"].format(self.coverage, set_name)
        if not os.path.exists(train_xs_pickle + ".npy"):
            logging.info(
                "Start creating {} x {}".format(len(data[set_name]), len(self.chars))
            )
            xs = self.transform_multiple(data[set_name], bar=True)
            # Serialize the transformed data
            np.save(train_xs_pickle, xs)
        else:
            # Load the transformed data
            xs = np.load(train_xs_pickle + ".npy")
        return xs

    def _get_common_characters(self, character_counter, coverage=1.0):
        """
        Get the most common characters of a language.

        Parameters
        ----------
        character_counter : collections.Counter
        coverage : float, optional (default: 1.0)
            Take the most common characters that make up `coverage` of the
            dataset

        Returns
        -------
        common_characters : list of most common characters that cover
            `coverage` of all character occurences, ordered by count (most
            common first).
        """
        assert coverage > 0.0
        counts = sorted(
            character_counter.items(), key=lambda n: (n[1], n[0]), reverse=True
        )
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


def get_features(config, data):
    """
    Get tf-idf features based on characters.

    Parameters
    ----------
    config : dict
    data : dict

    Returns
    -------
    features : dict
        'vectorizer' and 'xs'
    """
    cfg = lidtk.utils.load_cfg()
    feature_extractor_path = cfg["feature_extractor_path"].format(config["coverage"])
    if not os.path.isfile(feature_extractor_path):
        logging.info("Create vectorizer")
        vectorizer = FeatureExtractor(
            data["x_train"], data["y_train"], coverage=config["coverage"]
        )
        # Serialize trained vectorizer
        with open(feature_extractor_path, "wb") as f:
            pickle.dump(vectorizer, f)
    else:
        # Load the trained vectorizer
        with open(feature_extractor_path, "rb") as handle:
            vectorizer = pickle.load(handle)
    xs = {}
    for set_name in ["x_val", "x_train", "x_test"]:
        xs[set_name] = vectorizer.get_xs_set(data, set_name)
    return {"vectorizer": vectorizer, "xs": xs}
