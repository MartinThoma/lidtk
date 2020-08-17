r"""
Train and test character distance models.

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

# Core Library modules
import logging
import os
import pickle
import random
import sys
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# Third party modules
import click
import numpy as np
import scipy.stats
from scipy.spatial import distance

# First party modules
import lidtk.classifiers
from lidtk.classifiers.char_features import FeatureExtractor  # noqa
from lidtk.data import wili

random.seed(0)
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def ido(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate the distance of x and y with the ido metric.

    Parameters
    ----------
    x : np.ndarray
        A probability distribution
    y : np.ndarray
        A probability distribution

    Returns
    -------
    distance : float
        How much the two probability distributions overlap.

    Examples
    --------
    >>> ido ([0.5, 0.5], [0.1, 0.9])
    0.4
    """
    return 1 - np.sum(np.minimum(x, y))


language_models = None  # type: Optional[Dict[Any, Any]]
language_models_chars = None  # type: Optional[List[str]]
comp_metric = ido


###############################################################################
# CLI                                                                         #
###############################################################################
@click.group(name="char-distrib")
def entry_point() -> None:
    """Use the character distribution language classifier."""


@entry_point.command(name="predict")
@click.option("--text")
def predict_cli(text: str) -> None:
    """
    Command line interface function for predicting the language of a text.

    Parameters
    ----------
    text : str
    """
    print(predict(text))


@entry_point.command(name="wili")
@click.option(
    "--result_file",
    default="char_dist_metric_results.txt",
    show_default=True,
    help="Where to store the predictions",
)
def eval_wili(result_file: str) -> None:
    """
    CLI function evaluating the classifier on WiLI.

    Parameters
    ----------
    result_file : str
        Path to a file where the results will be stored
    """
    if language_models is None:
        init_language_models(comp_metric, unicode_cutoff=10 ** 6)
    lidtk.classifiers.eval_wili(result_file, predict)  # type: ignore


###############################################################################
# Logic                                                                       #
###############################################################################
@entry_point.command(name="train")
@click.option("--coverage", default=0.8, show_default=True)
@click.option("--metric", default=0, show_default=True)
@click.option(
    "--set_name",
    default="train",
    show_default=True,
    type=click.Choice(["train", "test", "val"]),
)
@click.option("--unicode_cutoff", default=10 ** 6, show_default=True)
def main(coverage: float, metric: int, unicode_cutoff: int, set_name: str = "train"):
    """
    Train and test character distance models.

    Parameters
    ----------
    coverage : float
    metric : int
        Specify a function
    unicode_cutoff : int
    set_name : str
        Define on which set to evaluate
    """
    metrics = [
        ido,  # 0
        distance.braycurtis,  # 1
        distance.canberra,  # 2
        distance.chebyshev,  # 3 - l_infty
        distance.cityblock,  # 4
        distance.correlation,  # 5
        distance.cosine,  # 6
        distance.euclidean,  # 7
        distance.sqeuclidean,  # 8
        scipy.stats.entropy,  # 9
    ]
    metric_function = metrics[metric]

    # Read data
    data = wili.load_data()
    logger.info("Finished loading data")

    # Train
    trained = train(data, unicode_cutoff, coverage, metric_function)

    # Create model for each language and store it
    out_tmp = get_counts_by_lang(
        trained["common_chars"], trained["char_counter_by_lang"]
    )
    language_models, chars = out_tmp
    model_filename = "~/.lidtk/models/char_dist_{metric}_{cutoff}.pickle".format(
        metric=metric_function.__name__, cutoff=unicode_cutoff
    )
    model_filename = os.path.expanduser(model_filename)
    with open(model_filename, "wb") as handle:
        model_info = {"language_models": language_models, "chars": chars}
        pickle.dump(model_info, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Evaluate
    cm_filepath = "char_{metric}_{coverage}_{cutoff}_{set_name}.cm.csv".format(
        metric=metric_function.__name__,
        coverage=coverage,
        cutoff=unicode_cutoff,
        set_name=set_name,
    )
    cfg = lidtk.utils.load_cfg()
    cm_filepath = os.path.join(cfg["artifacts_path"], cm_filepath)


def train(
    data: Dict[Any, Any], unicode_cutoff: int, coverage: float, metric: Callable
) -> Dict[Any, Any]:
    """Train a model which is purely based on character distributions."""
    char_counter_by_lang = defaultdict(Counter)  # type: Dict[str, Counter]
    for x, y in zip(data["x_train"], data["y_train"]):
        char_counter_by_lang[y] += Counter(preprocess(x, unicode_cutoff))

    common_chars_by_lang = {}
    for key, character_counter in char_counter_by_lang.items():
        common_chars_by_lang[key] = get_common_characters(
            character_counter, coverage=coverage
        )
    common_chars = set()  # type: Set[str]
    for _lang, char_list in common_chars_by_lang.items():
        common_chars = common_chars.union(char_list)
    logger.info(
        "|{metric} & {coverage}% & {cutoff} &  {characters} chars".format(
            metric=metric.__name__,
            coverage=(coverage * 100),
            cutoff=unicode_cutoff,
            characters=len(common_chars),
        )
    )
    results = {}  # type: Dict[str, Any]
    results["common_chars"] = list(common_chars)
    results["char_counter_by_lang"] = char_counter_by_lang
    return results


def get_counts_by_lang(
    common_chars: List[str], char_counter_by_lang: Dict[str, Dict[str, int]]
) -> Tuple[Dict[Any, Any], List[str]]:
    """
    Get a language model for each language.

    Parameters
    ----------
    common_chars : List[str]
    char_counter_by_lang : dict
        Maps language to list of int. This list has the same length as
        common_chars and represents the number of times the character was
        present in the corpus for the given language.

    Returns
    -------
    language_models, chars : (Dict, List[str])
        maps (code => np.ndarray)
    """
    language_model = {}
    for lang, char_counter in char_counter_by_lang.items():
        total_count = sum(count for count in char_counter.values())
        other_count = sum(
            count for char, count in char_counter.items() if char not in common_chars
        )
        language_model[lang] = {"other": (float(other_count) / float(total_count))}
        for char in common_chars:
            language_model[lang][char] = float(char_counter[char]) / total_count

    logger.info("Language model ready")
    for lang in ["deu", "eng", "fra"]:
        print(
            "{lang}: {model}".format(lang=lang, model=model_repr(language_model, lang))
        )

    # should be the same for all languages
    chars = sorted(language_model["eng"].keys())
    print(chars)

    language_models = {}
    for code, model in language_model.items():
        language_models[code] = np.array([model[char] for char in chars])
    return language_models, chars


def preprocess(x: str, unicode_cutoff: int, cut_off_char: str = "澳") -> str:
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
    y = ""
    for el in x:
        if unicode_cutoff is not None and ord(el) > unicode_cutoff:
            y += cut_off_char
        else:
            y += el
    return y


def get_common_characters(
    character_counter: Counter, coverage: float = 1.0
) -> List[str]:
    """
    Get the most common characters of a language.

    Parameters
    ----------
    character_counter : Counter
    coverage : float, optional (default: 1.0)
        Take the most common characters that make up `coverage` of the dataset

    Returns
    -------
    common_characters : list of most common characters that cover `coverage`
        of all character occurences, ordered by count (most common first).
    """
    assert coverage > 0.0, f"coverage={coverage}, but > 0 expected"
    counts = sorted(character_counter.items(), key=lambda n: (n[1], n[0]), reverse=True)
    chars = []
    count_sum = sum(el[1] for el in counts)
    count_sum_min = coverage * count_sum
    count = 0
    for char, char_count in counts:
        chars.append(char)
        count += char_count
        if count >= count_sum_min:
            break
    return chars


def model_repr(models: Dict[str, Dict[str, float]], key) -> str:
    """Get a model representation."""
    m = [(char, proba) for char, proba in models[key].items() if proba > 0.0001]
    m = sorted(m, key=lambda n: n[1], reverse=True)
    s = ""
    for char, prob in m:
        s += f"{char}={prob * 100:4.2f}%  "
    return s


def get_distribution(x: str, chars: List[str]) -> np.ndarray:
    """
    Get distribution of characters in sample.

    Parameters
    ----------
    x : str
    chars : List[str]

    Returns
    -------
    distribution : np.ndarray
        Has the same length as chars
    """
    dist = np.zeros(len(chars), dtype=np.float32)
    other_index = chars.index("other")
    for el in x:
        if el in chars:
            dist[chars.index(el)] += 1
        else:
            dist[other_index] += 1
    # Normalize
    for i in range(len(dist)):
        dist[i] /= len(x)
    return dist


def predict(text: str):
    """
    Predict the language of a text.

    Parameters
    ----------
    text : str

    Returns
    -------
    language_code : str
    """
    if language_models is None:
        init_language_models(comp_metric, unicode_cutoff=10 ** 6)
    assert language_models is not None, "assert for mypy"
    assert language_models_chars is not None, "assert for mypy"
    x_distribution = get_distribution(text, language_models_chars)
    return predict_param(language_models, comp_metric, x_distribution, best_only=True)


def init_language_models(metric: Callable, unicode_cutoff: int) -> None:
    """Initialize the language_models global variable."""
    model_filename = "~/.lidtk/models/char_dist_{metric}_{cutoff}.pickle".format(
        metric=metric.__name__, cutoff=unicode_cutoff
    )
    model_filename = os.path.expanduser(model_filename)
    # Load data (deserialize)
    with open(model_filename, "rb") as handle:
        data = pickle.load(handle)
    globals()["language_models"] = data["language_models"]
    globals()["language_models_chars"] = data["chars"]


def predict_param(
    language_models: Dict[str, Any],
    comp_metric: Callable,
    x_distribution: np.ndarray,
    best_only: bool = True,
) -> Union[List[Tuple[float, str]], str]:
    """
    Predict the language of x.

    Parameters
    ----------
    language_models : Dict[str, Any]
        language => model
    comp_metric : function with two parameters (model_dist, x_dist)
    x_distribution : np.ndarray of dtype float
    best_only : bool, optional (default: True)
        If this is True, then only the most likely language code is be returned
        Otherwise, the list of tuples is returned.

    Returns
    -------
    distances : List[Tuple[float, str]]
        If best_only, then [(distance, 'language'), ...]
        Otherwise, 'language'
    """
    distances = []  # type: List[Tuple[float, str]]
    for lang, model_distribution in language_models.items():
        distance = comp_metric(model_distribution, x_distribution)
        distances.append((distance, lang))
    if best_only:
        return min(distances)[1]
    else:
        return distances
