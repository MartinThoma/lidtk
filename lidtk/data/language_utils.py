#!/usr/bin/env python

"""Utility function for the languages themselves."""

# Core Library modules
import csv
import glob
import os
import pickle
from collections import Counter

# Third party modules
import click
import numpy as np

# First party modules
import lidtk.utils
from lidtk.classifiers.char_distribution.char_dist_metric_train_test import (
    get_common_characters,
)
from lidtk.data import char_distribution

iso2wiki = None
wiki2iso = None
wiki2label = None
cfg = lidtk.utils.load_cfg()


@click.command(name="analyze-data", help=__doc__)
@click.option("--lang_dir", default=cfg["lang_dir_path"], show_default=True)
@click.option("--theta", default=0.99, show_default=True)
def main(lang_dir, theta=0.99):
    """
    Analyze the distribution of languages.

    Paramerters
    -----------
    lang_dir : str
    theta : float
        How much coverage of the language should be displayed.
    """
    files = sorted(glob.glob(lang_dir))
    lang_stats = {}
    if len(files) == 0:
        print(
            "No files found at '{}'. You might want to download first.".format(lang_dir)
        )
        return
    print("theta={}".format(theta))
    print("lang:                 characters             paragraphs      ")
    print("-------------------------------------------------------------")
    for filepath in files:
        wiki_code = os.path.splitext(os.path.split(filepath)[1])[0]
        iso = get_iso(wiki_code)
        lang_data = read_language_file(filepath)
        chars = get_characters(lang_data["paragraphs"])
        char_occurences = np.array([el[1] for el in chars.items()])
        char_occurences = char_occurences / float(char_occurences.sum())
        paraphgrah_lengths = np.array([len(el) for el in lang_data["paragraphs"]])
        common_chars = get_common_characters(chars, theta)  # sorted()
        lang_stats[iso] = {
            "theta_100_len": len(chars),
            "theta_99_len": len(get_common_characters(chars, 0.99)),
            "theta_80_len": len(get_common_characters(chars, 0.99)),
            "chars": chars,
            "paraphgrah_len_min": paraphgrah_lengths.min(),
            "paraphgrah_len_max": paraphgrah_lengths.max(),
            "paraphgrah_len_mean": paraphgrah_lengths.mean(),
        }
        print(
            u"{:>9} || {:>5}: {:>5.2f} {:>5.2f} {:>5.2f} {:>5.2f} "
            u"| [{:>5} {:6.1f} {:>6}] '{}' ({} chars)".format(
                iso,
                len(chars),
                char_occurences.min() * 100,  # least common char
                get_percentile_like(char_occurences, 0.99) * 100,
                char_occurences.mean() * 100,  # mean common char
                char_occurences.max() * 100,  # most common char
                paraphgrah_lengths.min(),
                paraphgrah_lengths.mean(),
                paraphgrah_lengths.max(),
                u"".join(common_chars),
                len(common_chars),
            )
        )
    print(
        "Mean paragraph length of mean language paragraph lengths: {}".format(
            np.array([el["paraphgrah_len_mean"] for el in lang_stats.values()]).mean()
        )
    )
    print(
        "Longest paragraph length: {}".format(
            np.array([el["paraphgrah_len_max"] for el in lang_stats.values()]).max()
        )
    )
    i = 0
    for lang, info in lang_stats.items():
        if info["theta_99_len"] >= 150:
            i += 1
            print("{}. {}: {} characters".format(i, lang, info["theta_99_len"]))
    char_distribution.main(lang_stats)


def check_presence(lang_dir="lang"):
    """
    Check how many files of the wiki2iso dict are present.

    Parameters
    ----------
    lang_dir : str
        Directory where language .pickle files can be found

    Returns
    -------
    results : dict
    """
    wiki = wiki2iso.keys()
    for wikicode in wiki:
        path = os.path.join(lang_dir, "{}.pickle".format(wikicode))
        if not os.path.isfile(path):
            print(
                "{} could not be found, but was expected due to wikifile".format(path)
            )
    found_files = glob.glob("{}/*.pickle".format(lang_dir))
    for path in found_files:
        wikicode = path.split("/")[1].split(".")[0]
        if wikicode not in wiki2iso:
            print("Found '{}' unexpectedly".format(wikicode))
    return {"found_files": found_files}


def get_label(wiki_code):
    """
    Get label from wiki code.

    Parameters
    ----------
    wiki_code : str

    Returns
    -------
    str

    Examples
    --------
    >>> get_label('de')
    'deu'
    >>> get_label('en')
    'eng'
    >>> get_label('roa-tara')
    'roa-tara'
    >>> get_label('gom')
    'kok'
    """
    return wiki2label[wiki_code]


def get_iso(wiki_code):
    """
    Get ISO code from wiki code.

    Parameters
    ----------
    wiki_code : str

    Returns
    -------
    str

    Examples
    --------
    >>> get_iso('de')
    'deu'
    >>> get_iso('en')
    'eng'
    >>> get_iso('roa-tara')
    'roa-tara'
    >>> get_iso('gom')
    'kok'
    """
    iso = wiki2iso.get(wiki_code, wiki_code)
    if len(iso) == 0:
        iso = wiki_code
    return iso


def read_language_file(pickle_filepath):
    """
    Read language file.

    Parameters
    ----------
    pickle_filepath : str

    Returns
    -------
    language_data : dict

    Examples
    --------
    >> from lidtk.utils import make_path_absolute
    >> path = make_path_absolute('~/.lidtk/lang/de.pickle')
    >> data = read_language_file(path)
    >> sorted(list(data.keys()))
    ['paragraphs', 'used_pages']
    """
    with open(pickle_filepath, "rb") as handle:
        unserialized_data = pickle.load(handle)
    return unserialized_data


def analyze_language_families(csv_filepath):
    """Analyze which language families are present in the dataset."""
    # Read CSV file
    with open(csv_filepath, "r") as fp:
        reader = csv.reader(fp, delimiter=";", quotechar='"')
        # next(reader, None)  # skip the headers
        wiki = [row for row in reader]

    languages = sorted([el["English"] for el in wiki])
    language_fams = [el["Language family"] for el in wiki]
    language_family_counter = sorted(Counter(language_fams).items(), key=lambda n: n[1])
    for key, value in language_family_counter:
        print("{}: {}".format(key, value))
    print(", ".join(languages))


def get_language_data(csv_filepath=None):
    """
    Get language data.

    Parameters
    ----------
    csv_filepath : str

    Returns
    -------
    wiki : list of dicts
        Each dict represents a langauge

    Example
    -------
    >>> wiki = get_language_data()
    >>> sorted(wiki[0].keys())[:6]
    ['English', 'German', 'ISO 369-3', 'Label', 'Language family', 'Remarks']
    >>> sorted(wiki[0].keys())[6:]
    ['Synonyms', 'Wiki Code', 'Writing system']
    >>> wiki[0]['ISO 369-3']
    'ace'
    """
    if csv_filepath is None:
        cfg = lidtk.utils.load_cfg()
        csv_filepath = cfg["labels_path"]
    with open(csv_filepath, "r") as fp:
        wiki = [
            {k: v for k, v in row.items()}
            for row in csv.DictReader(
                fp, skipinitialspace=True, delimiter=";", quotechar='"'
            )
        ]
    return wiki


def initialize(wiki):
    """
    Initialize global datastructures.

    Parameters
    ----------
    wiki : list of dicts
        Each dict represents a langauge
    """
    globals()["iso2wiki"] = {}
    globals()["wiki2iso"] = {}
    globals()["wiki2label"] = {}
    for el in wiki:
        globals()["iso2wiki"][el["ISO 369-3"]] = el["Wiki Code"]
        globals()["wiki2iso"][el["Wiki Code"]] = el["ISO 369-3"]
        globals()["wiki2label"][el["Wiki Code"]] = el["Label"]


def print_all_languages(wiki):
    """
    Print all languages as a sorted list.

    Parameters
    ----------
    wiki : list of dicts
        Each dict represents a langauge
    """
    languages = sorted([el["English"] for el in wiki])
    languages = [l for l in languages if len(l) > 0]
    print(", ".join(languages))


def print_language_families(wiki, found_files):
    """
    Print how often each language family is represented.

    Parameters
    ----------
    wiki : list of dicts
        Each dict represents a langauge
    found_files : list of str
    """
    languages = sorted([el["English"] for el in wiki])
    languages = [l for l in languages if len(l) > 0]
    language_fams = [el["Language family"] for el in wiki]
    sorted_fams = sorted(
        Counter(language_fams).items(), key=lambda n: n[1], reverse=True
    )
    print(
        "## Total languages: {} ({} files)".format(len(language_fams), len(found_files))
    )
    for key, value in sorted_fams:
        print("{}: {}".format(key, value))
    print(", ".join(languages))


def group_by_language_family(wiki):
    """
    Group languages by language family.

    Parameters
    ----------
    wiki : list of dicts
        Each dict represents a langauge

    Returns
    -------
    dict
    """
    families = []
    fam2index = {}
    for language in wiki:
        eng = language["English"]
        if language["Language family"] not in fam2index:
            fam2index[language["Language family"]] = len(fam2index)
            families.append({language["Language family"]: [eng]})
        else:
            d = families[fam2index[language["Language family"]]]
            key = d.keys()[0]
            d[key].append(eng)
    return {"": families}


def get_characters(lang_data):
    """
    Return a sorted list of characters in the language corpus.

    Parameters
    ----------
    lang_data : list of str
        A list of all paragraphs

    Returns
    -------
    characters : Counter Object
    """
    from collections import Counter

    characters = Counter()  # maps the character to the count
    for paragraph in lang_data:
        characters += Counter(paragraph)
    return characters


def get_percentile_like(xs, min_amount):
    """
    Get the minimum value so that the sum of bigger values makes min_amount.

    Parameters
    ----------
    xs : list of float
    min_amount : float

    Returns
    -------
    float
    """
    xs = sorted(xs, reverse=True)
    sum_ = 0
    i = 0
    while sum_ < min_amount:
        sum_ += xs[i]
        i += 1
    return xs[i - 1]


wiki = get_language_data()
initialize(wiki)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    found = check_presence()
    main()
    # print(globals()['wiki2iso'])
    print_language_families(wiki, found["found_files"])
    print(group_by_language_family(wiki))
