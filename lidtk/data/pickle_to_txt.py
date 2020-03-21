#!/usr/bin/env python

"""Convert a language pickle file to a txt."""

# Core Library modules
import codecs
import glob
import pickle


def main():
    """Convert all pickle language files to txt files."""
    lang_files = glob.glob("lang/*.pickle")
    for lang_file in lang_files:
        lang = lang_file.split("/")[-1].split(".")[0]
        target_path = "lang_txt/{}.txt".format(lang)
        convert(lang_file, target_path)


def convert(source_path, target_path):
    """
    Convert a single pickle language files to a txt file.

    Parameters
    ----------
    source_path : str
    target_path : str
    """
    # Load data (deserialize)
    with open(source_path, "rb") as handle:
        unserialized_data = pickle.load(handle)["paragraphs"]

    with codecs.open(target_path, "w", "utf8") as f:
        f.write("\n".join(unserialized_data))


if __name__ == "__main__":
    main()
