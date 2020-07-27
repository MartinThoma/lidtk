"""Language identification toolkit."""

# Third party modules
from setuptools import setup

setup(
    install_requires=[
        "cld2-cffi",
        "click",
        "detectlanguage",
        "fuzzywuzzy",
        "python-Levenshtein",
        "h5py",
        "Keras<2.4.0",
        "langdetect",
        "langid",
        "matplotlib",
        "nltk",
        "numpy",
        "progressbar2",
        "PyYAML",
        "scikit-learn",
        "scipy",
        "seaborn",
        "tensorflow",
        "wikipedia",
    ],
)
