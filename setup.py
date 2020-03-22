# Third party modules
from setuptools import setup

setup(
    install_requires=[
        "cld2-cffi>=0.1.4",
        "click>=6.7",
        "detectlanguage>=1.2.1",
        "fuzzywuzzy>=0.16.0",
        "python-Levenshtein",
        "h5py>=2.7.1",
        "Keras>=2.0.6",
        "langdetect>=1.0.7",
        "langid>=1.1.6",
        "matplotlib>=2.1.2",
        "nltk>=3.2.5",
        "numpy>=1.14.0",
        "progressbar2>=3.34.3",
        "PyYAML>=3.12",
        "scikit-learn>=0.19.1",
        "scipy>=1.0.0",
        "seaborn>=0.8.1",
        "tensorflow>=1.2.0",
        "wikipedia>=1.4.0",
    ],
    tests_require=["pytest>=3.3.2", "pytest-cov>=2.5.1", "pytest-pep8>=1.0.6",],
)
