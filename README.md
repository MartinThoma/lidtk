# lidtk

lidtk - the language identification toolkit - was written in order to
investigate the current state of language performance.


## Installation

The recommended way to install clana is:

```
$ pip install lidtk --user
```

If you want the latest version:

```
$ git clone https://github.com/MartinThoma/lidtk.git; cd lidtk
$ pip instell -e . --user
```

I recommend getting the [WiLI-2018 dataset](https://zenodo.org/record/841984).


## Usage


```
$ lidtk --help

Usage: lidtk [OPTIONS] COMMAND [ARGS]...

Options:
  --version  Show the version and exit.
  --help     Show this message and exit.

Commands:
  analyze-data           Utility function for the languages...
  analyze-unicode-block  Analyze how important a Unicode block is for...
  char-distrib           Use the character distribution language...
  cld2                   Use the CLD-2 language classifier.
  create-dataset         Create sharable dataset from downloaded...
  download               Download 1000 documents of each language.
  google-cloud           Use the CLD-2 language classifier.
  langdetect             Use the langdetect language classifier.
  langid                 Use the langid language classifier.
  map                    Map predictions to something known by WiLI
  nn                     Use a neural network classifier.
  textcat                Use the CLD-2 language classifier.
  tfidf_nn               Use the TfidfNNClassifier classifier.

```

For example:

```
$ lidtk cld2 predict --text 'This is a test.'
eng
```

The usual order is:

1. `lidtk download`: Please use [WiLI-2018](https://zenodo.org/record/841984) instead of downloading the dataset on your own.
2. `lidtk create-dataset`: This step can be skipped if you use WiLI-2018
3. `lidtk analyze-data`
4. `lidtk tfidf_nn vectorizer train`
5. `lidtk tfidf_nn mlp train`
6. `lidtk tfidf_nn wili`

Or to use one directly:

```
$ lidtk cld2 predict --text 'This text is written in some language.'

eng
```


## Development

Check tests with `tox`.
