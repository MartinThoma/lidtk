# lidtk

lidtk - the language identification toolkit - was written in order to
investigate the current state of language performance.


## Installation

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
  mlp                    Use the character distribution language...
  nn                     Use a neural network classifier.
  textcat                Use the CLD-2 language classifier.
```

For example:

```
$ lidtk cld2 predict --text 'This is a test.'
eng
```

The usual order is:

* `lidtk download`
* `lidtk create-dataset`
* `lidtk analyze-data`

## Scripts

Models:

* `create_cm.py`: Create a confusion matrix
* `char_dist_metric_train_test.py -m 0 -c 0.8`


## Development

Check tests with `tox`.
