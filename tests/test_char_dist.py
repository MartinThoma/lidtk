# Core Library modules
import unittest

# Third party modules
from click.testing import CliRunner

# First party modules
import lidtk.classifiers.char_distribution.char_dist_metric_train_test as todo


class CharDistTest(unittest.TestCase):
    def test_main(self):
        runner = CliRunner()
        result = runner.invoke(todo.main, ["--coverage", 0.1, "--set_name", "val"])
        print(result)
