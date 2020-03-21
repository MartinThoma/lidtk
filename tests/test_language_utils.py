# core modules
import pkg_resources
import unittest

# 3rd party modules
from click.testing import CliRunner

# internal modules
from lidtk.data import language_utils


class LanguageUtilsTest(unittest.TestCase):
    def test_main(self):
        runner = CliRunner()
        result = runner.invoke(language_utils.main, ["--theta", 0.5])
        print(result)
