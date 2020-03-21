# core modules
import unittest

# 3rd party modules
from click.testing import CliRunner

# internal modules
import lidtk.classifiers.cld2_mod


class ClassifiersTest(unittest.TestCase):
    def test_cld2_predict(self):
        runner = CliRunner()
        result = runner.invoke(
            lidtk.classifiers.cld2_mod.entry_point,
            ["predict", "--text", "I don't go to school."],
        )
        self.assertEqual(result.output, "eng\n")

    def test_cld2_get_languages(self):
        runner = CliRunner()
        runner.invoke(lidtk.classifiers.cld2_mod.entry_point, ["get_languages"])

    def test_cld2_eval_willi(self):
        runner = CliRunner()
        runner.invoke(lidtk.classifiers.cld2_mod.entry_point, ["wili"])
