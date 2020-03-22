# Third party modules
from click.testing import CliRunner

# First party modules
import lidtk.classifiers.cld2_mod


def test_cld2_predict():
    runner = CliRunner()
    result = runner.invoke(
        lidtk.classifiers.cld2_mod.entry_point,
        ["predict", "--text", "I don't go to school."],
    )
    assert result.output == "eng\n"


def test_cld2_get_languages():
    runner = CliRunner()
    runner.invoke(lidtk.classifiers.cld2_mod.entry_point, ["get_languages"])


def test_cld2_eval_willi():
    runner = CliRunner()
    runner.invoke(lidtk.classifiers.cld2_mod.entry_point, ["wili"])
