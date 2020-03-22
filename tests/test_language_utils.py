# Third party modules
from click.testing import CliRunner

# First party modules
from lidtk.data import language_utils


def test_main():
    runner = CliRunner()
    result = runner.invoke(language_utils.main, ["--theta", 0.5])
    print(result)
