# Third party modules
from click.testing import CliRunner

# First party modules
import lidtk.classifiers.char_distribution.char_dist_metric_train_test as todo


def test_main():
    runner = CliRunner()
    result = runner.invoke(todo.main, ["--coverage", 0.1, "--set_name", "val"])
    print(result)
