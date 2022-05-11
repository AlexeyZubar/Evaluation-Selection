from click.testing import CliRunner
import pytest
from Forest.train import train


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_max_features(runner: CliRunner) -> None:
    """It fails when max-features is greater than 1."""
    result = runner.invoke(
        train,
        [
            "--max-features",
            2,
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--max-features'" in result.output

def test_error_for_invalid_max_depth(runner: CliRunner) -> None:
    """It fails when max_depth is lower than 0."""
    result = runner.invoke(
        train,
        [
            "--max-depth",
            -1,
        ],
    )
    assert result.exit_code == 2
    assert "Invalid value for '--max_depth'" in result.output

"""
@click.command()
@click.argument('f', type=click.File())
def cat(f):
   click.echo(f.read())

def test_cat():
   runner = CliRunner()
   with runner.isolated_filesystem():
      with open('hello.txt', 'w') as f:
          f.write('Hello World!')

      result = runner.invoke(cat, ['hello.txt'])
      assert result.exit_code == 0
      assert result.output == 'Hello World!\n'
"""
