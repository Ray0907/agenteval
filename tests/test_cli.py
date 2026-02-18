import pytest
from typer.testing import CliRunner


@pytest.fixture
def runner():
    from agenteval.cli import app
    return CliRunner()


def test_init(runner, tmp_path):
    from agenteval.cli import app
    result = runner.invoke(app, ["init", str(tmp_path / "proj")])
    assert result.exit_code == 0
    assert (tmp_path / "proj" / "agenteval.yaml").exists()
    assert (tmp_path / "proj" / "scenarios").is_dir()


def test_version(runner):
    from agenteval.cli import app
    result = runner.invoke(app, ["--version"])
    assert "0.1.0" in result.stdout
