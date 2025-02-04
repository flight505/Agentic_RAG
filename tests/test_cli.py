import os
import pytest
from pathlib import Path
from typer.testing import CliRunner
from agentic_rag.cli import app, RAGSingleton, _loaded_pdfs
from unittest.mock import patch
from rich.console import Console

# Use a non-interactive console for testing
console = Console(force_terminal=False)

# Create CLI runner
runner = CliRunner()

@pytest.fixture(autouse=True)
def reset_state():
    """Reset global state before each test."""
    RAGSingleton._instance = None
    _loaded_pdfs.clear()
    yield

@pytest.fixture(autouse=True)
def mock_rich_live():
    """Mock Rich Live display to prevent conflicts."""
    with patch("rich.live.Live.__enter__", return_value=None), \
         patch("rich.live.Live.__exit__", return_value=None):
        yield

@pytest.fixture
def env_setup():
    """Setup environment variables for testing."""
    os.environ["DEEPSEEK_API_KEY"] = "dummy-key"
    os.environ["DEEPSEEK_API_BASE"] = "https://api.deepseek.com/v1"
    yield
    # Cleanup
    os.environ.pop("DEEPSEEK_API_KEY", None)
    os.environ.pop("DEEPSEEK_API_BASE", None)

def test_main_menu_help(env_setup):
    """Test that the main menu help command works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "main-menu" in result.stdout

def test_main_menu_command(env_setup):
    """Test that the main-menu command is available."""
    result = runner.invoke(app, ["main-menu", "--help"])
    assert result.exit_code == 0
    assert "Run the main menu interface" in result.stdout

def test_query_command(env_setup):
    """Test the direct query command."""
    result = runner.invoke(app, ["query", "--help"])
    assert result.exit_code == 0
    assert "Query the RAG system" in result.stdout

def test_load_pdf_command(env_setup):
    """Test the PDF loading command."""
    result = runner.invoke(app, ["load-pdf", "--help"])
    assert result.exit_code == 0
    assert "Load a PDF file" in result.stdout 