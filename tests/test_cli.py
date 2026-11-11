"""Tests for src/cli.py: run main() with the toy CSV and check the exit code.

The charts go to a temporary folder, so output/ is not touched.
"""

import io
import os
import tempfile
from contextlib import redirect_stdout, redirect_stderr

from src.cli import main

TESTS_DIR = os.path.dirname(__file__)
TOY_CSV = os.path.join(TESTS_DIR, "toy_data.csv")


def run(args):
    """Run main(args) without printing; return (exit code, printed text)."""

    out = io.StringIO()  # CATCH WHAT IS PRINTED

    with redirect_stdout(out), redirect_stderr(out):
        code = main(args + ["--data", TOY_CSV, "--output", tempfile.mkdtemp()])

    return code, out.getvalue()


def test_predict():
    """A valid season: exit code 0 and the real MVP printed."""

    code, text = run(["predict", "2001"])

    assert code == 0

    assert "Real MVP: Alpha Center" in text


def test_unknown_season():
    """A season that is not in the data: exit code 1 and an error message."""

    code, text = run(["predict", "1950"])

    assert code == 1

    assert "Error" in text
