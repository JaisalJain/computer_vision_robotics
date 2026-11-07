"""Unit tests for the notebook-standard checker in scripts/verify_repo.py."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_repo", ROOT / "scripts" / "verify_repo.py")
verify_repo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify_repo)

PNG = {"image/png": "iVBORw0KGgo=", "text/plain": "<Figure>"}
TABLE = {"text/html": "<table><tr><td>1</td></tr></table>", "text/plain": "df"}


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(src, n, outputs=()):
    outs = [{"output_type": "display_data", "data": d, "metadata": {}} for d in outputs]
    return {
        "cell_type": "code",
        "metadata": {},
        "source": src,
        "execution_count": n,
        "outputs": outs,
    }


def good_cells():
    return [
        md("# Title\nOverview"),
        md("Setup explanation"),
        code("# --- setup ---\nimport numpy", 1),
        md("## Results\nFigure"),
        code("# plot\nplt.plot()", 2, [PNG]),
        md("Table"),
        code("# table\ndf", 3, [TABLE]),
        md("### Takeaways\n- ok"),
        md("## Exercises\n1."),
        code("# TODO: exercise 1", 4),
        md("2."),
        code("# TODO: exercise 2", 5),
    ]


def write_nb(tmp_path, cells):
    path = tmp_path / "nb.ipynb"
    path.write_text(
        json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5})
    )
    return path


@pytest.fixture
def nb_path(tmp_path):
    return lambda cells: write_nb(tmp_path, cells)


def test_good_notebook_passes(nb_path):
    assert verify_repo.check_notebook(nb_path(good_cells())) == []


def test_missing_comment_and_markdown(nb_path):
    cells = good_cells()
    cells[2]["source"] = "import numpy"
    cells.insert(3, code("# extra", 9))
    errs = " ".join(verify_repo.check_notebook(nb_path(cells)))
    assert "'#' comment" in errs
    assert "preceded by an explanatory markdown" in errs
    assert "1..N" in errs


def test_errors_unexecuted_and_missing_sections(nb_path):
    cells = good_cells()
    cells[2]["execution_count"] = None
    cells[2]["outputs"] = [
        {"output_type": "error", "ename": "ValueError", "evalue": "", "traceback": []}
    ]
    cells[3]["source"] = "## Outcome"
    errs = " ".join(verify_repo.check_notebook(nb_path(cells)))
    assert "not executed" in errs and "error output" in errs and "## Results" in errs


def test_results_requires_table_and_figure(nb_path):
    cells = good_cells()
    cells[6]["outputs"] = []
    errs = " ".join(verify_repo.check_notebook(nb_path(cells)))
    assert "summary table" in errs


def test_patterns_do_not_match_this_repo_text():
    assert verify_repo.ATTRIBUTION_RE.search("an ordinary sentence about cameras") is None
    assert verify_repo.HOME_PATH_RE.search("~/.cache/huggingface") is None
