"""Execute the tutorial notebooks headlessly (see README.md for the environment variables)."""
import json
import os
from pathlib import Path

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor

FOLDER = Path(os.environ.get("NB_TEST_FOLDER", Path(__file__).parent))
SUBFOLDERS = json.loads(os.getenv("SUBFOLDERS", '["0","1","2","3","4"]'))
TIMEOUT = int(os.environ.get("NB_TEST_TIMEOUT", "600"))
QUICK = os.environ.get("NB_TEST_QUICK", "False").lower() in ("1", "true", "yes")

NOTEBOOKS = sorted(p for p in FOLDER.rglob("*.ipynb")
                   if p.parent.name[:1] in SUBFOLDERS and ".ipynb_checkpoints" not in p.parts)


@pytest.mark.parametrize("notebook_path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_runs(notebook_path):
    """The notebook runs without error, with plotting disabled."""
    if "TODO" in str(notebook_path):
        pytest.skip("notebook under development")
    notebook = nbformat.read(notebook_path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            if QUICK and ("train" in cell.source or "anim." in cell.source):
                pytest.skip("training or animation cell (NB_TEST_QUICK)")
            cell.source = "import matplotlib; matplotlib.use('Agg')\n" + cell.source
    ExecutePreprocessor(timeout=TIMEOUT, kernel_name="python3").preprocess(
        notebook, {"metadata": {"path": str(notebook_path.parent)}})
