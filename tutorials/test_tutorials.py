"""Execute the tutorial notebooks headlessly, with plotting disabled.

Run from the repository root, e.g.

    SUBFOLDERS='["0","1"]' python -m pytest -s --log-cli-level=INFO tutorials/test_tutorials.py

Environment variables:
  SUBFOLDERS       topic folders to run, e.g. '["0","1"]' (default: all, '["0","1","2","3","4"]')
  NB_TEST_QUICK=1  skip the notebooks with training or animation cells
  NB_TEST_TIMEOUT  timeout per notebook in seconds (default: 600)
  NB_TEST_FOLDER   notebook folder to test (default: this folder)

These tests are not part of the main suite (`pytest` collects `tests/` only): invoke this file
explicitly. Notebooks 22, 31, 35 and 42 download datasets from Zenodo on first run; 43 integrates
the Kuramoto-Sivashinsky equation and needs torch; 44 needs torch and the Kolmogorov record
(jax-cfd, not distributed) in $DATA_ROOT/Kolmogorov/jax-cfd/.
Notebooks 40 and 41 train their echo state networks on the first run (41: about two minutes) and keep them in the
ESN store (results/esn_configs).

To execute a single notebook in place:

    cd tutorials/1_Introduction_to_real-time_DA
    jupyter nbconvert --to notebook --execute --inplace 10_real-time-DA_intro.ipynb
"""
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
