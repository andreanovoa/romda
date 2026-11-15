# Structure of the tutorials

Each tutorial file name starts with a two-digit code: the first digit is the topic
folder, the second the notebook's place in it. A `TODO` in the filename marks a
notebook under development — the test runner skips it.

Two arcs run through the topics: DA concepts (1) applied to physical models (2),
and reduced-order-model concepts (3) combined with DA in real time (4). Read the topics in
order. Each topic has a companion in [`scripts/`](../scripts/README.md), which runs the same
experiments from the command line: [`lorenz/`](../scripts/examples/lorenz/README.md) for topic 1,
[`thermoacoustics/`](../scripts/thermoacoustics/README.md) for topic 2 and
[`podesn_cylinder/`](../scripts/podesn_cylinder/README.md) for topics 3 and 4.

### Repo-specific tutorials [0x]
- [x] 00 - Class Model
- [x] 02 - Class ESN_model -- combining 00 and 01
- [x] 04 - Class Observations
- [x] 05 - Class Bias
- [x] 06 - Class Estimator

***

### Introduction to real-time data assimilation [1x]

* [x] 10 - Introduction to real-time DA from a Bayesian perspective: MAP estimation and the classical Kalman filter on a linear model.
* [x] 11 - Introduction to ensemble DA: the EnKF, twin experiment on the Van der Pol model, Monte Carlo convergence to the KF.
* [x] 13 - Augmented formulation: combined state and parameter estimation
* [x] 15 - Chaos and DA: twin experiment on the Lorenz63
* [x] 16 - Model bias and DA: bias-aware DA with a twin experiment on the Van der Pol model
* [x] 17 - Bimodal posterior: Lorenz63 observed through z alone, and the Gaussian-mixture ensemble filter

***

### Data assimilation on thermoacoustics [2x]

* Longitudinal thermoacoustics ([Nóvoa et al. 2023](https://doi.org/10.1016/j.cma.2023.116502))
    * [x] 20 - The Rijke tube, from the low-order model (Galerkin method) to real-time DA
    * [x] 21 - Bias-aware DA on the Rijke tube

* Azimuthal thermoacoustics  ([Nóvoa et al. 2024](https://doi.org/10.1017/jfm.2024.1052))
    * [x] 22 - The annular combustor, from the low-order model and the experimental data to a real-data digital twin

***

### Introduction to reduced-order models [3x]

* [x] 31 - POD, Sieber SPOD and Towne SPOD on the cylinder wake ([Nóvoa & Magri 2025](https://doi.org/10.1007/978-3-031-97567-7_5))
* [x] 35 - The packaged POD-ESN reduced-order model on the cylinder wake


***

### Real-time data assimilation on reduced-order models [4x]

* [x] 44 - Real-time DA on the POD-ESN cylinder-wake model ([Nóvoa & Magri 2025](https://doi.org/10.1007/978-3-031-97567-7_5))

****
****

# Running the tutorials as tests
`test_tutorials.py` (in this folder) executes the notebooks through pytest, e.g.:

```bash
SUBFOLDERS='["0","1"]' python -m pytest -s --log-cli-level=INFO tutorials/test_tutorials.py  # from the repository root
```

Configuration via environment variables:
- `SUBFOLDERS` — categories to run, e.g. `'["0","1"]'` (default: all, `'["0","1","2","3","4"]'`)
- `NB_TEST_QUICK=1` — skip notebooks containing training or animation cells
- `NB_TEST_TIMEOUT` — per-notebook timeout in seconds (default: 600)
- `NB_TEST_FOLDER` — notebook folder to test (default: this folder)

These tests are not part of the main suite (`pytest` collects `tests/` only) — invoke
`tutorials/test_tutorials.py` explicitly. Notebooks 22, 31, 35 and 44 download datasets
from Zenodo on first run.

To (re-)execute a single notebook in place:
```bash
cd tutorials/1_Introduction_to_real-time_DA
jupyter nbconvert --to notebook --execute --inplace 10_real-time-DA_intro.ipynb
```