# Structure of the tutorials

Each tutorial file name starts with a two-digit code: the first digit is the topic
folder, the second the notebook's place in it. A `TODO` in the filename marks a
notebook under development — the test runner skips it.

Two arcs run through the topics: DA concepts (1) applied to physical models (2),
and reduced-order-model concepts (3) combined with DA in real time (4). Read the topics in
order. Each topic has a companion in [`scripts/`](../scripts/README.md), which runs the same
experiments from the command line: [`lorenz/`](../scripts/examples/lorenz/README.md) for topic 1,
[`thermoacoustics/`](../scripts/thermoacoustics/README.md) for topic 2 and
[`podesn_cylinder/`](../scripts/podesn_cylinder/README.md) for topics 3 and 4, and
[`caeesnda/`](../scripts/caeesnda/README.md) for notebooks 45 and 46.

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
* [x] 45 - Real-time DA with the convolutional autoencoder echo state network (CAE-ESN) on the Kuramoto–Sivashinsky equation ([Özalp, Nóvoa & Magri 2026](https://doi.org/10.1016/j.cma.2025.118600))
* [x] 46 - Real-time DA with the CAE-ESN on the Kolmogorov flow (the record, computed with jax-cfd, is not distributed)
