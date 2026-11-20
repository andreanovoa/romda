# Thermoacoustics

## What the example shows

Real-time data assimilation of thermoacoustic oscillations, the self-excited acoustic waves driven by
unsteady heat release in combustors.

- **`rijke`**: a twin experiment on the Rijke tube, a Galerkin low-order model of longitudinal
  thermoacoustics. The heat-release gain `beta` and the flame time delay `tau` are estimated together
  with the acoustic state.
- **`vdp`**: a twin experiment on the Van der Pol oscillator, the simplest model of a thermoacoustic
  limit cycle, estimating the parameter `beta`. The damping `zeta` stays at its true value: on the limit
  cycle the data fix only the difference `beta - zeta`, so estimating both spreads the ensemble along it.
- **`annular`**: the real-time digital twin of an annular combustor. The truth is raw experimental
  pressure data, the forecast model is a low-order model of azimuthal thermoacoustics, and an echo state
  network estimates the model bias, so that the regularized bias-aware EnKF (rBA-EnKF) assimilates the
  raw data. `annular.yml` holds the settings of Nóvoa, Noiray, Dawson & Magri (2024) at an equivalence ratio
  of 0.5125, and `annular_release.yml` those of the tutorial `32_DA_annular_raw` of the release v1.0.

## Main results of Nóvoa, Racca & Magri (2023)

`cmame2023/replicate.py` repeats the two test cases of the paper with the regularized bias-aware EnKF of romda
and redraws their main figures: the Van der Pol oscillator with the bias $b = \cos\eta$ (figures 6 to 8) and the
time-delayed Rijke tube with a linear and a nonlinear bias (figures 10 to 12 and Table 2). The settings are those of
Tables 3 and 4 of the paper (`cmame2023/vdp.yml`, `cmame2023/rijke.yml`). The time-dependent bias (figures 13 to
15, third row of Table 2) and the studies of the noise and of the ensemble size are not repeated.

romda implements the filter of the erratum of the paper, which transposes the Jacobian of the bias. The two forms
coincide for one observable (Van der Pol) and differ for the six microphones of the Rijke tube, whose published
numbers come from the form before the erratum. The conclusions are therefore compared, and the digits only as an
indication. Each case is one realization (seed 0), as in the paper.

### What matches

Errors are the root-mean-square (RMS) errors of the paper, relative to the truth with its bias, over one $t_\mathrm{CR}$
after the assimilation; "biased" is the model estimate and "unbiased" adds the bias estimate of the network.

- **Without regularization the model keeps its bias.** With $\gamma = 0$ the unbiased estimate is accurate but the
  biased error stays at the level before the assimilation: 0.47 to 0.59 against 0.50 for the Van der Pol oscillator
  (0.51 in the paper), and a median of 0.83 (linear bias) and 1.29 (nonlinear) against 1.1 for the Rijke tube.
- **With regularization the filter finds a bias of small norm.** Van der Pol, $L = 10$ and $50$, $\gamma \geq 5$:
  the biased error is 0.17, the norm of the true bias (0.165), and the unbiased error 0.05 to 0.07. Larger $\gamma$
  gives larger jumps of the parameters in the first analyses (figure 7).
- **The parameters converge, but not to one set of values.** The paper normalizes the parameters by their initial
  mean, does not expect them to converge to 1, and finds many combinations $(\beta, \zeta, \kappa)$ with a 'good'
  analysis (its figure F.20). Here, with the prior of the paper (70, 60, 4), which is within 18 % of the truth
  (75, 55, 3.4), $(\beta - \zeta)/\kappa$ ends between 5.9 and 6.4 for $\gamma \geq 5$ and every $L$ (truth 5.88,
  prior 2.5), whereas $\beta$ ends between 81 and 106 and $\zeta$ between 62 and 80 for $L = 10$. With a prior
  further from the truth (the variant `vdp_far_prior`, not in the paper: mean 37.5, 27.5, 5.1), the biased error
  still falls to the norm of the true bias for $\gamma \geq 5$, but the parameters do not reach the truth:
  $\beta$ ends between 33 and 93, $\zeta$ between 18 and 60, and $(\beta - \zeta)/\kappa$ between 3.2 and 5.0. Up
  to 76 of the analyses are rejected in that variant, because members cross the limits of the parameters.
- **Rijke tube, linear bias.** For $1.25 \leq \gamma \leq 5$ the biased error has a median of 0.185 over the ten
  values of $L$ and the unbiased error one of 0.021 (paper, at its minimum: 0.1817 and 0.0157). The biased error is
  below that of the true bias (0.262; 0.2623 in the paper): the filter finds a bias of smaller norm than the true one.
- **Rijke tube, nonlinear bias.** For $1.25 \leq \gamma \leq 5$ the biased error has a median of 0.227, and is within 15 % of it in 98 % of the runs,
  slightly above the true bias (0.222), as in the paper (0.2279 against 0.2217), and the parameters are unique:
  the medians of $\tau$ and $\beta$ are within 1 % and 2 % of the truth. The minimum of the sum of the two errors is at
  $L = 70$, $\gamma = 3$ (0.2245 and 0.0803); the paper finds $L = 60$, $\gamma = 2.75$ (0.2279 and 0.0792).
- **Small $\gamma$ fails.** For $\gamma \leq 0.5$ the biased error of the Rijke tube stays between 0.25 and 1.6.
- **The single training set without augmentation is the least robust** (figure 6a): its unbiased error is 0.12 to
  0.14, two to three times that of the other sets, and at $\gamma = 20$ the filter rejects 91 of 334 analyses.

### What does not match

- **Scatter of the parameters with $\gamma$: explained by the realization.** With seed 0, the final parameters
  vary smoothly with $\gamma$, whereas in figures 7 and F.20 of the paper they scatter between 0.2 and 2.3 times
  their initial mean. The code of the paper creates its random generator once and does not reseed it between the
  runs of a sweep, so each $\gamma$ has its own realization of the perturbed observations. Over six seeds
  (`replicate.py vdp_seeds`, 378 runs), for $\gamma > 10$ and $L = 50$ the final $\beta$ lies between 0.36 and
  1.55 times its initial mean, $\zeta$ between 0.27 and 1.41 and $\kappa$ between 0.47 and 1.07, while the biased
  error stays below 0.2 in 93 % of the runs. The options of the network of the paper's code (no centring of the
  input, connectivity 5, training spread 0.3) do not change this.
- **Figure 6(b) and figure 8, $L = 1$.** In the paper the filter with $L = 1$ is almost unaffected by $\gamma$ and
  its bias has a large norm. Here the augmented set with $L = 1$ behaves as $L = 10$ and $50$ (biased error 0.17 for
  $\gamma \geq 5$). The trend with $L$ of the paper (more members, wider range of $\gamma$) is weak: with $L = 10$
  one or two analyses are rejected for $\gamma \geq 18$, and none with $L = 50$.
- **Large $\gamma$ in the Rijke tube.** The paper reports inaccurate solutions for $\gamma \gtrsim 5$. Here the
  errors for $6 \leq \gamma \leq 10$ have the medians of the range below, with isolated failures: 14 % of the runs
  with the linear bias and 28 % with the nonlinear one.
- **The optimum over $L$ and $\gamma$ is flat.** Figure 10 has no trend with $L$: the differences between rows
  come from the network that each $L$ trains, and the minima of Table 2 move within a plateau (linear bias:
  $L = 10$, $\gamma = 5$ here against $L = 100$, $\gamma = 1.75$ in the paper, with biased errors within 1 % of each other).
- **Nonlinear bias, unbiased error.** After the assimilation it depends on the network: 0.08 for $L = 70$ and
  0.13 to 0.4 for most of the others (median 0.145), against 0.079 at the minimum of the paper. At the minimum of
  the paper ($L = 60$, $\gamma = 2.75$) it is 0.27.
- **$\beta$ with the linear bias.** The gain $\beta$ ends at 4.9, against its limit of 5 (truth: 4.2), and more
  than half of the analyses are rejected because a member crosses that limit. The paper normalizes the parameters by
  their initial mean and gives no values to compare with.

Table 2 of the paper and these runs (`results/table2.md`; "end": last $t_\mathrm{CR}$ of the assimilation;
"after": first $t_\mathrm{CR}$ after it):

| bias | source | true biased RMS | end: $L$, $\gamma$ | biased | unbiased | after: $L$, $\gamma$ | biased | unbiased |
|---|---|---|---|---|---|---|---|---|
| linear | paper | 0.2623 | 10, 3.50 | 0.1761 | 0.0244 | 100, 1.75 | 0.1817 | 0.0157 |
| linear | romda | 0.2620 | 60, 2.00 | 0.1787 | 0.0132 | 10, 5.00 | 0.1795 | 0.0158 |
| linear | romda at the minima of the paper | | 10, 3.50 | 0.1829 | 0.0159 | 100, 1.75 | 0.1793 | 0.0259 |
| nonlinear | paper | 0.2217 | 60, 2.75 | 0.2303 | 0.0799 | 60, 2.75 | 0.2279 | 0.0792 |
| nonlinear | romda | 0.2219 | 90, 10.00 | 0.2268 | 0.0625 | 70, 3.00 | 0.2245 | 0.0803 |
| nonlinear | romda at the minima of the paper | | 60, 2.75 | 0.2280 | 0.2193 | 60, 2.75 | 0.2224 | 0.2684 |

### Choices where the paper is silent, and differences from it

- **Code of the paper.** The settings that the paper does not state follow its code (commit `f557a3d` of this
  repository): multiplicative noise of 1 % on the observations, $\mathsf{C}_{dd} = 0.01 \max|d|^2\,\mathbb{I}$, an
  inflation of 1.002, a noise of 3 % on the training input of the network, a validation time of 0.1 s for the Van der
  Pol network (Table 3 prints 0.01 s), and the bias between two network steps as a cubic spline. With the default
  training noise of romda (1 %) the networks of five of the ten values of $L$ diverge in closed loop after the
  assimilation of the Rijke tube.
- **Initial ensemble.** Van der Pol: as in the paper, the members are drawn 0.02 s before the first datum about the
  state of the model with the prior parameters (`spinup_cr`, `t_min`). Rijke: the members are drawn 0.5 s before the
  first datum and forecast with their own parameters. Drawn 12 ms before it, as in the paper, the ensemble is in
  antiphase with the truth with a spread of 20 %, and the EnKF moves $\tau$ to twice its true value within two
  analyses, with or without a bias in the truth.
- **Parameters outside their limits.** The code of the paper assimilates a crossed limit as an observation; romda
  rejects the analysis. The normal prior of $\beta$ (mean 4, standard deviation 0.8, limit 5) has members outside,
  which `romda.run` now reflects into the limits; before, no analysis was accepted for $\gamma \leq 0.25$.
- **Sweep of figure 10.** $L = 10, 20, \dots, 100$ as in the paper, and 20 values of $\gamma$ between 0 and 10 (41 in
  the paper): steps of 0.25 or 0.5 up to 5, and of 1 beyond.
- **Windows and limits.** The error before the assimilation is taken from the start of the ensemble to the first
  datum. The limits of the Van der Pol parameters are those of `dynamodels` ($\beta, \zeta \in (5, 120)$,
  $\kappa \in (0.1, 20)$), wider than those of Table 3.
- **Figures.** The parameters are normalized by their true values (by their initial mean in the paper), and
  figures 8 and 11 leave out the correlation.

### How to run

```bash
python scripts/thermoacoustics/cmame2023/replicate.py vdp --save-figs      # figures 6-8: 66 runs, 5 min
python scripts/thermoacoustics/cmame2023/replicate.py vdp_far_prior --save-figs   # the same with a prior far from the truth
python scripts/thermoacoustics/cmame2023/replicate.py vdp_seeds --save-figs   # figure 8 and (beta - zeta)/kappa over six seeds: 378 runs, 30 min
python scripts/thermoacoustics/cmame2023/replicate.py rijke --save-figs    # figures 10-12, Table 2: 400 runs, 3.5 h
python scripts/thermoacoustics/cmame2023/replicate.py rijke --plot --save-figs   # the figures, from the records
python scripts/thermoacoustics/cmame2023/replicate.py rijke --quick        # 12 runs, as a smoke test
pytest scripts/thermoacoustics/cmame2023
```

A run of the Rijke tube takes 18 s (50 members on 24 cores) and the network of each $L$ and bias 1 to 7 min; a
process stays below 2 GB. The runs are recorded in `cmame2023/results/<case>.jsonl` as they finish, so an
interrupted call resumes, and the figures are written to `cmame2023/figs/` (`vdp_fig06_training_sets.png`,
`vdp_fig07_time_series.png`, `vdp_fig08_gamma.png`, `rijke_fig10_L_gamma.png`, `rijke_fig11_gamma.png`,
`rijke_fig12_time_series.png`). Both folders are untracked.

## Published works

- Nóvoa, Noiray, Dawson & Magri (2024). A real-time digital twin of azimuthal thermoacoustic
  instabilities. *Journal of Fluid Mechanics*. [doi:10.1017/jfm.2024.1052](https://doi.org/10.1017/jfm.2024.1052).
  The code of the paper is the release
  [v1.0](https://github.com/andreanovoa/romda/releases/tag/v1.0).
- Nóvoa, Racca & Magri (2024). Inferring unknown unknowns: regularized bias-aware ensemble Kalman filter.
  *Computer Methods in Applied Mechanics and Engineering* 418, 116502.
  [doi:10.1016/j.cma.2023.116502](https://doi.org/10.1016/j.cma.2023.116502). The rBA-EnKF and the Rijke
  tube ([legacy repository](https://github.com/MagriLab/rBA-EnKF)).
  `cmame2023/replicate.py` reproduces its main results.
- Nóvoa & Magri (2022). Real-time thermoacoustic data assimilation. *Journal of Fluid Mechanics*.
  [doi:10.1017/jfm.2022.653](https://doi.org/10.1017/jfm.2022.653)
  ([legacy repository](https://github.com/MagriLab/Real-time-TA-DA)).
- Nóvoa & Magri (2022). Bias-aware thermoacoustic data assimilation. *51st International Congress and
  Exposition on Noise Control Engineering*
  ([legacy repository](https://github.com/MagriLab/IN22-Bias-aware-TADA)).

These cases accompany topic 2 of the [tutorials](../../tutorials/README.md).

## Data

- `rijke` and `vdp`: none; the truth is generated on the fly. These two cases have no script or sweep of
  their own: they run through `run_config.py` or `python -m romda.run`.
- `annular`: the azimuthal pressure measurements of [Zenodo record 15609832](https://zenodo.org/records/15609832),
  one file per equivalence ratio. The first run downloads them into `data/annular/` at the repository
  root.

## How to run

From the repository root:

```bash
python scripts/run_config.py thermoacoustics/rijke --save-figs
python scripts/run_config.py thermoacoustics/vdp --members           # also plot the ensemble members
python -m romda.run scripts/thermoacoustics/annular.yml --set truth.ER=0.5375
```


The first annular run trains the echo state network of the bias, which takes much longer than the other
cases. It caches the training data under `results/annular/` and the trained network under
`results/esn_configs/`, both at the repository root; later runs reload them. There is one
set of training data, and one network, for each equivalence ratio, assimilation window and set of parameter
ranges.


