# Regularized bias-aware EnKF in thermoacoustics

This page replicates the main results of Nóvoa, Racca & Magri (2023), *Inferring unknown unknowns: regularized
bias-aware ensemble Kalman filter*, Comput. Methods Appl. Mech. Eng. ([arXiv:2306.04315](https://arxiv.org/abs/2306.04315)),
with the current code: the Van der Pol oscillator (Figs. 6–8) and the time-delayed Rijke tube (Figs. 10–12 and
Table 2).

romda implements the filter of the erratum of the paper. The two forms of the filter coincide for one observable
(Van der Pol) and differ for the six microphones of the Rijke tube. The conclusions are therefore compared, and not
the digits. Three conclusions are drawn:

1. **The bias is recovered.** The regularized filter infers a bias of small norm for $\gamma \geq 5$ in the Van der
   Pol oscillator and for $1.25 \leq \gamma \leq 5$ in the Rijke tube, as in the paper.
2. **The number of training sets matters less than in the paper.** The error is almost flat in $L$, so the optimum
   of Table 2 moves.
3. **The parameters converge, but not to one set of values.** The final parameters of the Van der Pol oscillator
   depend on the realization of the run, as in the paper.

## Set-up

A low-order model with uncertain parameters is biased with respect to the truth. The regularized bias-aware EnKF
(r-EnKF) estimates the state and the parameters of the model, whilst an echo state network estimates the model
bias. The regularization factor, $\gamma$, penalizes the norm of the bias in the analysis: $\gamma = 0$ recovers
the bias-unaware filter. The network is trained on $L$ sets of innovations, which are augmented with scaled copies.

Two errors are reported, both normalized root-mean-square (RMS) errors with respect to the truth. The *biased*
error is that of the model estimate, and the *unbiased* error is that of the model estimate corrected with the
estimated bias. A successful run has a biased error equal to the norm of the true bias and a small unbiased error.

## Van der Pol oscillator

### The bias is recovered for γ ≥ 5

Without regularization, the filter keeps the bias: the biased error is 0.47–0.59 after the assimilation, against
0.50 before it (0.51 in the paper). For $L = 10$ and $L = 50$ and $\gamma \geq 5$, the biased error is 0.17, which
is the norm of the true bias (0.165). One result differs from the paper: $L = 1$ with augmentation behaves as
$L = 10$ and $L = 50$, whereas the paper finds it almost insensitive to $\gamma$.

![Figure 6 of the paper with the current code](../../figs/replications/cmame2023/vdp_fig06.png)

*Figure 6. Van der Pol oscillator, seed 0: RMS errors after the assimilation for four training sets of the bias
network and three regularization factors.*

### The parameters depend on the realization

The filter recovers the bias, but not one set of parameters $(\beta, \zeta, \kappa)$. This agrees with the paper,
which normalizes the parameters by their initial mean, does not expect them to converge to one, and reports many
combinations with an accurate analysis (its Figs. 7 and F.20).

Over six seeds (378 runs), the scatter of the final parameters is of the size of Fig. F.20 of the paper and grows
with $\gamma$. The combination $(\beta - \zeta)/\kappa$ is better
identified than each parameter: it stays within 25 % of the truth for $5 \leq \gamma \leq 10$ and within 36 % for
larger $\gamma$, whereas $\beta$ alone spans a factor of four.

![The combination of the Van der Pol parameters over six seeds](../../figs/replications/cmame2023/vdp_fig08_ratio.png)

*$(\beta - \zeta)/\kappa$ at the end of the assimilation over its true value, for six seeds and (a) $L = 1$,
(b) $L = 10$ and (c) $L = 50$. The band spans the seeds.*

The parameters themselves scatter more, as the table and the figure below show.

| $L$ | $\gamma$ | $\beta$ over its initial mean | $\zeta$ | $\kappa$ | $(\beta - \zeta)/\kappa$ (truth 5.88) | Biased error below 0.2 |
|---|---|---|---|---|---|---|
| 1 | 5–10 | 0.90–1.28 | 0.71–1.12 | 0.77–1.14 | 4.5–6.3 | 100 % |
| 1 | 11–20 | 0.73–1.51 | 0.57–1.34 | 0.73–1.17 | 4.9–7.0 | 100 % |
| 10 | 5–10 | 0.98–1.46 | 0.78–1.32 | 0.81–1.10 | 4.9–6.9 | 100 % |
| 10 | 11–20 | 0.94–1.51 | 0.76–1.34 | 0.52–1.31 | 5.4–7.6 | 97 % |
| 50 | 5–10 | 0.86–1.40 | 0.68–1.22 | 0.79–1.09 | 4.7–6.6 | 100 % |
| 50 | 11–20 | 0.36–1.55 | 0.27–1.41 | 0.47–1.07 | 3.9–8.0 | 93 % |

![Van der Pol oscillator over six seeds](../../figs/replications/cmame2023/vdp_fig08_seeds.png)

*Figure 8 over six seeds (markers; lines: seed 0), for $L = 1$, 10 and 50 (top to bottom). Left: RMS errors after
the assimilation. Centre: parameters at the end of the assimilation over their initial mean; dotted lines are the
true values. Right: $(\beta - \zeta)/\kappa$ over its true value.*

The scatter of the paper originates from the realization, and not from the training or the update of the bias
network. The code of the paper creates its random generator once and does not reseed it between the runs of a
sweep, so each $\gamma$ has its own realization of the perturbed observations. With one seed for every run, the
final parameters vary smoothly with $\gamma$. (A seed here also changes the noise of the truth and the initial
ensemble, whereas in the paper only the perturbed observations differ between $\gamma$.)

### A prior far from the truth

The prior of the paper is within 18 % of the truth, so the parameters move little. With a prior mean of
(37.5, 27.5, 5.1) against the truth (75, 55, 3.4), which is not in the paper, the biased error still falls to the
norm of the true bias for $\gamma \geq 5$. The parameters do not reach the truth: $\beta$ ends between 33 and 93,
$\zeta$ between 18 and 60, and $(\beta - \zeta)/\kappa$ between 3.2 and 5.0. Up to 76 analyses are rejected
because members cross the limits of the parameters, which confounds this comparison.

![Van der Pol oscillator with a prior far from the truth](../../figs/replications/cmame2023/vdp_fig08_far_prior.png)

*Figure 8 with the prior far from the truth, seed 0, for $L = 1$, 10 and 50 (top to bottom). (a, e, i) RMS errors
against $\gamma$; (b, f, j) parameters at the end of the assimilation over their true values; (c, g, k) biased and
(d, h, l) unbiased errors in time.*

## Time-delayed Rijke tube

For $1.25 \leq \gamma \leq 5$, the filter recovers both biases. Small regularization fails ($\gamma \leq 0.5$), as
in the paper.

| Bias | Error | Replication (median) | Paper |
|---|---|---|---|
| Linear | biased | 0.185 | 0.1817 |
| Linear | unbiased | 0.021 | 0.0157 |
| Nonlinear | biased | 0.227 | 0.2279 |

Three results differ from the paper.

- **$\gamma \gtrsim 5$.** Only isolated runs fail (14 % with the linear bias and 28 % with the nonlinear bias),
  whereas the paper reports inaccurate solutions.
- **Optimum of Table 2.** The error is almost flat in $L$, so the optimum moves: $L = 10$ and $\gamma = 5$ with the
  linear bias, against $L = 100$ and $\gamma = 1.75$ in the paper.
- **Unbiased error with the nonlinear bias.** It depends on the trained network: 0.27 at the optimum of the paper,
  against 0.079 in the paper.

![Figure 10 of the paper with the current code](../../figs/replications/cmame2023/rijke_fig10.png)

*Figure 10. Rijke tube: sum of the biased and unbiased RMS errors after the assimilation against the regularization
factor and the number of training sets, for (a) the linear and (b) the nonlinear bias. Stars: optimum.*

![Figure 11 of the paper with the current code](../../figs/replications/cmame2023/rijke_fig11.png)

*Figure 11. Rijke tube with (a–d) the linear and (e–h) the nonlinear bias.
(a, e) RMS errors and (b, f) parameters over their true values against $\gamma$; (c, g) biased and (d, h) unbiased
errors in time.*

## Differences from the code of the paper

The training and the update of the bias network are the same as in the code of the paper: the training data and
their augmentation, the reset of the bias after an analysis, the open-loop Jacobian and the washout of 30 steps.
The remaining differences are listed below.

| Step | Code of the paper | Replication | Effect |
|---|---|---|---|
| Spread of the training ensemble | 0.3 | 0.5, as Table E.3 | tested together: the parameters stay smooth with $\gamma$ |
| Input normalization | range | mean and range | as above |
| Reservoir connectivity | 5 | 3 | as above |
| Filter equations | before the erratum | erratum | identical for one observable |
| Parameter outside its limits | analysis repeated with the limit as a pseudo-observation | analysis rejected, with inflation | not tested |
| Limits of $\beta$ and $\zeta$ | (20, 120) | (5, 120) | not tested |
| Members take their own parameters | 6 ms before the first datum | 20 ms | not tested |

Three choices depart from the set-up of the paper: the Rijke members are drawn 0.5 s before the first datum (12 ms
in the paper), the training noise is 3 % as in the code of the paper, and Fig. 10 has 20 values of $\gamma$ in
place of 41.

## What is not replicated

The form of the filter before the erratum is not run, so the differences in the Rijke tube cannot be attributed to
the erratum. The time-dependent bias (Figs. 13–14 and the third row of Table 2) and a second seed for the Rijke
tube are left for future work.

## Run it

```bash
python scripts/thermoacoustics/cmame2023/replicate.py vdp --save-figs            # Figs. 6-8: 66 runs, 5 min
python scripts/thermoacoustics/cmame2023/replicate.py vdp_seeds --save-figs      # Fig. 8 over six seeds: 378 runs, 30 min
python scripts/thermoacoustics/cmame2023/replicate.py vdp_far_prior --save-figs  # prior far from the truth
python scripts/thermoacoustics/cmame2023/replicate.py rijke --save-figs          # Figs. 10-12, Table 2: 400 runs, 3.5 h
```

The settings are in `scripts/thermoacoustics/cmame2023/vdp.yml` and `rijke.yml`. The runs are recorded as they
finish, so an interrupted call resumes, and `--plot` redraws the figures from the records.
