"""
pod_utils.py
============
Standalone POD/SPOD decomposition algorithms (functions only). The linear
ROM classes that wrap these algorithms with a sklearn-style fit/encode/decode
interface (`POD`, `SPOD`) live alongside this module in
`romda.models.data_driven.autoencoders`.

Functions
---------
energy_fraction         : relative/cumulative energy from singular values
snapshot_pod            : Snapshot POD — exact solver (Sirovich 1987)
snapshot_pod_randomized : Randomized snapshot POD (Halko et al. 2011)
spod_sieber             : SPOD via filtered correlation matrix (Sieber et al. 2016)
spod_towne              : SPOD via Welch CSD (Towne et al. 2018)
print_spod_towne_summary: pretty-print Towne SPOD parameters

Usage
-----
::

    from romda.models.data_driven.autoencoders import POD, SPOD

    pod = POD(n_modes=20).fit(Q)      # Q: (N_x, N_t) zero-mean-able data matrix
    Z   = pod.encode(Q)               # (N_modes, N_t) -- POD coefficients
    Q_r = pod.decode(Z)               # (N_x, N_t) -- reconstruction

Convention (Mendez 2023)
------------------------
  Psi   (N_x, N_modes)   spatial modes, orthonormal columns
  Phi   (N_modes, N_t)   temporal coefficients  Phi[k, t] = Psi_k . q(t)
  Sigma (N_modes,)       singular values  Sigma_k = sqrt(lambda_k)

References
----------
[Sirovich 1987]  Quart. Appl. Math., 45(3), 561-571, doi:10.1090/qam/910462.
[Halko 2011]     SIAM Review, 53(2), 217-288.
[Sieber 2016]    JFM 792, 798-828.
[Towne 2018]     JFM 847, 821-867.
[Mendez 2023]    Data-Driven Fluid Mechanics, Cambridge University Press.
"""

import numpy as np
from scipy.signal import get_window
from scipy.special import gammaincinv
from sklearn.utils.extmath import randomized_svd  # type: ignore[import-untyped]


def energy_fraction(Sigma):
    r"""
    Relative energy fraction and cumulative energy per mode.

    From the eigenvalues $\lambda_j = \Sigma_j^2$ of the (possibly filtered)
    temporal correlation matrix,

    $$
    \mathrm{rel}_j = \frac{\lambda_j}{\sum_k \lambda_k}, \qquad
    \mathrm{cum}_j = \sum_{k \le j} \mathrm{rel}_k.
    $$

    Parameters
    ----------
    Sigma : np.ndarray
        Singular values, as returned by `snapshot_pod`,
        `snapshot_pod_randomized` or `spod_sieber`.

    Returns
    -------
    rel : np.ndarray
        Relative energy per mode (sums to 1).
    cum : np.ndarray
        Cumulative relative energy.
    """
    lam = Sigma ** 2
    rel = lam / lam.sum()
    return rel, np.cumsum(rel)


# ─────────────────────────────────────────────────────────────────────────────
# Decomposition algorithms
# ─────────────────────────────────────────────────────────────────────────────

def snapshot_pod(Q):
    r"""Snapshot POD — exact solver.

    Solves the eigenvalue problem of the temporal correlation matrix,

    $$
    \matr{C}\matr{A} = \matr{A}\,\mathrm{diag}(\vect{\lambda}),
    \qquad \matr{C} = \matr{Q}^\mathrm{T} \matr{Q} / N_t,
    \qquad \lambda_1 \ge \lambda_2 \ge \cdots \ge 0,
    $$

    and reconstructs the (large) spatial modes from the eigenvectors
    $\matr{A}$ of the (small, $N_t \times N_t$) matrix $\matr{C}$ — the
    "method of snapshots" of Sirovich (1987):

    $$
    \matr{\Psi} = \frac{1}{\sqrt{N_t}}\,
    \matr{Q}\matr{A}\,\mathrm{diag}(\vect{\lambda})^{-1/2},
    \qquad
    \matr{\Phi} = \matr{\Psi}^\mathrm{T}\matr{Q},
    \qquad
    \matr{\Sigma} = \sqrt{\vect{\lambda}}.
    $$

    $\matr{\Psi}$ has orthonormal columns and, for the modes with
    $\lambda_j > 0$, $\matr{Q} = \matr{\Psi}\matr{\Phi}$ exactly.
    Modes with $\lambda_j \le 0$ (numerical noise) are set to zero.

    Parameters
    ----------
    Q : np.ndarray
        Zero-mean data matrix, shape $(N_x, N_t)$.

    Returns
    -------
    Sigma : np.ndarray
        Singular values (descending), shape $(N_t,)$.
    Psi : np.ndarray
        Spatial modes with orthonormal columns, shape $(N_x, N_t)$.
    Phi : np.ndarray
        Temporal coefficients, shape $(N_t, N_t)$.
    C : np.ndarray
        Temporal correlation matrix, shape $(N_t, N_t)$.

    References
    ----------
    Sirovich (1987). Turbulence and the dynamics of coherent structures.
    Part I: Coherent structures. *Quart. Appl. Math.*, 45(3), 561–571.
    doi:10.1090/qam/910462.
    """
    C = (Q.T @ Q) / Q.shape[1]
    Sigma, Psi, _ = _modes_from_correlation(Q, C)
    return Sigma, Psi, Psi.T @ Q, C  # the projection stays exact in the near-null modes


def _modes_from_correlation(Q, C):
    r"""Method-of-snapshots modes from a (possibly filtered) correlation matrix.

    With $\matr{C}\matr{A} = \matr{A}\,\mathrm{diag}(\vect{\lambda})$
    (descending), returns $\matr{\Sigma} = \sqrt{\vect{\lambda}}$, the spatial
    modes $\matr{\Psi} = \matr{Q}\matr{A}\,\mathrm{diag}(N_t\vect{\lambda})^{-1/2}$
    and the temporal coefficients
    $\matr{\Phi} = \mathrm{diag}(N_t\vect{\lambda})^{1/2}\matr{A}^\mathrm{T}$,
    so that $\matr{Q} = \matr{\Psi}\matr{\Phi}$ at full rank. Modes with
    $\lambda_j \le 0$ (numerical noise) are set to zero.
    """
    N_t      = Q.shape[1]
    lam, A   = np.linalg.eigh(C)
    idx      = lam.argsort()[::-1]
    lam, A   = lam[idx], A[:, idx]
    lam      = np.where(lam > 0, lam, 0.0)
    safe_lam = np.where(lam > 0, lam, np.inf)
    Psi      = Q @ A / np.sqrt(N_t * safe_lam)
    Phi      = np.sqrt(N_t * lam)[:, None] * A.T
    return np.sqrt(lam), Psi, Phi


def snapshot_pod_randomized(Q, n_modes=20, n_iter=4, random_state=None):
    r"""Randomized snapshot POD.

    Computes a truncated randomized SVD of $\matr{Q}$ (via
    ``sklearn.utils.extmath.randomized_svd``),

    $$
    \matr{Q} \approx \matr{U}\,\matr{S}\,\matr{V}^\mathrm{T},
    $$

    and returns

    $$
    \matr{\Psi} = \matr{U}, \qquad
    \matr{\Sigma} = \matr{S} / \sqrt{N_t}, \qquad
    \matr{\Phi} = \matr{\Psi}^\mathrm{T}\matr{Q}.
    $$

    The $\matr{\Sigma} = \matr{S}/\sqrt{N_t}$ scaling matches the
    eigenvalue-based normalisation of `snapshot_pod`, since
    $\matr{Q}^\mathrm{T}\matr{Q}/N_t = \matr{V}(\matr{S}^2/N_t)\matr{V}^\mathrm{T}$.

    Parameters
    ----------
    Q : np.ndarray
        Zero-mean data matrix, shape $(N_x, N_t)$.
    n_modes : int
        Leading modes to compute. Default 20.
    n_iter : int
        Power-iteration steps. Default 4.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    Sigma : np.ndarray
        Singular values (descending), shape $(N_\mathrm{modes},)$.
    Psi : np.ndarray
        Spatial modes (approximately orthonormal), shape $(N_x, N_\mathrm{modes})$.
    Phi : np.ndarray
        Temporal coefficients, shape $(N_\mathrm{modes}, N_t)$.

    References
    ----------
    Halko, Martinsson & Tropp (2011). Finding structure with randomness.
    *SIAM Review*, 53(2), 217–288.
    """
    N_x, N_t = Q.shape
    n_modes  = min(n_modes, N_x, N_t)
    U, s, _ = randomized_svd(Q, n_components=n_modes,
                             n_iter=n_iter, random_state=random_state)
    Psi   = U
    Sigma = s / np.sqrt(N_t)
    Phi   = Psi.T @ Q
    return Sigma, Psi, Phi


def _filter_kernel(Nf, kind='gaussian'):
    """
    Symmetric, unit-sum 1-D low-pass filter kernel of half-width Nf
    (length ``2*Nf + 1``), used by the SPOD-Sieber diagonal filter.

    Parameters
    ----------
    Nf : int
        Filter half-width. ``Nf=0`` returns the trivial kernel ``[1.0]``.
    kind : str
        ``'gaussian'``, ``'box'`` or ``'hann'``.

    Returns
    -------
    np.ndarray
        Normalised kernel of length ``2*Nf + 1`` (or 1 if ``Nf=0``).
    """
    if Nf == 0:
        return np.array([1.0])
    k = np.arange(-Nf, Nf + 1, dtype=float)
    if kind == 'box':
        g = np.ones(2 * Nf + 1)
    elif kind == 'gaussian':
        g = np.exp(-0.5 * (k / (Nf / 2.0)) ** 2)
    elif kind == 'hann':
        g = np.hanning(2 * Nf + 1)
    else:
        raise ValueError(f"Unknown filter kind '{kind}'. "
                         "Choose 'box', 'gaussian' or 'hann'.")
    return g / g.sum()


def spod_sieber(Q, Nf, kind='gaussian'):
    r"""Sieber spectral POD — diagonally filtered correlation matrix.

    Filters the temporal correlation matrix
    $\matr{C} = \matr{Q}^\mathrm{T}\matr{Q}/N_t$ along its diagonals with a
    normalised kernel $g_k$ of half-width ``Nf`` (see `_filter_kernel`),
    continuing $\matr{C}$ periodically at the ends (Sieber et al. 2016, eq. 2.9),

    $$
    S_{ij} = \sum_{k=-N_f}^{N_f} g_k\, C_{i+k,\,j+k},
    $$

    then solves the "method of snapshots" eigenvalue problem
    $\matr{S}\matr{A} = \matr{A}\,\mathrm{diag}(\vect{\lambda})$ and returns
    the Sieber temporal coefficients and spatial modes,

    $$
    \matr{\Phi} = \mathrm{diag}(N_t\vect{\lambda})^{1/2}\matr{A}^\mathrm{T},
    \qquad
    \matr{\Psi} = \matr{Q}\matr{A}\,\mathrm{diag}(N_t\vect{\lambda})^{-1/2},
    \qquad
    \matr{\Sigma} = \sqrt{\vect{\lambda}},
    $$

    so that $\matr{Q} = \matr{\Psi}\matr{\Phi}$ at full rank. The temporal
    coefficients are orthogonal; for ``Nf > 0`` the spatial modes are not.
    ``Nf=0`` skips the filtering step ($\matr{S} = \matr{C}$) and recovers
    snapshot POD exactly. With the periodic continuation $\matr{S}$ becomes
    circulant as ``Nf`` approaches $N_t/2$ (box kernel), so the modes approach
    Fourier (DFT) modes.

    Parameters
    ----------
    Q : np.ndarray
        Zero-mean data matrix, shape $(N_x, N_t)$.
    Nf : int
        Filter half-width (0 recovers POD; $N_t/2$ approaches the DFT).
    kind : str
        ``'gaussian'``, ``'box'`` or ``'hann'``.

    Returns
    -------
    Sigma : np.ndarray
        Singular values (descending), shape $(N_t,)$.
    Psi : np.ndarray
        Spatial modes (not orthonormal for ``Nf > 0``), shape $(N_x, N_t)$.
    Phi : np.ndarray
        Temporal SPOD coefficients, shape $(N_t, N_t)$.
    C_tilde : np.ndarray
        Filtered correlation matrix $\matr{S}$, shape $(N_t, N_t)$.

    References
    ----------
    Sieber, Paschereit & Oberleithner (2016). Spectral proper orthogonal
    decomposition. *J. Fluid Mech.*, 792, 798–828.
    """
    C = (Q.T @ Q) / Q.shape[1]
    g = _filter_kernel(Nf, kind)
    C_tilde = sum(g[k + Nf] * np.roll(C, (-k, -k), axis=(0, 1)) for k in range(-Nf, Nf + 1))
    return (*_modes_from_correlation(Q, C_tilde), C_tilde)


# ─────────────────────────────────────────────────────────────────────────────
# SPOD — Towne, Schmidt & Colonius (JFM 2018)
# ─────────────────────────────────────────────────────────────────────────────

def spod_towne(Q, dt=1.0, n_fft=None, n_ovlp=None, window='hamming',
               weight=None, conf_level=0.95):
    r"""Spectral POD via Welch-averaged cross-spectral density (Towne et al., 2018).

    Splits $\matr{Q}$ into ``n_blks`` overlapping blocks of length ``n_fft``
    (Welch's method, step ``n_fft - n_ovlp``), windows and Fourier-transforms
    each block, then — at every frequency — solves a "method of snapshots"
    eigenvalue problem across blocks (analogous to `snapshot_pod`, but with
    blocks in place of time snapshots) to avoid ever forming the full
    $N_x \times N_x$ cross-spectral density (CSD) matrix.

    For block $b = 0, \dots, n_\mathrm{blk}-1$ starting at snapshot
    $b\,(n_\mathrm{fft}-n_\mathrm{ovlp})$, the windowed block DFT at
    frequency $f_k$ is

    $$
    \hat{\vect{q}}^{(b)}_k = \frac{1}{n_\mathrm{fft}\,\bar{w}}
    \sum_{n=0}^{n_\mathrm{fft}-1} w[n]\,\vect{q}^{(b)}[n]\,
    e^{-\mathrm{i}2\pi kn/n_\mathrm{fft}}, \qquad \bar{w} = \mathrm{mean}(w),
    $$

    with $w$ the ``window``. Stacking the blocks,
    $\hat{\matr{Q}}_k = [\hat{\vect{q}}^{(1)}_k, \dots,
    \hat{\vect{q}}^{(n_\mathrm{blk})}_k] \in \mathbb{C}^{N_x \times n_\mathrm{blk}}$,
    the (weighted) cross-block Gram matrix and its eigendecomposition give the
    SPOD modes and modal energies at frequency $f_k$:

    $$
    \matr{M}_k = \frac{1}{n_\mathrm{blk}}\,
    \hat{\matr{Q}}_k^\mathrm{H}\, \matr{W}\, \hat{\matr{Q}}_k,
    \qquad
    \matr{M}_k \matr{\Theta}_k
    = \matr{\Theta}_k\, \mathrm{diag}(\vect{\lambda}_k),
    $$

    $$
    \matr{\Psi}_k = \frac{1}{\sqrt{n_\mathrm{blk}}}\,
    \hat{\matr{Q}}_k\, \matr{\Theta}_k\,
    \mathrm{diag}(\vect{\lambda}_k)^{-1/2},
    $$

    with $\matr{W} = \mathrm{diag}(\mathrm{weight})$ the spatial weight matrix
    ($\mathbb{I}$ by default). The modal energy spectrum is
    $L_k = \vect{\lambda}_k$, doubled ($L_k = 2\vect{\lambda}_k$) at
    interior frequency bins of a real, one-sided spectrum to account for the
    folded negative-frequency energy.

    Parameters
    ----------
    Q : np.ndarray
        Zero-mean data matrix, shape $(N_x, N_t)$.
    dt : float
        Time step.
    n_fft : int, optional
        Block/FFT length. Default $2^{\lfloor \log_2 (N_t / 10) \rfloor}$.
    n_ovlp : int, optional
        Block overlap. Default ``n_fft // 2``.
    window : str or np.ndarray
        Window name (passed to ``scipy.signal.get_window``) or an array of
        length ``n_fft``.
    weight : np.ndarray, optional
        Spatial integration weights $\mathrm{diag}(\matr{W})$, shape
        $(N_x,)$. Uniform weights (no integration) by default.
    conf_level : float
        Target confidence level for the chi-squared-based interval `Lc`
        (Welch's method, nominally $2\,n_\mathrm{blk}$ degrees of freedom).

    Returns
    -------
    L : np.ndarray
        Modal energy spectrum, shape ``(n_freq, n_blks)``.
    Psi : np.ndarray
        Complex SPOD spatial modes, shape ``(n_freq, N_x, n_blks)``.
    f : np.ndarray
        Frequency vector, shape ``(n_freq,)``.
    Lc : np.ndarray
        Confidence bounds ``[lower, upper]`` for `L`, shape
        ``(n_freq, n_blks, 2)``: the chi-squared interval with
        $2\,n_\mathrm{blk}$ degrees of freedom (Towne et al. 2018).
    info : dict
        Effective ``n_fft``, ``n_ovlp``, ``n_blks``, ``n_freq`` and window used.

    References
    ----------
    Towne, Schmidt & Colonius (2018). Spectral proper orthogonal decomposition and
    its relationship to dynamic mode decomposition and resolvent analysis.
    *J. Fluid Mech.*, 847, 821–867.
    """
    N_x, N_t = Q.shape
    is_real  = np.isrealobj(Q)

    if n_fft is None:
        n_fft = int(2 ** np.floor(np.log2(N_t / 10)))
    if n_ovlp is None:
        n_ovlp = n_fft // 2
    if n_ovlp >= n_fft:
        raise ValueError('n_ovlp must be < n_fft.')

    if isinstance(window, str):
        win = get_window(window, n_fft)
    else:
        win = np.asarray(window, dtype=float)
        if win.size != n_fft:
            raise ValueError(f'window length ({win.size}) must equal n_fft ({n_fft}).')
    win_norm = 1.0 / win.mean()
    win_col  = win[:, None]

    W = np.ones(N_x) if weight is None else np.asarray(weight, dtype=float).ravel()
    if W.size != N_x:
        raise ValueError('weight must have length N_x.')

    n_step = n_fft - n_ovlp
    n_blks = int(np.floor((N_t - n_ovlp) / n_step))
    if n_blks < 2:
        raise ValueError(
            f'Too few blocks ({n_blks}). Reduce n_fft/n_ovlp or use more snapshots.')

    if is_real:
        n_freq = n_fft // 2 + 1
        f      = np.arange(n_freq) / (n_fft * dt)
    else:
        n_freq = n_fft
        f      = np.fft.fftfreq(n_fft, d=dt)

    Q_hat = np.zeros((n_freq, N_x, n_blks), dtype=complex)
    for b in range(n_blks):
        i0     = b * n_step
        Q_blk  = Q[:, i0:i0 + n_fft]
        Q_fft  = np.fft.fft(Q_blk * win_col.T, axis=1) * win_norm / n_fft
        if is_real:
            Q_hat[:, :, b] = Q_fft[:, :n_freq].T
        else:
            Q_hat[:, :, b] = Q_fft.T

    L   = np.zeros((n_freq, n_blks))
    Psi = np.zeros((n_freq, N_x, n_blks), dtype=complex)
    for k in range(n_freq):
        Qf         = Q_hat[k]
        M          = (Qf * W[:, None]).conj().T @ Qf / n_blks
        lam, Theta = np.linalg.eigh(M)
        idx        = lam.argsort()[::-1]
        lam        = np.abs(lam[idx])
        Theta      = Theta[:, idx]
        Psi[k]     = Qf @ Theta / (np.sqrt(lam) * np.sqrt(n_blks))
        if is_real and 0 < k < n_freq - 1:
            L[k] = 2.0 * lam
        else:
            L[k] = lam

    xi2_up = 2 * gammaincinv(n_blks, 1 - conf_level)
    xi2_lo = 2 * gammaincinv(n_blks, conf_level)
    Lc     = np.stack([L * 2 * n_blks / xi2_lo,
                       L * 2 * n_blks / xi2_up], axis=-1)
    info = dict(n_fft=n_fft, n_ovlp=n_ovlp, n_blks=n_blks,
                window=win, n_freq=n_freq)
    return L, Psi, f, Lc, info


def print_spod_towne_summary(info):
    """
    Pretty-print the block/frequency parameters of a `spod_towne` run.

    Parameters
    ----------
    info : dict
        The ``info`` dictionary returned by `spod_towne` (must contain
        ``n_fft``, ``n_ovlp``, ``n_blks`` and ``n_freq``).
    """
    print('SPOD (Towne / Welch) parameters')
    print('────────────────────────────────')
    print(f'  Snapshots per block (n_fft)  : {info["n_fft"]}')
    print(f'  Block overlap (n_ovlp)       : {info["n_ovlp"]}')
    print(f'  Number of blocks             : {info["n_blks"]}')
    print(f'  Resolved frequencies         : {info["n_freq"]}')

