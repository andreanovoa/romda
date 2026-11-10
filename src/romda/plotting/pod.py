
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from romda.models.data_driven.autoencoders import POD


def plot_modes(case: POD,
                num_modes: int = 2, save: bool = False,
                cmap: str = 'viridis', dim=None,
                n_col=4):
    """Plot the first ``num_modes`` spatial modes of a fitted POD, one figure per field component in ``dim``."""

    assert case.fitted, "POD must be fitted before plotting modes."

    Psi = case.Psi.copy()
    if Psi.ndim == 2:
        Psi = case._to_physical_grid(Psi)
    dim = np.arange(Psi.shape[0]) if dim is None else (
            [dim] if isinstance(dim, int) else dim)
    X1, X2 = case.domain_mesh

    num_modes = min(num_modes, Psi.shape[1])
    n_col = min(n_col, num_modes)
    n_row = int(np.ceil(num_modes / n_col))

    for jj, d in enumerate(dim):
        data = Psi[d]
        fig  = plt.figure(figsize=(n_col * 2, n_row), layout='constrained')
        axs  = fig.subplots(nrows=n_row, ncols=n_col,
                            sharex=True, sharey=True, squeeze=False).ravel()

        for kk, ax in zip(range(num_modes), axs):
            im = ax.pcolormesh(X1, X2, data[kk], cmap=cmap,
                                vmin=np.nanmin(data)*.8, vmax=np.nanmax(data)*.8,
                                rasterized=True)
            ax.set_title(f'mode {kk+1}', fontsize='xx-small')
            ax.set_aspect('equal')
            if kk >= num_modes - n_col:
                ax.set_xlabel('$x$')
            if kk % n_col == 0:
                ax.set_ylabel('$y$')
        fig.colorbar(im, ax=axs, shrink=0.25, aspect=20) #type: ignore

        if num_modes < n_col * n_row:
            for ax in axs[num_modes:]:
                ax.set_visible(False)

        if save:
            plt.savefig(f'modes_dim{jj}.png', dpi=300)



def plot_time_coefficients(case: POD,
                           Phi: np.ndarray = None,
                            num_modes: int = None):
    """Imshow of the temporal coefficients Phi (N_latent, N_t), default ``case.Phi``, split into rows."""
    if Phi is None:
        Phi = case.Phi.copy()
    if num_modes is not None:
        Phi = Phi[:num_modes, :]
    else:
        num_modes = Phi.shape[0]
    N_t    = Phi.shape[1]
    window = N_t if num_modes < 10 else (200 if num_modes < 50 else num_modes)
    nrows  = max(int(N_t // window), 1)
    slices = [Phi[:, i * window:(i + 1) * window] for i in range(nrows)]

    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(10, 1.5 * nrows),
                                layout='tight', squeeze=False)
    axs = axs.ravel()
    for i0, (ax, sl) in enumerate(zip(axs, slices)):
        im = ax.imshow(sl, cmap='viridis', vmin=Phi.min(), vmax=Phi.max(),
                        aspect='auto',
                        extent=[i0 * window, (i0 + 1) * window,
                                0, num_modes],
                        origin='lower')
    fig.colorbar(im, ax=axs[0], shrink=0.75, orientation='vertical')


def plot_spectrum(case: POD, max_mode: int | None = None):
    """Bar chart of the eigenvalue spectrum and the cumulative energy, zoomed up to ``max_mode`` if given."""
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    Lambda            = case.Sigma ** 2
    normalised_Lambda = Lambda / Lambda[0]
    cum_energy = np.cumsum(Lambda) / sum(Lambda)

    axs[0].bar(np.arange(case.N_latent) + 1, normalised_Lambda, color='C4')
    axs[0].set(xlabel='Mode $j$', title='$\\lambda_j / \\lambda_0$',
                xlim=[0, max(case.N_latent, 10)])

    axs[1].plot(np.arange(case.N_latent) + 1, cum_energy, 'o-', color='C4',
                label='$\\Sigma \\lambda_j / \\Sigma_k \\lambda_k$')

    if case._TKE is not None:

        axs[1].plot(np.arange(case.N_latent) + 1,
                    np.cumsum(Lambda) / 2 / case._TKE,
                    dashes=[10, 5], color='k', label='$\\Sigma \\lambda_j /$ 2TKE')

    axs[1].grid(visible=True, linestyle='--', alpha=0.5)
    axs[1].set(xlabel='# modes', title='Cumulative energy')
    axs[1].legend(ncol=1, bbox_to_anchor=[1, 1], loc='upper left')
    for ax in axs:
        ax.set(ylim=[0, 1.05], xlim=[-1, None])

    if max_mode is not None and max_mode < case.N_latent:
        ax0 = fig.add_axes((0.30, 0.50, 0.15, 0.35))
        ax0.bar(np.arange(max_mode) + 1, normalised_Lambda[:max_mode],
                color='C4')
        ax0.set_xlim(0, max_mode + 1)
        ax1 = fig.add_axes((0.72, 0.20, 0.15, 0.35))
        ax1.grid(visible=True, linestyle='--', alpha=0.5)
        ax1.plot(np.arange(max_mode) + 1,
                    cum_energy[:max_mode], color='C4')
        ax1.set_xlim(0, max_mode + 1)
    return fig




def plot_flows_rms(case: POD,
                    datasets: dict,
                    display_dims=None,
                    norm_flow=None,
                    norm_rms=None,
                    cmap_flow: str = 'viridis',
                    cmap_rms:  str = 'Reds',
                    save: bool = False,
                    display_sensors: bool = False):
    """Flow fields and RMS-error panels side by side, one row per entry of ``datasets`` (name -> data).

    Entries whose name contains 'error' or 'rms' use ``cmap_rms`` and ``norm_rms``; the rest share ``norm_flow``.
    """

    def _prep(d, target):
        if d.ndim > 3:
            d = d[..., -1]
        if d.shape != target:
            raise ValueError(
                f'Data shape {d.shape} does not match target {target}.')
        # Apply fluid mask if available
        if hasattr(case, 'fluid_mask_flat'):
            mask_2d = case.fluid_mask_flat.reshape(case.grid_shape[-2:])[np.newaxis, ...]
            mask_2d = np.broadcast_to(mask_2d, d.shape)
            return np.ma.masked_where(~mask_2d, d)
        return d

    ref_data = list(datasets.values())[0]
    if ref_data.ndim > 3:
        ref_data = ref_data[..., -1]
    display_dims = np.atleast_1d(np.arange(ref_data.shape[0]) if display_dims is None else display_dims)

    X1, X2 = case.domain_mesh

    idx, display_sensors = [], display_sensors
    if display_sensors and hasattr(case, 'sensor_locations'):
        sensor_locs = case.sensor_locations
        idx = case.sensor_locations[sensor_locs < len(X1.ravel())]

    if isinstance(cmap_flow, str):
        cmap_flow = plt.get_cmap(cmap_flow)
        cmap_flow.set_bad(color='lightgray')
    if isinstance(cmap_rms, str):
        cmap_rms = plt.get_cmap(cmap_rms)
        cmap_rms.set_bad(color='lightgray')

    _titles   = list(datasets)
    _datasets = [_prep(ds, ref_data.shape) for ds in datasets.values()]
    is_err    = ['error' in t.lower() or 'rms' in t.lower() for t in _titles]
    _cmaps    = [cmap_rms if e else cmap_flow for e in is_err]

    if norm_rms is None:
        rms_data = [d for d, e in zip(_datasets, is_err) if e]
        norm_rms = colors.Normalize(vmin=0., vmax=np.nanmax(rms_data) if rms_data else 1.)
    if norm_flow is None:
        flow_data = [d for d, e in zip(_datasets, is_err) if not e]
        norm_flow = [colors.Normalize(vmin=np.nanmin([y[r] for y in flow_data]),
                                      vmax=np.nanmax([y[r] for y in flow_data]))
                     for r in display_dims]

    nrows   = len(_datasets)
    ncols = len(display_dims)
    x0, x1, y0, y1 = case.domain
    aspect = abs(y1 - y0) / abs(x1 - x0) if x1 != x0 else 1
    figsize = (3 * ncols, 3 * aspect * nrows)

    fig, all_axs = plt.subplots(figsize=figsize, layout='constrained', nrows=nrows, ncols=ncols,
                                sharex=True, sharey=True, squeeze=False)

    for jj in range(ncols):

        axs = all_axs[:, jj]
        nf = norm_flow[jj] if isinstance(norm_flow, list) else norm_flow

        im_rms = im_flow = None
        for ax, ds, title, cm, e in zip(axs, _datasets, _titles, _cmaps, is_err):
            if e:
                im_rms = ax.pcolormesh(X1, X2, ds[display_dims[jj]], cmap=cm,  norm=norm_rms, rasterized=True)
            else:
                im_flow = ax.pcolormesh(X1, X2, ds[display_dims[jj]], cmap=cm,  norm=nf, rasterized=True)

            if jj == 0:
                ax.set_ylabel(title, fontsize='small')

            if display_sensors and len(idx):
                ax.scatter(X1.ravel()[idx], X2.ravel()[idx],
                            c=np.arange(len(idx)),
                            cmap='YlOrRd', edgecolors='k', s=12.25, lw=.5)
            ax.set_aspect('equal')
        for im in [im_rms, im_flow]:
            if im is not None:
                plt.colorbar(im, ax=axs, shrink=0.5, orientation='horizontal')
    if save:
        plt.savefig('rom_flows_rms.png', dpi=300)
    return fig
