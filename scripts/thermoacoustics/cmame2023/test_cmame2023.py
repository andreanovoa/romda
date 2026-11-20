"""Tests of the reproduction of Nóvoa, Racca & Magri (2023, CMAME), with synthetic records: the training members of
the network, the selection of the minimum of the errors, Table 2 and the figures of the sweeps (replicate.py).

    pytest scripts/thermoacoustics/cmame2023
"""

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
import replicate as rep

GAMMAS = (0., 1.75, 2.75, 3.5)
LS = (10, 60, 100)


def record(L, gamma, best=(60, 2.75)):
    """Synthetic scores whose sum of errors is smallest at `best`, at the end of the assimilation and after it."""
    distance = abs(L - best[0]) / 100 + abs(gamma - best[1])
    out = dict(L=L, augment=True, regularization_factor=gamma, rejected=0, R_true=0.25, R_biased_pre=1.2,
               R_unbiased_pre=1.2, mae_t=[0.01, 0.03], mae_biased=[0.3, 0.1], mae_unbiased=[0.2, 0.02])
    for window in ('DA', 'post'):
        out[f'R_biased_{window}'], out[f'R_unbiased_{window}'] = 0.2 + distance, 0.02 + 0.1 * distance
    for name, value in (('beta', 4.2), ('tau', 1.4e-3)):
        out[f'{name}_end'], out[f'std_{name}_end'] = value, 0.05 * value
    return out


@pytest.fixture
def table():
    return {(L, True, gamma): record(L, gamma) for L in LS for gamma in GAMMAS}


def test_single_training_member_takes_the_prior():
    """With L = 1 the network is trained on the model at its prior parameters, as in the paper; with more
    members they are drawn from the ranges of the configuration."""
    assert rep.training_ranges(rep.CASES['vdp'], 10) == dict(L=10)
    single = rep.training_ranges(rep.CASES['vdp'], 1)
    assert single['L'] == 1 and single['std_phi'] == 0.
    assert single['std_alpha'] == dict(beta=[70., 70.], zeta=[60., 60.], kappa=[4., 4.])


def test_configurations_hold_the_settings_of_the_paper():
    """Tables 3 and 4 of the paper: the truth, the ensemble and the network of each case."""
    vdp, rijke = rep.CASES['vdp'], rep.CASES['rijke']
    assert vdp['truth']['params'] | dict(dt=1e-4) == dict(dt=1e-4, law='tan', beta=75., zeta=55., kappa=3.4)
    assert (vdp['da']['m'], vdp['truth']['Nt_obs'], vdp['bias']['N_units'], vdp['bias']['upsample']) == (10, 30, 100, 5)
    assert (rijke['truth']['params']['beta'], rijke['truth']['params']['tau']) == (4.2, 1.4e-3)
    assert (rijke['da']['m'], rijke['truth']['Nt_obs'], rijke['bias']['N_units'], rijke['bias']['upsample']) == (50, 20, 500, 2)
    # Van der Pol: the ensemble is drawn about the prior model where the clock of the experiment starts
    assert vdp['da']['spinup_cr'] * 0.04 == pytest.approx(vdp['truth']['t_min'])
    assert 'spinup_cr' not in rijke['da']       # Rijke: the members are forecast apart before the first datum
    for config in (vdp, rijke):
        assert config['da']['method'] == 'rBA_EnKF' and config['bias']['tikh_range'] == [1e-16]


def test_minimum_of_the_sum_of_the_errors(table):
    L, gamma, run = rep.minimum(table, 'post')
    assert (L, gamma) == (60, 2.75) and run['R_biased_post'] == pytest.approx(0.2)
    table[(10, True, 3.5)]['R_unbiased_DA'] = -2.              # another minimum at the end of the assimilation
    assert rep.minimum(table, 'DA')[:2] == (10, 3.5) and rep.minimum(table, 'post')[:2] == (60, 2.75)


def test_table_2_sets_the_runs_beside_the_paper(table):
    text = rep.table_2(dict(linear=table, nonlinear=table))
    rows = text.splitlines()
    assert len(rows) == 2 + 3 * 2
    assert rows[2].startswith('| linear | paper | 0.2623 | 10, 3.50 | 0.1761 | 0.0244 | 100, 1.75 | 0.1817 | 0.0157 |')
    assert rows[3].startswith('| linear | romda | 0.2500 | 60, 2.75 | 0.2000 | 0.0200 | 60, 2.75 |')
    assert '| nonlinear | romda at the minima of the paper | | 60, 2.75 | 0.2000 | 0.0200 |' in rows[7]
    del table[(100, True, 1.75)]                              # a minimum of the paper that the sweep does not hold
    assert '100, 1.75 | not run | not run |' in rep.table_2(dict(linear=table)).splitlines()[4]


def test_figures_of_the_sweeps_are_drawn_from_the_records(table):
    plt.close('all')
    rep.plot_contours(dict(linear=table, nonlinear=table))
    rep.plot_gamma_sweep('rijke', {'linear bias': {g: table[(60, True, g)] for g in GAMMAS}}, 'fig11_gamma')
    sets = {}
    for L, augment in rep.FIG6_SETS:
        for gamma in rep.FIG6_GAMMAS:
            sets[(L, augment, gamma)] = dict(record(L, gamma), R_biased_post_members=0.3, R_biased_post_std=0.02,
                                             R_unbiased_post_members=0.03, R_unbiased_post_std=0.002)
    rep.plot_training_sets(sets)
    assert [plt.figure(num).get_label() for num in plt.get_fignums()] == ['fig10_L_gamma', 'fig11_gamma',
                                                                         'fig06_training_sets']
    star = [line for line in plt.figure('fig10_L_gamma').axes[0].lines if line.get_marker() == '*'][0]
    assert (star.get_xdata()[0], star.get_ydata()[0]) == (2.75, 60)
    plt.close('all')
