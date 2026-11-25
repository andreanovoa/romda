"""The shared figures of the tutorials (`romda.plotting.tutorials`) draw what they promise, headless."""
import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from romda.plotting import COLORS, plot_error_spread, plot_grouped_dots, plot_scores, space_time, use_tutorial_style  # noqa: E402


def test_style_is_sans_serif():
    use_tutorial_style()
    assert plt.rcParams['font.family'] == ['sans-serif']
    assert not plt.rcParams['axes.spines.top']


def test_space_time_symmetric_and_error_ranges():
    fig, ax = plt.subplots()
    field = np.array([[1., -3.], [2., 0.5]])
    mesh = space_time(ax, [0., 1.], [0., 1.], field)
    assert mesh.get_clim() == (-3., 3.)
    assert space_time(ax, [0., 1.], [0., 1.], np.abs(field), vmin=0).get_clim() == (0., 3.)
    plt.close(fig)


def test_error_spread_draws_two_lines():
    fig, ax = plt.subplots()
    plot_error_spread(ax, np.arange(3), np.ones(3), 0.5 * np.ones(3), label='filter')
    assert len(ax.lines) == 2 and ax.get_yscale() == 'log'
    plt.close(fig)


@pytest.mark.parametrize('band, n_lines, n_fills', [('iqr', 1, 1), ('range', 1, 1), ('points', 2, 0), (None, 1, 0)])
def test_scores_median_and_band(band, n_lines, n_fills):
    fig, ax = plt.subplots()
    scores = np.array([[1., 2., 3.], [4., 5., 6.]])
    plot_scores(ax, [0.1, 0.2], scores, band=band)
    assert len(ax.lines) == n_lines and len(ax.collections) == n_fills
    np.testing.assert_allclose(ax.lines[0].get_ydata(), [2., 5.])
    plt.close(fig)


def test_scores_rejects_an_unknown_band():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        plot_scores(ax, [0.], [[1.]], band='box')
    plt.close(fig)


def test_grouped_dots_one_median_bar_per_group():
    fig, ax = plt.subplots()
    plot_grouped_dots(ax, [1., 1.1], {'a': np.ones((2, 3)), 'b': 2 * np.ones((2, 3))},
                      colors=[COLORS['blue'], COLORS['vermillion']])
    bars = [line for line in ax.lines if line.get_marker() == '_']
    assert len(bars) == 2
    assert [t.get_text() for t in ax.get_xticklabels()] == ['1', '1.1']
    plt.close(fig)
