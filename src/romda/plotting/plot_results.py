import matplotlib.pyplot as plt


def plot_train_dataset(clean_data, noisy_data, t, *split_times):



    # Visualize the training dataset
    fig = plt.figure(figsize=(12.5, 5), layout='tight')
    sfs = fig.subfigures(1, 2, width_ratios=[1.2, 1])

    axs = sfs[0].subplots(nrows=clean_data.shape[1], ncols=1, sharex='col', sharey='row')

    for axi, ax in enumerate(axs):
        ax.plot(t, clean_data[:, axi], c='k', lw=1.2, label='Truth')
        if noisy_data is not None:
            ax.plot(t, noisy_data[:, axi], c='r', lw=0.4, label='Noisy data')
        for kk, c, lbl in zip(range(1, len(split_times) + 2), ['C2', 'C1', 'C4', 'lightgray'],
                              ['Train', 'Validation', 'Test', 'Not used']):
            ax.axvspan(sum(split_times[:kk]), sum(split_times[:kk + 1]), facecolor=c, alpha=0.3,
                       label=lbl + ' data')
            ax.axvline(x=sum(split_times[:kk]), color='k', lw=0.5, dashes=[10, 5])
        if axi == 0:
            ax.legend(loc='lower center', ncol=10, bbox_to_anchor=(0.5, 1.0))
    axs[-1].set(xlabel='$t/T$', xlim=[0, t[-1]])

    ax = sfs[1].add_subplot(111, projection='3d')
    _nt = 1000
    ax.plot(clean_data[-_nt:, 0], clean_data[-_nt:, 1], clean_data[-_nt:, 2], '.-',
            c='k', ms=2, lw=.5, label='Truth')

    if noisy_data is not None:
        ax.plot(noisy_data[-_nt:, 0], noisy_data[-_nt:, 1], noisy_data[-_nt:, 2], '.',
                c='r', ms=1, label='Noisy data')
    ax.set(xlabel='$x$', ylabel='$y$', zlabel='$z$')
