import os
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scenario import helper
from semcom.dataset import cifar10, tsrd

# One row per dataset, three example classes each (first N_COLS labels in
# each dataset's own class order).
N_COLS = 3


def _first_per_class(dataset, n_class):
    # first test image of each of the first n_class classes (deterministic)
    by_class = {}
    for i in range(len(dataset)):
        y = int(dataset.targets[i])
        if y not in by_class and y < n_class:
            by_class[y] = i
        if len(by_class) == n_class:
            break
    return [by_class[c] for c in range(n_class)]


def main():
    args = helper.args.parse_args()

    tsrd_ds = tsrd.create_pil(args)
    cifar_ds = cifar10.create_pil(args)
    tsrd_labels = tsrd.get_labels()
    cifar_labels = cifar10.get_labels()

    rows = [
        ('TSRD', tsrd_ds, tsrd_labels, _first_per_class(tsrd_ds, N_COLS)),
        ('CIFAR-10', cifar_ds, cifar_labels, _first_per_class(cifar_ds, N_COLS)),
    ]

    helper.plotstyle.apply()
    # design width = one column, so \includegraphics{width=\linewidth} places
    # this at 1:1 scale in a single-column figure and the fonts below come
    # out at their literal point size, matching the caption/footnote text
    # around them (see plotstyle.py's FONT_PT comment).
    width = helper.plotstyle.COLUMN_PT
    fig, axes = plt.subplots(
        len(rows), N_COLS, figsize=helper.plotstyle.figsize(width, 0.62),
    )
    for row, (name, dataset, labels, indices) in enumerate(rows):
        for col, idx in enumerate(indices):
            image, y = dataset[idx]
            ax = axes[row, col]
            ax.imshow(image)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            # narrow columns can't fit long labels on one line at readable size
            title = textwrap.fill(labels[y], width=11)
            ax.set_title(title, fontsize=helper.plotstyle.FONT_PT, pad=2, linespacing=1.1)
            if col == 0:
                ax.set_ylabel(name, fontsize=helper.plotstyle.FONT_PT)

    fig.subplots_adjust(left=0.14, right=0.99, top=0.83, bottom=0.02, hspace=0.75, wspace=0.15)
    out = os.path.join(args.figure_dir, 'dataset_examples.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'[save] {out}')


if __name__ == '__main__':
    main()
