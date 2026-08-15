import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scenario import helper

METHODS = ('semcom', 'baseline')
DATASETS = ('cifar10', 'tsrd')
DATASET_LABEL = {'cifar10': 'CIFAR-10', 'tsrd': 'TSRD'}
# the two panels sit side by side inside a single column
_PANEL_PT = 0.48 * 252.0


def _has_run(args, method, model):
    # a series earns a bar only if it was actually measured on some dataset
    return any(
        os.path.isfile(
            os.path.join(args.csv_dir, f'rq1_run_{method}_{ds}_{model}.csv'),
        )
        for ds in DATASETS
    )


def _plot_payload(args):
    # SemCom: one OFDM block of n_symbol tones; baseline: n_baseline_frame blocks.
    # Both backbones emit a 512-d embedding, so the two SemCom bars are equal;
    # showing them anyway keeps the colours in step with the other panel.
    rows = []
    for (model, method), label in helper.series.SERIES.items():
        if not _has_run(args, method, model):
            continue
        symbols = args.n_symbol
        if method == 'baseline':
            symbols *= args.n_baseline_frame
        rows.append({'series': label, 'complex_symbols': symbols})
    df = pd.DataFrame(rows)
    order = [s for s in helper.series.SERIES_ORDER if s in set(df['series'])]
    fig, ax = plt.subplots(figsize=helper.plotstyle.figsize(_PANEL_PT, 0.90))
    sns.barplot(
        data=df, x='series', y='complex_symbols', hue='series',
        hue_order=order, order=order, legend=False, ax=ax,
    )
    ax.set_xlabel('')
    ax.set_ylabel('complex symbols')
    # series are named in the shared legend; repeating them here would not fit
    ax.set_xticklabels([])
    fig.tight_layout(pad=0.15)
    out = os.path.join(args.figure_dir, 'rq2_payload.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'[save] {out}')
    print(df.to_string(index=False))


def _plot_accuracy(args):
    # accuracy at highest tx_power present, every backbone on one pair of axes
    rows = []
    for ds in DATASETS:
        for (model, method), label in helper.series.SERIES.items():
            path = os.path.join(
                args.csv_dir, f'rq1_run_{method}_{ds}_{model}.csv',
            )
            if not os.path.isfile(path):
                print(f'[skip] missing {path}')
                continue
            run = pd.read_csv(path)
            tx = int(run['tx_power'].max())
            acc = float(run.loc[run['tx_power'] == tx, 'ok'].mean())
            rows.append({
                'dataset': DATASET_LABEL[ds], 'series': label,
                'tx_power': tx, 'accuracy': acc,
            })
            print(f'[load] {path} tx_power={tx} acc={acc:.3f} -> {label}')
    assert rows, 'no rq1_run_*.csv found'
    table = pd.DataFrame(rows)
    order = [s for s in helper.series.SERIES_ORDER if s in set(table['series'])]
    csv_out = os.path.join(args.figure_dir, 'rq2_accuracy.csv')
    table.to_csv(csv_out, index=False)
    print(f'[save] {csv_out}')

    fig, ax = plt.subplots(figsize=helper.plotstyle.figsize(_PANEL_PT, 0.90))
    sns.barplot(
        data=table, x='dataset', y='accuracy', hue='series',
        hue_order=order, ax=ax,
    )
    ax.set_xlabel('')
    ax.set_ylabel('top-1 accuracy')
    ax.set_ylim(0.0, 1.05)
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    fig.tight_layout(pad=0.15)
    helper.series.save_legend(
        handles, labels, os.path.join(args.figure_dir, 'rq2_legend.pdf'), helper.plotstyle.COLUMN_PT,
    )
    out = os.path.join(args.figure_dir, 'rq2_accuracy.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'[save] {out}')


def main():
    args = helper.args.parse_args()
    helper.plotstyle.apply()
    _plot_payload(args)
    _plot_accuracy(args)


if __name__ == '__main__':
    main()
