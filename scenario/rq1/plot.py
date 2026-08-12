import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from scenario import helper

def main():
    args = helper.args.parse_args()
    ds = args.dataset
    measure_path = os.path.join(args.csv_dir, f'rq1_measure_{ds}.csv')
    measure = pd.read_csv(measure_path)

    # every backbone on one pair of axes, so --model no longer selects the figure
    rows = []
    for (model, method), label in helper.series.SERIES.items():
        path = os.path.join(args.csv_dir, f'rq1_run_{method}_{ds}_{model}.csv')
        if not os.path.isfile(path):
            print(f'[skip] missing {path}')
            continue
        run = pd.read_csv(path)
        acc = run.groupby('tx_power', as_index=False)['ok'].mean()
        acc = acc.rename(columns={'ok': 'accuracy'})
        acc['series'] = label
        rows.append(acc)
        print(f'[load] {path} -> {label}')
    assert rows, f'no rq1_run_*_{ds}_*.csv found'
    run = pd.concat(rows, ignore_index=True)

    df = run.merge(measure[['tx_power', 'snr_db']], on='tx_power', how='inner')
    df = df.sort_values(['series', 'snr_db'])
    order = [s for s in helper.series.SERIES_ORDER if s in set(df['series'])]

    helper.plotstyle.apply()
    # two panels across the full text block of a figure*
    width = 0.48 * helper.plotstyle.TEXT_PT
    fig, ax = plt.subplots(figsize=helper.plotstyle.figsize(width, 0.62))
    sns.lineplot(
        data=df, x='snr_db', y='accuracy', hue='series', hue_order=order,
        style='series', style_order=order, marker='o', ax=ax,
    )
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('top-1 accuracy')
    ax.set_ylim(-0.1, 1.1)
    handles, labels = ax.get_legend_handles_labels()
    ax.get_legend().remove()
    fig.tight_layout(pad=0.15)
    helper.series.save_legend(
        handles, labels, os.path.join(args.figure_dir, 'rq1_legend.pdf'), helper.plotstyle.TEXT_PT,
    )
    out = os.path.join(args.figure_dir, f'rq1_accuracy_vs_snr_{ds}.pdf')
    fig.savefig(out)
    plt.close(fig)
    print(f'[save] {out}')


if __name__ == '__main__':
    main()
