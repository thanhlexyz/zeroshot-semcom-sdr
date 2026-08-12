import os

import pandas as pd

from scenario import helper


def main():
    args = helper.args.parse_args()
    ds, model = args.dataset, args.model
    path = os.path.join(args.csv_dir, f'rq3_runtime_{ds}_{model}.csv')
    run = pd.read_csv(path)
    print(f'[load] {path}')

    order = [m for m in ('semcom', 'baseline') if m in set(run['method'])]
    table = (
        run.groupby('method', as_index=False)[
            ['t_encode', 't_decode', 't_total']
        ]
        .mean()
        .set_index('method')
        .reindex(order)
        .reset_index()
        .rename(columns={
            't_encode': 'encode_s',
            't_decode': 'decode_s',
            't_total': 'total_s',
        })
    )
    print(table.to_string(index=False))

    out = os.path.join(args.figure_dir, f'rq3_table_{ds}_{model}.csv')
    table.to_csv(out, index=False)
    print(f'[save] {out}')


if __name__ == '__main__':
    main()
