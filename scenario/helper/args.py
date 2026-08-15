import argparse
import random
import torch
import os


def maybe_mkdir(args):
    for d in [args.data_dir, args.model_dir, args.csv_dir, args.figure_dir]:
        os.makedirs(d, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'tsrd'])
    parser.add_argument('--model', type=str, default='clip',
                        choices=['clip', 'mobileclip'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--data_dir', type=str, default='data/data')
    parser.add_argument('--model_dir', type=str, default='data/model')
    parser.add_argument('--csv_dir', type=str, default='data/csv')
    parser.add_argument('--figure_dir', type=str, default='data/figure')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--snr_db', type=float, default=40.0)
    parser.add_argument('--n_fft', type=int, default=512)
    parser.add_argument('--n_cp', type=int, default=64)
    parser.add_argument('--pilot_spacing', type=int, default=4)
    parser.add_argument('--n_symbol', type=int, default=256)

    parser.add_argument('--transceiver', type=str, default='software',
                        choices=['software', 'hardware'])
    parser.add_argument('--tx_uri', type=str, default='ip:192.168.3.1')
    parser.add_argument('--rx_uri', type=str, default='ip:192.168.2.1')
    parser.add_argument('--sample_rate', type=int, default=int(2e6))
    parser.add_argument('--carrier_freq', type=int, default=int(915e6))
    parser.add_argument('--tx_power', type=float, default=-20.0)
    parser.add_argument('--rx_gain', type=float, default=30.0)
    parser.add_argument('--n_clear_buffer', type=int, default=5)
    parser.add_argument('--gain_control_mode', type=str, default='manual',
                        choices=['fast_attack', 'slow_attack', 'manual'])
    parser.add_argument('--n_preamble', type=int, default=11)
    parser.add_argument('--n_sps', type=int, default=1)
    parser.add_argument('--n_measure', type=int, default=200)
    parser.add_argument('--n_per_class', type=int, default=10)
    parser.add_argument('--method', type=str, default='semcom',
                        choices=['semcom', 'baseline'])
    # baseline: OFDM blocks per air frame (one preamble, one TX/RX)
    parser.add_argument('--n_baseline_frame', type=int, default=9)

    args = parser.parse_args()
    maybe_mkdir(args)
    # n_symbol: complex latent symbols (CLIP half-dim).
    # n_payload: OFDM time samples.
    args.n_payload = args.n_fft + args.n_cp
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    return args
