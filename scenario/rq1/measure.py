import math
import os

import torch

from scenario import helper
from scenario.rq1 import TX_POWERS
import pysdr


def _random_data(n_symbol):
    # unit-mean-power complex Gaussian data tones
    s = torch.randn(n_symbol, dtype=torch.complex64)
    s = s / (torch.sqrt(torch.mean(torch.abs(s) ** 2)) + 1e-12)
    return s


def _frame_snr_db(s, s_hat):
    # wipe common phase, then SNR from residual vs unit-power tones
    s = torch.as_tensor(s, dtype=torch.complex64).reshape(-1)
    s_hat = torch.as_tensor(s_hat, dtype=torch.complex64).reshape(-1)
    s = s / (torch.sqrt(torch.mean(torch.abs(s) ** 2)) + 1e-12)
    s_hat = s_hat / (torch.sqrt(torch.mean(torch.abs(s_hat) ** 2)) + 1e-12)
    phi = torch.angle(torch.sum(torch.conj(s) * s_hat))
    s_hat = s_hat * torch.exp(-1j * phi)
    noise_power = float(torch.mean(torch.abs(s_hat - s) ** 2))
    return 10.0 * math.log10(1.0 / (noise_power + 1e-20))


def main():
    args = helper.args.parse_args()
    args.transceiver = 'hardware'

    csv_path = os.path.join(args.csv_dir, f'rq1_measure_{args.dataset}.csv')
    pbar = helper.progress_bar.create(
        total=len(TX_POWERS), unit='step',
        desc=f'rq1.measure.{args.dataset}', csv_path=csv_path,
    )
    transmitter, receiver = pysdr.create(args)
    try:
        for tx_power in TX_POWERS:
            transmitter.sdr.tx_hardwaregain_chan0 = float(tx_power)
            snrs = []
            for _ in range(args.n_measure):
                transmitter.stop()
                s = _random_data(args.n_symbol)
                transmitter.prepare(pysdr.ofdm.modulate(s, args))
                transmitter.transmit()
                z_hat = receiver.receive()
                transmitter.stop()
                s_hat = pysdr.ofdm.demodulate(z_hat, args)
                snrs.append(_frame_snr_db(s, s_hat))
            snr_db = float(sum(snrs) / len(snrs))
            pbar.step(tx_power=tx_power, snr_db=snr_db)
    finally:
        transmitter.stop()
    pbar.to_csv()
    print(f'[save] {csv_path}')


if __name__ == '__main__':
    main()
