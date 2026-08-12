import numpy as np
import torch
try:
    import adi
except ImportError:
    pass

from pysdr import util

class Receiver:

    def __init__(self, name, args):
        # save args
        self.args = args
        # initialize sdr
        sdr = adi.Pluto(name)
        # configure
        sdr.rx_lo                   = args.carrier_freq
        sdr.sample_rate             = args.sample_rate
        # x8 headroom so cyclic TX frames fit comfortably in the capture
        sdr.rx_buffer_size          = (args.n_payload + args.n_preamble) * args.n_sps * 8
        sdr.rx_rf_bandwidth         = args.sample_rate
        sdr.gain_control_mode_chan0 = args.gain_control_mode
        if args.gain_control_mode == 'manual':
            sdr.rx_hardwaregain_chan0 = args.rx_gain
        # save sdr
        self.sdr = sdr

    def receive(self):
        # extract args
        args = self.args
        sdr = self.sdr
        # extract preamble
        preamble = util.get_preamble()
        # clear buffer
        for _ in range(args.n_clear_buffer):
            sdr.rx()
        # receive samples
        samples = sdr.rx()
        # normalize
        samples = samples / np.max(np.abs(samples))
        s_hat = util.ofdm_frame_sync(samples, preamble, args)
        return torch.as_tensor(s_hat, dtype=torch.complex64)
