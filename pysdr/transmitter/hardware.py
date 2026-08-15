import numpy as np
try:
    import adi
except ImportError:
    # import warnings
    # warnings.warn('no plutosdr device found', UserWarning)
    pass


from pysdr import util

class Transmitter:

    def __init__(self, name, args):
        # save args
        self.args = args
        # initialize sdr
        sdr = adi.Pluto(name)
        # configure
        sdr.sample_rate           = args.sample_rate
        sdr.tx_rf_bandwidth       = args.sample_rate
        sdr.tx_lo                 = args.carrier_freq
        sdr.tx_hardwaregain_chan0 = args.tx_power
        sdr.tx_cyclic_buffer = True
        # save sdr
        self.sdr = sdr

    def prepare(self, s):
        # s: OFDM time block (torch or numpy) → numpy for Pluto
        if hasattr(s, 'detach'):
            s = s.detach().cpu().numpy()
        s = np.asarray(s, dtype=np.complex64).reshape(-1)
        # extract args
        args = self.args
        # extract preamble
        preamble = util.get_preamble()
        # create frame
        symbols = np.concatenate([preamble, s]).astype(np.complex64)
        # OFDM @ n_sps=1: skip RRC (it hurts SCS vs Pluto CFO); else pulse-shape
        if args.n_sps == 1:
            samples = symbols
        else:
            samples = util.apply_pulse_shaping(symbols, n_sps=args.n_sps)
        # peak-scale into Pluto 16-bit DAC (mean*2^14 clips OFDM PAPR≈9dB)
        peak = float(np.max(np.abs(samples))) + 1e-12
        samples = samples / peak * (2 ** 14)
        # save for transmit
        self.samples = samples.astype(np.complex64)

    def transmit(self):
        self.sdr.tx(self.samples)

    def stop(self):
        # stop transmission and cleanup
        self.sdr.tx_destroy_buffer()
