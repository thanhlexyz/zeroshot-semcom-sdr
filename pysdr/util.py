import os

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def raised_cosine_taps(n_sps=16):
    # same RC taps used for TX pulse shaping / RX matched filter
    beta = 0.35
    Ts = n_sps
    t = np.arange(-50, 51)
    h = np.sinc(t / Ts) * np.cos(np.pi * beta * t / Ts) / (1 - (2 * beta * t / Ts) ** 2)
    h[np.isnan(h)] = 1.0
    return h.astype(np.float64)


def apply_pulse_shaping(symbols, n_sps=16):
    # Apply raised cosine pulse shaping
    h = raised_cosine_taps(n_sps)
    # upsample symbols
    upsampled = np.zeros(len(symbols) * n_sps, dtype=np.complex64)
    upsampled[::n_sps] = symbols
    # apply filter
    shaped = np.convolve(upsampled, h, mode='same')
    return shaped

def frame_sync(samples, preamble, args):
    # Frame synchronization using preamble detection with phase correction
    n_payload = args.n_payload
    # cross-correlate with known preamble
    corr = np.abs(np.correlate(samples, preamble, mode='full'))
    # avoid decoding incomplete frame at the end
    trim = n_payload + args.n_preamble
    if trim < len(corr):
        corr = corr[:-trim]
    peak_idx = np.argmax(corr)
    # find frame start
    frame_start = peak_idx - args.n_preamble + 1
    if frame_start < 0:
        frame_start = 0
    # extract detected preamble for verification
    detected_preamble_samples = samples[frame_start:frame_start + args.n_preamble]
    # convert to bits for comparison (demodulate detected preamble)
    expected_preamble_bits = ['0' if s >= 0 else '1' for s in preamble]
    detected_preamble_bits = ['0' if np.real(s) >= 0 else '1' for s in detected_preamble_samples]
    # calculate bit error rate for preamble
    matches = sum(1 for e, d in zip(expected_preamble_bits, detected_preamble_bits) if e == d)
    ber = 1 - (matches / len(expected_preamble_bits))
    # check for phase inversion (180° phase ambiguity in BPSK)
    phase_inverted = False
    if ber > 0.5:  # If more than half the bits are wrong, likely phase inversion
        inverted_matches = sum(1 for e, d in zip(expected_preamble_bits, detected_preamble_bits) if e != d)
        if inverted_matches == len(expected_preamble_bits):
            phase_inverted = True
    # extract data portion (skip preamble)
    data_start = frame_start + args.n_preamble
    data_end = data_start + n_payload
    if data_end <= len(samples):
        data_samples = samples[data_start:data_end]
    else:
        data_samples = samples[data_start:]
    # apply phase correction if needed
    if phase_inverted:
        data_samples = -data_samples
    return data_samples

def ofdm_frame_sync(samples, preamble, args):
    # OFDM RX sync: preamble correlation (+ MF/decimate only if n_sps>1)
    # Mueller-Muller is single-carrier and destroys OFDM
    from pysdr import ofdm  # local import avoids package cycle

    n_sps = args.n_sps
    n_payload = args.n_payload
    n_preamble = args.n_preamble
    pref_sym = np.asarray(preamble, dtype=np.complex64)
    samples = np.asarray(samples, dtype=np.complex64)

    if n_sps == 1:
        # baseband symbols on the wire (TX skips RRC at n_sps=1)
        # CP-based timing (Van de Beek): |Σ r[i] conj(r[i+n_fft])| summed
        # over a CP-length window peaks at the CP start. Data-independent
        # (summands are |z|² terms, add coherently) unlike the 11-tap Barker
        # correlation, which false-peaks on Gaussian payload ~1% of frames.
        # Fold over the frame period so all TX repeats vote on one timing.
        n_fft, n_cp = args.n_fft, args.n_cp
        frame_len = n_preamble + n_payload
        c = samples[:-n_fft] * np.conj(samples[n_fft:])
        csum = np.cumsum(np.concatenate([[0.0 + 0.0j], c]))
        metric = np.abs(csum[n_cp:] - csum[:-n_cp])
        n_rep = len(metric) // frame_len
        if n_rep >= 2:
            fold = metric[:n_rep * frame_len].reshape(n_rep, frame_len).sum(0)
            t_cp = int(np.argmax(fold))
        else:
            t_cp = int(np.argmax(metric))
        # window biased TIMING_BACKOFF early: starts before the CP so timing
        # jitter is a pure circular shift the OFDM pilot EQ removes exactly
        # (a late window would eat the next frame's preamble → ICI)
        start = t_cp - ofdm.TIMING_BACKOFF
        while start < 0:
            start += frame_len
        while start + n_payload > len(samples) and start - frame_len >= 0:
            start -= frame_len
        end = start + n_payload
        if end > len(samples):
            samples = np.pad(samples, (0, end - len(samples)))
        return samples[start:end].astype(np.complex64)
    else:
        h = raised_cosine_taps(n_sps)
        filtered = np.convolve(samples, h, mode='same').astype(np.complex64)
        pref = apply_pulse_shaping(pref_sym, n_sps=n_sps)
        corr = np.abs(np.correlate(filtered, pref, mode='full'))
        peak = int(np.argmax(corr))
        start = peak - (len(pref) - 1)
        if start < 0:
            start = 0
        best_phase, best_score = 0, -1.0
        for phase in range(n_sps):
            idxs = start + phase + np.arange(n_preamble) * n_sps
            if idxs[-1] >= len(filtered):
                continue
            score = float(np.abs(np.vdot(filtered[idxs], pref_sym)))
            if score > best_score:
                best_score, best_phase = score, phase
        n_sym = n_preamble + n_payload
        idxs = start + best_phase + np.arange(n_sym) * n_sps
        if idxs[-1] >= len(filtered):
            filtered = np.pad(filtered, (0, idxs[-1] + 1 - len(filtered)))
        symbols = filtered[idxs]

    detected = symbols[:n_preamble]
    # Barker 180° ambiguity
    if np.sum(np.real(detected * np.conj(pref_sym)) > 0) < n_preamble / 2:
        symbols = -symbols
    # remove common phase using preamble (before residual-h on data)
    phi = float(np.angle(np.vdot(symbols[:n_preamble], pref_sym)))
    symbols = symbols * np.exp(-1j * phi)
    return symbols[n_preamble:n_preamble + n_payload].astype(np.complex64)


def mueller_clock_recovery(samples, args):
    # extract args
    n_sps = args.n_sps
    # Mueller and Muller clock recovery
    mu = 0
    out = np.zeros(len(samples) + 10, dtype=np.complex64)
    out_rail = np.zeros(len(samples) + 10, dtype=np.complex64)
    i_in = 0
    i_out = 2
    # more conservative loop condition to preserve more symbols
    while i_out < len(out) - 1 and i_in < len(samples) - n_sps:
        out[i_out] = samples[i_in]
        out_rail[i_out] = (np.real(out[i_out]) > 0) + 1j*(np.imag(out[i_out]) > 0)
        if i_out >= 2:
            x = (out_rail[i_out] - out_rail[i_out-2]) * np.conj(out[i_out-1])
            y = (out[i_out] - out[i_out-2]) * np.conj(out_rail[i_out-1])
            mm_val = np.real(y - x)
            mu += n_sps + 0.3 * mm_val
        else:
            mu += n_sps  # Initial advance without correction
        i_in += int(np.floor(mu))
        mu = mu - np.floor(mu)
        i_out += 1
    result = out[2:i_out]
    # print(f'[mueller_clock_recovery] {len(samples)=} {len(result)=} (reduction factor={len(samples)/len(result):.2f})')
    return result

def get_preamble():
    return np.array([1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1])  # 11-bit Barker code

def plot_constellation(symbols, name, args):
    if plt is None:
        return
    plt.scatter(np.real(symbols), np.imag(symbols), color='blue')
    plt.axhline(0, color='gray', lw=0.5)
    plt.axvline(0, color='gray', lw=0.5)
    path = os.path.join(args.figure_dir, f'{name}_constellation.pdf')
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.tight_layout()
    plt.savefig(path)
    plt.cla()
    plt.clf()
