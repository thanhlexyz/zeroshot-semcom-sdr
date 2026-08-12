import torch


# sync window starts this many samples early (inside the CP)
TIMING_BACKOFF = 16
# CP samples to skip in the CFO estimator
CFO_SKIP = TIMING_BACKOFF + 8


# Pluto corrupts bins near DC (LO leakage / IQ offset) and the fs/2 band
# edge (analog filter rolloff); keep data tones off those bins
GUARD_BINS = 4


def _bad_bins(n_fft):
    centers = (0, n_fft // 2)
    return {(c + d) % n_fft
            for c in centers for d in range(-GUARD_BINS, GUARD_BINS + 1)}


def pilot_indices(n_fft, pilot_spacing):
    return torch.arange(0, n_fft, pilot_spacing, dtype=torch.long)


def data_indices(n_fft, pilot_spacing, n_symbol):
    skip = set(pilot_indices(n_fft, pilot_spacing).tolist()) | _bad_bins(n_fft)
    idx = [i for i in range(n_fft) if i not in skip]
    assert len(idx) >= n_symbol, f'need {n_symbol} data bins, have {len(idx)}'
    return torch.tensor(idx[:n_symbol], dtype=torch.long)


def known_pilots(n_pilot):
    # fixed deterministic unit BPSK pilots
    g = torch.Generator().manual_seed(0)
    bits = torch.randint(0, 2, (n_pilot,), generator=g)
    return (2 * bits - 1).to(torch.complex64)


def modulate(s, args):
    # radio path stays on CPU
    s = torch.as_tensor(s, dtype=torch.complex64).cpu()
    if s.ndim == 2:
        # stacked blocks → one contiguous multi-block air payload
        return torch.cat([modulate(block, args) for block in s])
    s = s.reshape(-1)
    n_fft, n_cp = args.n_fft, args.n_cp
    pilot_spacing = args.pilot_spacing
    p_idx = pilot_indices(n_fft, pilot_spacing)
    d_idx = data_indices(n_fft, pilot_spacing, s.numel())
    grid = torch.zeros(n_fft, dtype=torch.complex64)
    grid[p_idx] = known_pilots(p_idx.numel())
    grid[d_idx] = s
    # IFFT then cyclic prefix
    time = torch.fft.ifft(grid, n=n_fft)
    block = torch.cat([time[-n_cp:], time])
    # unit mean power
    block = block / (torch.sqrt(torch.mean(torch.abs(block) ** 2)) + 1e-12)
    return block.to(torch.complex64)


def _trim_block(time_block, n_fft, n_cp):
    time_block = torch.as_tensor(time_block, dtype=torch.complex64).reshape(-1).cpu()
    need = n_fft + n_cp
    if time_block.numel() < need:
        time_block = torch.nn.functional.pad(time_block, (0, need - time_block.numel()))
    elif time_block.numel() > need:
        time_block = time_block[:need]
    return time_block


def fft_grid(time_block, n_fft, n_cp):
    time_block = _trim_block(time_block, n_fft, n_cp)
    time = time_block[n_cp:n_cp + n_fft]
    return torch.fft.fft(time, n=n_fft).to(torch.complex64)


def _natural_freq(bins, n_fft):
    bins = torch.as_tensor(bins)
    return torch.where(bins >= n_fft // 2, bins - n_fft, bins)


def _unwrap(phase):
    # 1-D phase unwrap
    dp = phase[1:] - phase[:-1]
    dp = torch.remainder(dp + torch.pi, 2 * torch.pi) - torch.pi
    return torch.cat([phase[:1], phase[:1] + torch.cumsum(dp, dim=0)])


def _lin_interp(x, xp, fp):
    # 1-D linear interpolation; xp must be sorted ascending
    inds = torch.searchsorted(xp, x, right=True).clamp(1, xp.numel() - 1)
    x0, x1 = xp[inds - 1], xp[inds]
    y0, y1 = fp[inds - 1], fp[inds]
    t = (x - x0) / (x1 - x0 + 1e-12)
    return y0 + t * (y1 - y0)


def _interp_channel(h_pilot, p_idx, n_fft):
    # linear interp of complex channel in natural frequency order
    nat_p = _natural_freq(p_idx, n_fft)
    order = torch.argsort(nat_p)
    x = nat_p[order].to(torch.float32)
    h_p = h_pilot[order]
    phase = _unwrap(torch.angle(h_p))
    a = torch.stack([x, torch.ones_like(x)], dim=1)
    slope = torch.linalg.lstsq(a, phase.unsqueeze(1)).solution[0, 0]
    h_flat = h_p * torch.exp(-1j * slope * x)
    nat_all = _natural_freq(torch.arange(n_fft), n_fft).to(torch.float32)
    re = _lin_interp(nat_all, x, h_flat.real)
    im = _lin_interp(nat_all, x, h_flat.imag)
    return ((re + 1j * im) * torch.exp(1j * slope * nat_all)).to(torch.complex64)


def equalize(y, n_fft, pilot_spacing):
    p_idx = pilot_indices(n_fft, pilot_spacing)
    x_p = known_pilots(p_idx.numel())
    h_p = y[p_idx] / (x_p + 1e-12)
    h = _interp_channel(h_p, p_idx, n_fft)
    return (y / (h + 1e-12)).to(torch.complex64)


def extract_data(y_eq, n_fft, pilot_spacing, n_symbol):
    d_idx = data_indices(n_fft, pilot_spacing, n_symbol)
    return y_eq[d_idx].to(torch.complex64)


def integer_cfo_shift(y, n_fft, pilot_spacing, max_shift=8):
    # differential pilot correlation over candidate integer bin shifts
    p_idx = pilot_indices(n_fft, pilot_spacing)
    x_p = known_pilots(p_idx.numel())
    best_shift, best_score = 0, -1.0
    for shift in range(-max_shift, max_shift + 1):
        v = y[(p_idx + shift) % n_fft] * torch.conj(x_p)
        score = float(torch.abs(torch.sum(v[1:] * torch.conj(v[:-1]))))
        if score > best_score:
            best_score, best_shift = score, shift
    return best_shift


def estimate_cfo_cp(time_block, n_fft, n_cp, skip=0):
    # CP correlation → rad/sample
    # (elementwise sum, not torch.vdot: complex vdot silently returns 0
    #  on Pi 4 aarch64 torch builds)
    time_block = _trim_block(time_block, n_fft, n_cp)
    corr = torch.sum(
        time_block[skip:n_cp].conj()
        * time_block[n_fft + skip:n_fft + n_cp]
    )
    return float(torch.angle(corr) / n_fft)


def correct_cfo(time_block, cfo_rad):
    n = torch.arange(time_block.numel(), dtype=torch.float64)
    return (time_block * torch.exp(-1j * cfo_rad * n)).to(torch.complex64)


def demodulate(z_hat, args):
    # fractional CFO (CP) → FFT → integer CFO → pilot EQ → data
    z_hat = torch.as_tensor(z_hat, dtype=torch.complex64).reshape(-1).cpu()
    n_block = z_hat.numel() // (args.n_fft + args.n_cp)
    if n_block > 1:
        # multi-block air payload → (n_block, n_symbol); each block carries
        # its own CP and pilots, so per-block CFO/EQ works unchanged
        blocks = z_hat.reshape(n_block, args.n_fft + args.n_cp)
        return torch.stack([demodulate(block, args) for block in blocks])
    time_block = _trim_block(z_hat, args.n_fft, args.n_cp)
    cfo = estimate_cfo_cp(time_block, args.n_fft, args.n_cp, skip=CFO_SKIP)
    time_block = correct_cfo(time_block, cfo)
    y = fft_grid(time_block, args.n_fft, args.n_cp)
    shift = integer_cfo_shift(y, args.n_fft, args.pilot_spacing)
    if shift:
        y = torch.roll(y, -shift)
    y = equalize(y, args.n_fft, args.pilot_spacing)
    return extract_data(y, args.n_fft, args.pilot_spacing, args.n_symbol)
