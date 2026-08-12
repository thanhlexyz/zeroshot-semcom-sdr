import torch
import numpy as np

SYNC = b'\xad\x75'
# sync(2) + length(2) + payload + crc(2)
FRAME_OVERHEAD = 6
# baseline PHY: Gray 16-QAM on the same n_symbol OFDM data tones as SemCom
BITS_PER_TONE = 4
# square 16-QAM: Gray code 00,01,11,10 maps to PAM levels -3,-1,+1,+3
_GRAY_TO_LEVEL = np.array([-3, -1, 3, 1], dtype=np.float32)
_LEVELS = np.array([-3, -1, 1, 3], dtype=np.float32)
_LEVEL_TO_GRAY = np.array([0b00, 0b01, 0b11, 0b10], dtype=np.uint8)
_SCALE = np.float32(np.sqrt(10.0))  # E[|s|^2] = 10 before scaling

# CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF)
_CRC_TABLE = []
for _b in range(256):
    _c = _b << 8
    for _ in range(8):
        _c = ((_c << 1) ^ 0x1021) if (_c & 0x8000) else (_c << 1)
    _CRC_TABLE.append(_c & 0xFFFF)


def crc16(data):
    crc = 0xFFFF
    for byte in data:
        crc = ((crc << 8) & 0xFFFF) ^ _CRC_TABLE[((crc >> 8) ^ byte) & 0xFF]
    return crc


def payload_budget(n_symbol, n_frame):
    # n_frame OFDM blocks of 16-QAM tones, minus framing
    return n_frame * n_symbol * BITS_PER_TONE // 8 - FRAME_OVERHEAD


def pack_frame(payload, length):
    head = SYNC + int(length).to_bytes(2, 'big')
    body = head[2:] + payload
    return head + payload + crc16(body).to_bytes(2, 'big')


def unpack_frame(frame):
    # return (payload, length) or None on sync/CRC fail
    if len(frame) < FRAME_OVERHEAD:
        return None
    sync, length_b = frame[:2], frame[2:4]
    payload_len = len(frame) - FRAME_OVERHEAD
    payload = frame[4:4 + payload_len]
    crc_rx = int.from_bytes(frame[-2:], 'big')
    if sync != SYNC:
        return None
    if crc_rx != crc16(length_b + payload):
        return None
    length = int.from_bytes(length_b, 'big')
    return payload, length


def modulate(payload, length, n_symbol, n_frame):
    # framed bits → (n_frame, n_symbol) blocks of 16-QAM tones
    budget = payload_budget(n_symbol, n_frame)
    assert len(payload) == budget, f'{len(payload)=} {budget=}'
    frame = pack_frame(payload, length)
    bits = np.unpackbits(np.frombuffer(frame, dtype=np.uint8))
    need = n_frame * n_symbol * BITS_PER_TONE
    assert bits.size == need, f'{bits.size=} {need=}'
    bits = bits.reshape(-1, BITS_PER_TONE)
    i_gray = (bits[:, 0] << 1) | bits[:, 1]
    q_gray = (bits[:, 2] << 1) | bits[:, 3]
    s = (_GRAY_TO_LEVEL[i_gray] + 1j * _GRAY_TO_LEVEL[q_gray]) / _SCALE
    return torch.from_numpy(s.astype(np.complex64)).reshape(n_frame, n_symbol)


def demodulate(tones, n_symbol, n_frame):
    # hard-decide 16-QAM → framed bytes; None on sync/CRC fail
    tones = torch.as_tensor(tones, dtype=torch.complex64).reshape(-1).cpu()
    total = n_frame * n_symbol
    if tones.numel() < total:
        tones = torch.nn.functional.pad(tones, (0, total - tones.numel()))
    tones = tones[:total].numpy()
    i_gray = _LEVEL_TO_GRAY[
        np.argmin(np.abs(tones.real[:, None] - _LEVELS[None, :] / _SCALE), axis=1)]
    q_gray = _LEVEL_TO_GRAY[
        np.argmin(np.abs(tones.imag[:, None] - _LEVELS[None, :] / _SCALE), axis=1)]
    bits = np.empty((total, BITS_PER_TONE), dtype=np.uint8)
    bits[:, 0] = (i_gray >> 1) & 1
    bits[:, 1] = i_gray & 1
    bits[:, 2] = (q_gray >> 1) & 1
    bits[:, 3] = q_gray & 1
    data = np.packbits(bits.reshape(-1)).reshape(n_frame, -1)
    # RX sync locks onto any block boundary, so blocks arrive as an unknown
    # cyclic rotation; the rotation whose SYNC marker and CRC pass is the
    # right one (false accept ≈ n_frame / 2^16)
    for shift in range(n_frame):
        frame = np.roll(data, -shift, axis=0).tobytes()
        out = unpack_frame(frame)
        if out is not None:
            return out
    return None, None
