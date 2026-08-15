import io

from PIL import Image

# Resolution ladder tried in order. Below roughly 290 B no JPEG exists at any
# resolution at 224², because markers + tables dominate; CIFAR (32²) can go
# smaller. When even quality 1 overflows, downscale and retry.
JPEG_SIDES = (224, 192, 160, 128, 96, 64, 48, 32, 16, 8)


def _encode_mode(image, budget, mode):
    # Try ladder for one color mode. Returns tuple or None.
    max_side = max(image.size)
    src = image.convert(mode)
    for side in JPEG_SIDES:
        if side > max_side:
            continue
        img = src if src.size == (side, side) else src.resize(
            (side, side), Image.LANCZOS)
        lo, hi, best, best_q = 1, 95, None, 1
        while lo <= hi:
            q = (lo + hi) // 2
            buf = io.BytesIO()
            img.save(buf, format='JPEG', quality=q, optimize=True)
            data = buf.getvalue()
            if len(data) <= budget:
                best, best_q = data, q
                lo = q + 1
            else:
                hi = q - 1
        if best is not None:
            return best + b'\x00' * (budget - len(best)), len(best), best_q, side
    return None


def encode_to_budget(image, budget):
    # Best decodable JPEG that fits budget. Returns (padded, length, quality, side).
    # RGB first; grayscale fallback when the budget is tight.
    out = _encode_mode(image, budget, 'RGB')
    if out is not None:
        return out
    out = _encode_mode(image, budget, 'L')
    if out is not None:
        return out
    raise ValueError(
        f'no JPEG fits {budget} B down to {JPEG_SIDES[-1]}x{JPEG_SIDES[-1]}')


def decode(payload, length):
    try:
        return Image.open(io.BytesIO(payload[:length])).convert('RGB')
    except Exception:
        return None
