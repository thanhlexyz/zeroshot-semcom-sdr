"""RQ3 runtime for the semcom encode path on the Hailo-10H NPU.

Mirrors scenario/rq3/run.py: times encode_image and predict per image, but the
image encoder runs on the NPU instead of the CPU. Also records top-1 accuracy,
since the compiled HEF is quantized and the CPU model is not.

Usage:
    python3 rq3_hailo.py --hef PATH --csv-out PATH [--dataset cifar10] [...]
Remaining flags are passed through to scenario.helper.args.
"""
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

from hailo_platform import VDevice, FormatType


def _pop_flag(argv, name, default=None):
    if name in argv:
        i = argv.index(name)
        value = argv[i + 1]
        del argv[i:i + 2]
        return value
    return default


HEF_PATH = _pop_flag(sys.argv, '--hef')
CSV_OUT = _pop_flag(sys.argv, '--csv-out')
assert HEF_PATH and CSV_OUT, 'need --hef and --csv-out'

import semcom  # noqa: E402
import semcom.dataset  # noqa: E402
import semcom.model  # noqa: E402
from scenario import helper  # noqa: E402


def _to_symbols(feat):
    # same as semcom.model.encode_image, minus the backbone forward
    feat = feat / np.linalg.norm(feat)
    return feat[0::2] + 1j * feat[1::2]


def _predict(symbols, text_feats):
    s = torch.as_tensor(symbols, dtype=torch.complex64).reshape(-1)
    feat = torch.empty(s.numel() * 2, dtype=torch.float32)
    feat[0::2] = s.real
    feat[1::2] = s.imag
    feat = feat / (feat.norm() + 1e-12)
    return int(torch.argmax(text_feats @ feat).item())


def _indices_per_class(targets, n_per_class, n_class):
    by_class = defaultdict(list)
    for i, y in enumerate(targets):
        y = int(y)
        if len(by_class[y]) < n_per_class:
            by_class[y].append(i)
        if all(len(by_class[c]) >= n_per_class for c in range(n_class)):
            break
    return [i for c in range(n_class) for i in by_class[c]]


def main():
    args = helper.args.parse_args()
    args.device = 'cpu'
    args.model = 'clip'

    # CPU backbone is loaded only for the text tower; in the paper's scheme the
    # prompt embeddings are cached at the receiver, so this is off the timed path
    model, _ = semcom.model.load(args)
    labels = semcom.dataset.get_labels(args)
    text_feats = semcom.model.encode_text(
        model, semcom.dataset.get_tokens(args),
    ).cpu()
    del model

    pil_ds = semcom.dataset.create_pil(args)
    targets = getattr(pil_ds, 'targets', None)
    if targets is None:
        targets = [pil_ds[i][1] for i in range(len(pil_ds))]
    indices = _indices_per_class(targets, args.n_per_class, len(labels))

    target = VDevice()
    infer_model = target.create_infer_model(HEF_PATH)
    infer_model.output().set_format_type(FormatType.FLOAT32)
    configured = infer_model.configure()
    bindings = configured.create_bindings()
    inp, out = infer_model.input(), infer_model.output()
    h, w, _ = inp.shape
    outbuf = np.zeros(out.shape, dtype=np.float32)
    bindings.output(out.name).set_buffer(outbuf)

    rows = []
    correct = 0

    warm = np.ascontiguousarray(
        np.array(pil_ds[indices[0]][0].convert('RGB').resize((w, h))))
    bindings.input(inp.name).set_buffer(warm)
    configured.run([bindings], timeout=5000)

    for idx in indices:
        pil, y = pil_ds[idx]
        arr = np.ascontiguousarray(
            np.array(pil.convert('RGB').resize((w, h))))

        t0 = time.perf_counter()
        bindings.input(inp.name).set_buffer(arr)
        configured.run([bindings], timeout=5000)
        symbols = _to_symbols(outbuf.flatten().astype(np.float32))
        t_encode = time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = _predict(symbols, text_feats)
        t_decode = time.perf_counter() - t0

        correct += int(pred == int(y))
        rows.append(dict(
            method='semcom', idx=idx,
            t_encode=t_encode, t_decode=t_decode,
            t_total=t_encode + t_decode,
        ))

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(CSV_OUT) or '.', exist_ok=True)
    df.to_csv(CSV_OUT, index=False)
    print(f'[save] {CSV_OUT}')
    print(f'N={len(indices)} acc={correct / len(indices):.4f}')
    print(f'encode median={df.t_encode.median() * 1000:.3f} ms '
          f'mean={df.t_encode.mean() * 1000:.3f} ms')
    print(f'decode median={df.t_decode.median() * 1000:.4f} ms '
          f'mean={df.t_decode.mean() * 1000:.4f} ms')


if __name__ == '__main__':
    main()
