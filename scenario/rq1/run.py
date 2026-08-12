import os
import time
from collections import defaultdict

from scenario import helper
from scenario.rq1 import TX_POWERS
import semcom
import semcom.jpeg
import pysdr
import pysdr.qam


def _indices_per_class(dataset, n_per_class, n_class):
    # first n_per_class test indices for each class (deterministic)
    by_class = defaultdict(list)
    for i in range(len(dataset)):
        y = int(dataset.targets[i])
        if len(by_class[y]) < n_per_class:
            by_class[y].append(i)
        if all(len(by_class[c]) >= n_per_class for c in range(n_class)):
            break
    return [i for c in range(n_class) for i in by_class[c]]


def _run_semcom(model, image, y, text_feats, args, transmitter, receiver):
    # time only semcom (CLIP); OFDM/radio excluded
    t0 = time.perf_counter()
    s = semcom.model.encode_image(model, image)[0].cpu()
    t_encode = time.perf_counter() - t0

    transmitter.stop()
    transmitter.prepare(pysdr.ofdm.modulate(s, args))
    transmitter.transmit()
    z_hat = receiver.receive()
    transmitter.stop()
    s_hat = pysdr.ofdm.demodulate(z_hat, args)

    t0 = time.perf_counter()
    pred = semcom.model.predict(s_hat, text_feats)
    t_decode = time.perf_counter() - t0
    ok = int(pred == y)
    return pred, ok, t_encode, t_decode


def _run_baseline(model, preprocess, pil, y, text_feats, args,
                  transmitter, receiver, budget):
    # time only semcom (JPEG + CLIP re-embed/predict); 16-QAM/OFDM/radio excluded
    t0 = time.perf_counter()
    payload, length, _, _ = semcom.jpeg.encode_to_budget(pil, budget)
    t_encode = time.perf_counter() - t0

    s = pysdr.qam.modulate(payload, length, args.n_symbol, args.n_baseline_frame)
    transmitter.stop()
    transmitter.prepare(pysdr.ofdm.modulate(s, args))
    transmitter.transmit()
    z_hat = receiver.receive()
    transmitter.stop()
    s_hat = pysdr.ofdm.demodulate(z_hat, args)
    rx_payload, rx_length = pysdr.qam.demodulate(
        s_hat, args.n_symbol, args.n_baseline_frame)

    t0 = time.perf_counter()
    if rx_payload is None:
        pred = -1
        ok = 0
    else:
        image = semcom.jpeg.decode(rx_payload, rx_length)
        if image is None:
            pred = -1
            ok = 0
        else:
            x = preprocess(image).unsqueeze(0).to(args.device)
            s_clip = semcom.model.encode_image(model, x)[0].cpu()
            pred = semcom.model.predict(s_clip, text_feats)
            ok = int(pred == y)
    t_decode = time.perf_counter() - t0
    return pred, ok, t_encode, t_decode


def main():
    args = helper.args.parse_args()
    args.transceiver = 'hardware'
    method = args.method
    if method == 'baseline':
        # ONE air frame carries all blocks (single preamble, single TX/RX)
        args.n_payload = args.n_baseline_frame * (args.n_fft + args.n_cp)

    model, preprocess = semcom.model.load(args)
    budget = pysdr.qam.payload_budget(args.n_symbol, args.n_baseline_frame)
    clip_ds = semcom.dataset.create(args, transform=preprocess).dataset
    labels = semcom.dataset.get_labels(args)
    indices = _indices_per_class(clip_ds, args.n_per_class, len(labels))
    text_feats = semcom.model.encode_text(
        model, semcom.dataset.get_tokens(args),
    ).cpu()

    pil_ds = None
    if method == 'baseline':
        pil_ds = semcom.dataset.create_pil(args)

    csv_path = os.path.join(
        args.csv_dir,
        f'rq1_run_{method}_{args.dataset}_{args.model}.csv',
    )
    pbar = helper.progress_bar.create(
        total=len(TX_POWERS) * len(indices),
        unit='image',
        desc=f'rq1.run.{method}.{args.dataset}.{args.model}',
        csv_path=csv_path,
    )
    transmitter, receiver = pysdr.create(args)
    try:
        for tx_power in TX_POWERS:
            transmitter.sdr.tx_hardwaregain_chan0 = float(tx_power)
            n_ok = 0
            for k, idx in enumerate(indices):
                y = int(clip_ds.targets[idx])
                if method == 'semcom':
                    image, _ = clip_ds[idx]
                    image = image.unsqueeze(0).to(args.device)
                    pred, ok, t_encode, t_decode = _run_semcom(
                        model, image, y, text_feats, args,
                        transmitter, receiver,
                    )
                else:
                    pil, _ = pil_ds[idx]
                    pred, ok, t_encode, t_decode = _run_baseline(
                        model, preprocess, pil, y, text_feats, args,
                        transmitter, receiver, budget,
                    )
                n_ok += ok
                acc = n_ok / (k + 1)
                pbar.step(
                    tx_power=tx_power,
                    idx=idx,
                    y=labels[y],
                    pred=labels[pred] if pred >= 0 else None,
                    ok=ok,
                    acc=acc,
                    t_encode=t_encode,
                    t_decode=t_decode,
                )
            print(
                f'[rq1.run] dataset={args.dataset} model={args.model} '
                f'method={method} tx_power={tx_power} '
                f'acc={n_ok / len(indices):.3f} ({n_ok}/{len(indices)})'
            )
    finally:
        transmitter.stop()
    pbar.to_csv()
    print(f'[save] {csv_path}')


if __name__ == '__main__':
    main()
