from scenario import helper
import semcom
import semcom.jpeg
import pysdr
import pysdr.qam


def main():
    args = helper.args.parse_args()
    n_frame = args.n_baseline_frame
    # ONE air frame carries all blocks (single preamble, single TX/RX)
    args.n_payload = n_frame * (args.n_fft + args.n_cp)

    model, preprocess = semcom.model.load(args)
    budget = pysdr.qam.payload_budget(args.n_symbol, n_frame)
    tokens = semcom.dataset.get_tokens(args)
    labels = semcom.dataset.get_labels(args)
    text_feats = semcom.model.encode_text(model, tokens).cpu()

    # PIL source for JPEG (no CLIP preprocess on TX)
    raw = semcom.dataset.create_pil(args)
    pil, y = raw[0]
    y = int(y)

    payload, length, quality, side = semcom.jpeg.encode_to_budget(pil, budget)
    s = pysdr.qam.modulate(payload, length, args.n_symbol, n_frame)

    transmitter, receiver = pysdr.create(args)
    transmitter.prepare(pysdr.ofdm.modulate(s, args))
    transmitter.transmit()
    z_hat = receiver.receive()
    transmitter.stop()
    s_hat = pysdr.ofdm.demodulate(z_hat, args)

    rx_payload, rx_length = pysdr.qam.demodulate(s_hat, args.n_symbol, n_frame)
    if rx_payload is None:
        pred = -1
        ok = False
    else:
        image = semcom.jpeg.decode(rx_payload, rx_length)
        if image is None:
            pred = -1
            ok = False
        else:
            x = preprocess(image).unsqueeze(0).to(args.device)
            # re-embed recovered image, then classify in CLIP space
            s_clip = semcom.model.encode_image(model, x)[0].cpu()
            pred = semcom.model.predict(s_clip, text_feats)
            ok = pred == y

    print(
        f'[test_baseline] dataset={args.dataset} model={args.model} '
        f'transceiver={args.transceiver} snr_db={args.snr_db} '
        f'n_symbol={args.n_symbol} budget={budget}B '
        f'jpeg_q={quality} side={side} '
        f'y={labels[y]} pred={labels[pred] if pred >= 0 else None} ok={ok}'
    )


if __name__ == '__main__':
    main()
