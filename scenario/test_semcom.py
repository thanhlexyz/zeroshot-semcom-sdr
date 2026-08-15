from scenario import helper
import semcom
import pysdr


def main():
    args = helper.args.parse_args()

    model, preprocess = semcom.model.load(args)
    loader = semcom.dataset.create(args, transform=preprocess)
    tokens = semcom.dataset.get_tokens(args)
    labels = semcom.dataset.get_labels(args)
    text_feats = semcom.model.encode_text(model, tokens).cpu()
    images, targets = next(iter(loader))
    image = images[0:1].to(args.device)
    y = int(targets[0])
    s = semcom.model.encode_image(model, image)[0].cpu()

    transmitter, receiver = pysdr.create(args)
    transmitter.prepare(pysdr.ofdm.modulate(s, args))
    transmitter.transmit()
    z_hat = receiver.receive()
    transmitter.stop()
    s_hat = pysdr.ofdm.demodulate(z_hat, args)
    pred = semcom.model.predict(s_hat, text_feats)
    print(
        f'[test] dataset={args.dataset} model={args.model} '
        f'transceiver={args.transceiver} snr_db={args.snr_db} '
        f'n_symbol={args.n_symbol} '
        f'y={labels[y]} pred={labels[pred]} ok={pred == y}'
    )


if __name__ == '__main__':
    main()
