import os
import time
from collections import defaultdict

from scenario import helper
import semcom
import semcom.jpeg
import pysdr.qam


def _indices_per_class(dataset, n_per_class, n_class):
    by_class = defaultdict(list)
    for i in range(len(dataset)):
        y = int(dataset.targets[i])
        if len(by_class[y]) < n_per_class:
            by_class[y].append(i)
        if all(len(by_class[c]) >= n_per_class for c in range(n_class)):
            break
    return [i for c in range(n_class) for i in by_class[c]]


def _time_semcom(model, image, text_feats):
    # processing only (no radio)
    t0 = time.perf_counter()
    s = semcom.model.encode_image(model, image)[0].cpu()
    t_encode = time.perf_counter() - t0

    t0 = time.perf_counter()
    semcom.model.predict(s, text_feats)
    t_decode = time.perf_counter() - t0
    return t_encode, t_decode


def _time_baseline(model, preprocess, pil, text_feats, budget, args):
    # processing only (no radio): JPEG + CLIP re-embed/predict
    t0 = time.perf_counter()
    payload, length, _, _ = semcom.jpeg.encode_to_budget(pil, budget)
    t_encode = time.perf_counter() - t0

    t0 = time.perf_counter()
    image = semcom.jpeg.decode(payload, length)
    x = preprocess(image).unsqueeze(0).to(args.device)
    s_clip = semcom.model.encode_image(model, x)[0].cpu()
    semcom.model.predict(s_clip, text_feats)
    t_decode = time.perf_counter() - t0
    return t_encode, t_decode


def main():
    args = helper.args.parse_args()
    args.device = 'cpu'

    model, preprocess = semcom.model.load(args)
    budget = pysdr.qam.payload_budget(args.n_symbol, args.n_baseline_frame)
    clip_ds = semcom.dataset.create(args, transform=preprocess).dataset
    labels = semcom.dataset.get_labels(args)
    indices = _indices_per_class(clip_ds, args.n_per_class, len(labels))
    text_feats = semcom.model.encode_text(
        model, semcom.dataset.get_tokens(args),
    ).cpu()
    pil_ds = semcom.dataset.create_pil(args)

    # baseline JPEG path does not use the VL backbone; MobileCLIP RQ3 is SemCom only
    methods = ('semcom',) if args.model == 'mobileclip' else ('semcom', 'baseline')

    csv_path = os.path.join(
        args.csv_dir, f'rq3_runtime_{args.dataset}_{args.model}.csv',
    )
    pbar = helper.progress_bar.create(
        total=len(methods) * len(indices),
        unit='image',
        desc=f'rq3.run.{args.dataset}.{args.model}',
        csv_path=csv_path,
    )
    for method in methods:
        for idx in indices:
            if method == 'semcom':
                image, _ = clip_ds[idx]
                image = image.unsqueeze(0).to(args.device)
                t_encode, t_decode = _time_semcom(model, image, text_feats)
            else:
                pil, _ = pil_ds[idx]
                t_encode, t_decode = _time_baseline(
                    model, preprocess, pil, text_feats, budget, args,
                )
            pbar.step(
                method=method,
                idx=idx,
                t_encode=t_encode,
                t_decode=t_decode,
                t_total=t_encode + t_decode,
            )
    pbar.to_csv()
    print(f'[save] {csv_path}')


if __name__ == '__main__':
    main()
