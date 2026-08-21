"""Top-1 accuracy of the float CPU encoder on the same per-class subset the
Hailo run uses, so the quantized-vs-float gap is measured on identical images.
"""
import sys
from collections import defaultdict

import torch

import semcom
import semcom.dataset
import semcom.model
from scenario import helper


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

    model, preprocess = semcom.model.load(args)
    labels = semcom.dataset.get_labels(args)
    text_feats = semcom.model.encode_text(
        model, semcom.dataset.get_tokens(args),
    ).cpu()

    clip_ds = semcom.dataset.create(args, transform=preprocess).dataset
    targets = [int(t) for t in clip_ds.targets]
    indices = _indices_per_class(targets, args.n_per_class, len(labels))

    correct = 0
    for idx in indices:
        image, y = clip_ds[idx]
        s = semcom.model.encode_image(model, image.unsqueeze(0))[0].cpu()
        if semcom.model.predict(s, text_feats) == int(y):
            correct += 1

    print(f'{args.dataset} {args.model} float-CPU '
          f'N={len(indices)} acc={correct / len(indices):.4f}')


if __name__ == '__main__':
    main()
