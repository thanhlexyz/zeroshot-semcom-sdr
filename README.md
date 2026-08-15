# PlutoSDR implementation of Zero-shot Semantic Communication

## Layout

- `pysdr/` — OFDM, software Rayleigh link, Pluto hardware TX/RX
- `semcom/` — CLIP / MobileCLIP encode / predict
- `scenario/` — experiment scripts

## Installation (laptop)

```bash
pip3 install -r requirements.txt
pip3 install git+https://github.com/openai/CLIP.git
pip3 install open_clip_torch timm
# for hardware on a machine with Plutos:
# pip3 install pyadi-iio
```

CIFAR-10: copy extracted batches (not the tarball) from verify-djscc:

```bash
mkdir -p data/data
cp -a ../verify-djscc/code/data/data/cifar10 data/data/
```

## Models

`--model=clip` (default) — OpenAI CLIP ViT-B/32  
`--model=mobileclip` — Apple MobileCLIP2-S0 via `open_clip` (`pretrained=dfndr2b`)

Both produce a 512-d embedding → 256 complex OFDM symbols. Results are tagged
`_{dataset}_{model}` in CSVs/figures so CLIP and MobileCLIP can coexist.

## Smoke test (software, no radio)

```bash
python3 -m scenario.test_semcom --model=clip
python3 -m scenario.test_semcom --model=mobileclip --device=cpu
# or: make test MODEL=mobileclip
```

## Pi4 + Pluto (over the air)

The Pi 4 (Cortex-A72, ARMv8.0) needs `torch>=2.8`; older aarch64 wheels are built
with ARMv8.1 LSE atomics and crash with `Illegal instruction` when loading CLIP.
Install CLIP and open_clip. Weights download into `data/model/` on first use
(Pi needs network, or rsync checkpoints). MobileOne fuse is in-tree — no
`ml-mobileclip` package required.

```bash
ssh pi4 '~/venv/bin/pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu'
ssh pi4 '~/venv/bin/pip install ftfy regex git+https://github.com/openai/CLIP.git'
ssh pi4 '~/venv/bin/pip install open_clip_torch timm'
# optional if no Pi internet for weights:
# rsync -avz data/model/ pi4:~/wpmc/data/model/
```

OTA smoke test:

```bash
make test-ota MODEL=mobileclip DATASET=tsrd
```

Full RQ1+RQ3 for one dataset×model (on Pi after `make sync`):

```bash
# reuse existing rq1_measure_{dataset}.csv
make all-nomeasure DATASET=cifar10 MODEL=mobileclip
make all-nomeasure DATASET=tsrd MODEL=mobileclip
```

Then locally:

```bash
make all-plot DATASET=cifar10 MODEL=mobileclip
make all-plot DATASET=tsrd MODEL=mobileclip
```
