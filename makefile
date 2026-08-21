PI = pi4
PI_DIR = ~/wpmc
PI_PY = HF_HUB_OFFLINE=1 ~/venv/bin/python
DATASET ?= cifar10
MODEL ?= clip

sync:
	rsync -avz --delete \
		--exclude '__pycache__' --exclude '.git' \
		--exclude 'data/data/cifar10/cifar-10-python.tar.gz' \
		--exclude 'data/model' \
		./ $(PI):$(PI_DIR)/

test:
	python3 -m scenario.test_semcom --dataset=$(DATASET) --model=$(MODEL)

test-baseline:
	python3 -m scenario.test_baseline --dataset=$(DATASET) --model=$(MODEL)

test-ota: sync
	ssh $(PI) 'cd $(PI_DIR) && $(PI_PY) -u -m scenario.test_semcom --transceiver=hardware --device=cpu --dataset=$(DATASET) --model=$(MODEL)'

# --------------------------------------------------------------------
# RQ1 (SDR on pi4)
# --------------------------------------------------------------------
rq1-sync: sync

rq1-pull:
	rsync -avz $(PI):$(PI_DIR)/data/csv/rq1_* data/csv/

# run on pi4 after sync
rq1-measure:
	$(PI_PY) -u -m scenario.rq1.measure --device=cpu --dataset=$(DATASET)

# run on pi4 (one method at a time)
rq1-run-semcom:
	$(PI_PY) -u -m scenario.rq1.run --device=cpu --method=semcom --dataset=$(DATASET) --model=$(MODEL)

rq1-run-baseline:
	$(PI_PY) -u -m scenario.rq1.run --device=cpu --method=baseline --dataset=$(DATASET) --model=$(MODEL)

# baseline is JPEG+QAM (not VL-model-specific); only run with MODEL=clip
rq1-run: rq1-run-semcom
ifeq ($(MODEL),clip)
rq1-run: rq1-run-baseline
endif

# laptop: plot after pull
rq1-plot:
	python3 -m scenario.rq1.plot --dataset=$(DATASET) --model=$(MODEL)

rq1: rq1-sync
	@echo 'On pi4: cd ~/wpmc && make all DATASET=$(DATASET) MODEL=$(MODEL)'
	@echo 'Then locally: make all-plot DATASET=$(DATASET) MODEL=$(MODEL)'

# --------------------------------------------------------------------
# RQ2 (laptop: payload + cross-dataset accuracy from RQ1 CSVs)
# --------------------------------------------------------------------
rq2-plot:
	python3 -m scenario.rq2.plot --model=$(MODEL)

# --------------------------------------------------------------------
# Dataset examples (laptop: one image per class, raw datasets only)
# --------------------------------------------------------------------
examples-plot:
	python3 -m scenario.examples.plot

# --------------------------------------------------------------------
# RQ3 (processing time on pi4; table on laptop)
# --------------------------------------------------------------------
# MobileCLIP: SemCom encode/decode only (baseline does not use the VL model)
rq3-run:
ifeq ($(MODEL),mobileclip)
	$(PI_PY) -u -m scenario.rq3.run --device=cpu --dataset=$(DATASET) --model=$(MODEL) --method=semcom
else
	$(PI_PY) -u -m scenario.rq3.run --device=cpu --dataset=$(DATASET) --model=$(MODEL)
endif

rq3-pull:
	rsync -avz $(PI):$(PI_DIR)/data/csv/rq3_* data/csv/

rq3-plot:
	python3 -m scenario.rq3.plot --dataset=$(DATASET) --model=$(MODEL)

rq3: sync
	@echo 'On pi4: cd ~/wpmc && make rq3-run DATASET=$(DATASET) MODEL=$(MODEL)'
	@echo 'Then locally: make rq3-pull rq3-plot DATASET=$(DATASET) MODEL=$(MODEL)'

# --------------------------------------------------------------------
# all (one dataset × model): measure + RQ1 (+ baseline if clip) + RQ3 on pi4
# --------------------------------------------------------------------
all: rq1-measure rq1-run rq3-run
	@echo 'Done on pi4 for DATASET=$(DATASET) MODEL=$(MODEL).'
	@echo 'Locally: make all-plot DATASET=$(DATASET) MODEL=$(MODEL)'

# skip measure if SNR CSV already exists (reuse for another model)
all-nomeasure: rq1-run rq3-run
	@echo 'Done on pi4 for DATASET=$(DATASET) MODEL=$(MODEL) (no measure).'
	@echo 'Locally: make all-plot DATASET=$(DATASET) MODEL=$(MODEL)'

# laptop: pull CSVs then plot RQ1/RQ3 for DATASET×MODEL + RQ2
all-plot: rq1-pull rq3-pull rq1-plot rq3-plot rq2-plot
	@echo 'Plots/tables ready for DATASET=$(DATASET) MODEL=$(MODEL)'
# --------------------------------------------------------------------
