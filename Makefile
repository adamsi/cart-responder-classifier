# Run the pipeline step by step. `make` alone runs everything that is out of date.
#   make setup      create .venv and install requirements
#   make data       download GEO data and build data/processed/dataset.h5ad
#   make train      cross-validate, then fit the final model -> artifacts/
#   make evaluate   AUC with CI, attention by cell type, example patients -> results/, examples/
#   make baseline   the paper's 27-feature models on identical folds -> results/baseline_metrics.json
#   make app        launch the demo UI
#   make deploy     upload the demo to Hugging Face Spaces
PY  = .venv/bin/python
CFG = configs/default.yaml

.PHONY: all setup data train evaluate baseline app deploy clean
SPACE = AdamSion74/cart-responder-classifier

all: evaluate baseline

setup:
	python3 -m venv .venv && .venv/bin/pip install -q --upgrade pip && .venv/bin/pip install -q -r requirements-dev.txt

data/processed/dataset.h5ad: src/data.py src/common.py $(CFG)
	$(PY) -m src.data --config $(CFG)

artifacts/model.pt: src/train.py src/model.py src/common.py data/processed/dataset.h5ad
	$(PY) -m src.train --config $(CFG)

results/metrics.json: src/evaluate.py artifacts/model.pt
	$(PY) -m src.evaluate --config $(CFG)

results/baseline_metrics.json: src/baseline.py results/metrics.json
	$(PY) -m src.baseline --config $(CFG)

data:     data/processed/dataset.h5ad
train:    artifacts/model.pt
evaluate: results/metrics.json
baseline: results/baseline_metrics.json

app:
	$(PY) app.py

deploy:
	.venv/bin/hf repos create $(SPACE) --type space --sdk gradio --exist-ok
	.venv/bin/hf upload $(SPACE) . . --repo-type space --exclude ".venv/*" "data/*" "results/*" ".git/*" "__pycache__/*" "*/__pycache__/*" ".DS_Store" --commit-message "deploy demo"

clean:
	rm -rf artifacts results data/processed
