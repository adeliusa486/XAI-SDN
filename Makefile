# XAI-SDN — Reproducibility Makefile
# Run `make reproduce` to regenerate all paper artifacts from scratch.
# Run `make paper`     to compile the LaTeX PDF only.
# Run `make clean`     to remove generated artifacts.

PYTHON    := python
PAPER_DIR := paper
ARTIFACTS := model/artifacts

.PHONY: all reproduce synthetic-experiments paper clean help

# ─── Default target ───────────────────────────────────────────────────────────
all: help

help:
	@echo ""
	@echo "  XAI-SDN Reproducibility Makefile"
	@echo "  ─────────────────────────────────"
	@echo "  make reproduce    Re-run ALL experiments + compile PDF"
	@echo "  make synthetic    Run experiments on synthetic data only"
	@echo "  make paper        Compile LaTeX PDF only"
	@echo "  make clean        Remove generated artifact files"
	@echo ""

# ─── Full reproduce (requires real CIC-DDoS2019 data in data/raw/) ───────────
reproduce: check-data
	@echo "[1/5] Generating ROC/PR curves..."
	$(PYTHON) scripts/generate_roc_pr.py
	@echo "[2/5] Running 10-seed Wilcoxon ablation..."
	$(PYTHON) scripts/run_10seed_wilcoxon.py
	@echo "[3/5] Running multi-partition evaluation..."
	$(PYTHON) scripts/run_multipartition.py
	@echo "[4/5] Running cross-partition generalization matrix..."
	$(PYTHON) scripts/run_cross_partition.py
	@echo "[5/5] Running XAI quality evaluation..."
	$(PYTHON) explainability/evaluate_xai_quality.py
	@echo ""
	@echo "All artifacts generated. Compiling PDF..."
	$(MAKE) paper

check-data:
	@if [ ! -d "data/raw" ]; then \
		echo "ERROR: data/raw/ not found."; \
		echo "Download CIC-DDoS2019 from https://www.unb.ca/cic/datasets/ddos-2019.html"; \
		echo "and place CSV files in data/raw/."; \
		echo ""; \
		echo "To run with synthetic data only: make synthetic"; \
		exit 1; \
	fi

# ─── Synthetic-only run (no real data needed) ─────────────────────────────────
synthetic:
	@echo "[1/5] Generating ROC/PR curves (synthetic)..."
	$(PYTHON) scripts/generate_roc_pr.py --use-synthetic
	@echo "[2/5] Running 10-seed Wilcoxon ablation (synthetic)..."
	$(PYTHON) scripts/run_10seed_wilcoxon.py --use-synthetic --max-samples 8000
	@echo "[3/5] Running multi-partition evaluation (synthetic)..."
	$(PYTHON) scripts/run_multipartition.py --use-synthetic
	@echo "[4/5] Running cross-partition matrix (synthetic)..."
	$(PYTHON) scripts/run_cross_partition.py --use-synthetic
	@echo "[5/5] Running XAI quality evaluation (synthetic)..."
	$(PYTHON) explainability/evaluate_xai_quality.py --use-synthetic --n-explain 300
	@echo "[6/6] Running InSDN cross-dataset transfer (synthetic)..."
	$(PYTHON) scripts/run_insdn_transfer.py --use-synthetic --n-samples 10000
	@echo ""
	@echo "All synthetic artifacts generated in $(ARTIFACTS)/"

# ─── PDF compilation ──────────────────────────────────────────────────────────
paper:
	@echo "Compiling LaTeX paper..."
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode XAI-SDN-journal.tex
	cd $(PAPER_DIR) && bibtex XAI-SDN-journal
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode XAI-SDN-journal.tex
	cd $(PAPER_DIR) && pdflatex -interaction=nonstopmode XAI-SDN-journal.tex
	@echo "PDF ready: $(PAPER_DIR)/XAI-SDN-journal.pdf"

# ─── Clean ────────────────────────────────────────────────────────────────────
clean:
	rm -f $(PAPER_DIR)/*.aux $(PAPER_DIR)/*.bbl $(PAPER_DIR)/*.blg \
	       $(PAPER_DIR)/*.log $(PAPER_DIR)/*.out $(PAPER_DIR)/*.synctex.gz
	@echo "LaTeX build files removed. Artifacts in $(ARTIFACTS)/ preserved."
