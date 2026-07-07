# Reproduction targets for rag-override-circuits.
# Thin wrapper over run_all.py so `make phase2` and
# `python run_all.py phase2` are equivalent. Override the interpreter with
#   make PYTHON=/path/to/python phase2
# On Windows the venv interpreter is .venv/Scripts/python.exe; on Unix it is
# .venv/bin/python. PYTHON defaults to whatever `python` resolves to.

PYTHON ?= python

.PHONY: all smoke phase0 phase1 phase2 phase3 phase4 phase5 install clean

install:
	$(PYTHON) -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -e .

smoke:
	$(PYTHON) run_all.py smoke

phase0:
	$(PYTHON) run_all.py phase0

phase1:
	$(PYTHON) run_all.py phase1

phase2:
	$(PYTHON) run_all.py phase2

phase3:
	$(PYTHON) run_all.py phase3

phase4:
	$(PYTHON) run_all.py phase4

phase5:
	$(PYTHON) run_all.py phase5

all:
	$(PYTHON) run_all.py all

clean:
	$(PYTHON) -c "import shutil,glob,os; [os.remove(p) for p in glob.glob('results/*.png')]; print('removed regenerable figures from results/')"
