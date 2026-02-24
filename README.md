# LSHA: Quickstart

## 1) Prerequisites
- Anaconda or Miniconda installed
- macOS/Linux/Windows terminal

Optional but recommended:
- Graphviz installed at system level (helps `pygraphviz`); on macOS: `brew install graphviz`, on Ubuntu/Debian: `sudo apt-get install graphviz`

## 2) Create the Conda environment
```
conda env create -f environment.yml
conda activate lsha
```

If the environment already exists and you want to update it:
```
conda env update -f environment.yml --prune
conda activate lsha
```

## 3) Run the learning and Uppaal conversion pipeline
From the project root:
```
python learn_and_convert_to_upp.py
```

What it does (high level):
- Learns a Stochastic Hybrid Automaton (LSHA) for the LEGO Factory case study
- Saves the learned SHA graph and source under `sha_learning/resources/learned_sha/`
- Converts the learned model to an Uppaal NTA and stores results under `uppaal_generator/resources/gen_models/` (per the project config)
