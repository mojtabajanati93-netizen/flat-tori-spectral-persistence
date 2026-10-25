# Spectral Minima and Persistent Homology of Flat Tori under Affine Metric Deformations

This repository contains the code and data to reproduce all figures and tables in the paper:

> **Spectral Minima and Persistent Homology of Flat Tori under Affine Metric Deformations**
> Akbar Dehghan Nezhad and Mojtaba Janati (2026)

## Requirements

- Python 3.10+
- NumPy, SciPy, Ripser, Persim, scikit-learn, matplotlib

Install with:

pip install -r requirements.txt

## How to Run

1. Generate all figures:

python src/all_figures_v4.py

2. Generate all tables (set TASK inside the script first):

python src/all_tables_FINAL.py

The random seed is fixed (RANDOM_SEED=42) for exact reproducibility.

## Outputs

- Figures: figures/fig1_spectral_min.pdf to figures/fig6_analytic_example.pdf
- Tables: tables/table1_crossings.tex to tables/table10_sample_structure.tex

## License

MIT License (see LICENSE)

## Citation

See CITATION.cff.