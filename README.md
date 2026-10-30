# Spectral Minima and Persistent Homology of Flat Tori under Affine Metric Deformations

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23652663.svg)](https://doi.org/10.5281/zenodo.23652663)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

Reproducibility artifact for the paper:

**Spectral Minima and Persistent Homology of Flat Tori under Affine Metric Deformations**
Akbar Dehghan Nezhad and Mojtaba Janati
School of Mathematics and Computer Science, Iran University of Science and Technology (2026)
DOI: https://doi.org/10.5281/zenodo.23652663

## Repository structure

- `figures/` — 6 figures (PDF + PNG), 12 files total
- `src/all_figures_v7.py` — generates all 6 figures
- `src/all_tables_final.py` — generates all 10 LaTeX tables
- `tables/` — 10 LaTeX tables (`tab_1_complexity.tex` through `tab_10_sample_structure.tex`)
- `requirements.txt` — Python dependencies
- `CITATION.cff` — machine-readable citation metadata
- `LICENSE` — MIT License

## Requirements

Python 3.10 or newer, with NumPy, SciPy, Ripser, Persim, scikit-learn, and Matplotlib. Install with:

    pip install -r requirements.txt

A virtual environment is recommended:

    python -m venv .venv
    .venv\Scripts\activate
    pip install -r requirements.txt

(On Linux or macOS, use `source .venv/bin/activate` instead.)

## Reproducing the figures

From the repository root:

    python src/all_figures_v7.py

This produces 6 PDF and 6 PNG files in `figures/`. Runtime is under one minute.

## Reproducing the tables

    python src/all_tables_final.py --task all

This generates all 10 tables sequentially (90-120 minutes on 4 cores). For a quick subset:

    python src/all_tables_final.py --task quick

For an individual table:

    python src/all_tables_final.py --task t1

Substitute `t1` through `t10` for the other tables. Output LaTeX files are written to `tables/`.

## Reproducibility settings

- Random seed: `RANDOM_SEED = 42`
- Classification realisations: `N_CLASS_REAL = 30`
- Sample size (tables 1-8, 10): `N_SAMPLE = 150`
- Sample size (figures 4-6): 200
- Deformation interval: `[0, 3]`
- Persistence library: Ripser 0.6+

With these settings, all figures and tables are reproduced exactly, up to floating-point round-off and library version differences.

## Hardware and runtime

Runtimes reported in the paper were obtained on an Intel i7 workstation with 32 GB RAM. Approximate wall-clock times:

- Figures: under 1 minute
- Table 5 (cost breakdown): under 1 minute
- Table 9 (small-epsilon regime): about 85 minutes on 4 cores
- All 10 tables: about 90-120 minutes on 4 cores

## What the code does

1. Spectral minimum Lambda(eps), evaluated exactly from the closed-form formula using a certified branch-and-bound enumeration (Lemma 4.1, Section 4 of the paper).
2. Spectral branches E_m(eps), evaluated exactly via the same quadratic form (Proposition 3.3). Used in Figures 1 and 3.
3. Heat trace Theta_t(eps), truncated sum over the frequency lattice with a certified error bound (Proposition 6.2).
4. Vietoris-Rips persistence diagrams, computed with Ripser using periodic torus distances.
5. Bottleneck distance, computed via Persim, used to compare persistence diagrams across deformation parameters (Theorem 5.3).
6. Spectral-persistent descriptor, the joint object D_{t,j}(eps; X) = (Lambda(eps), Theta_t(eps), Dgm_j(X, d_eps)) of Definition 6.4.

## Citation

If you use this code or build on the results of the paper, please cite:

Dehghan Nezhad, A., Janati, M. (2026). Spectral Minima and Persistent Homology of Flat Tori under Affine Metric Deformations. DOI: 10.5281/zenodo.23652663

BibTeX:

    @misc{dehghan2026spectral,
      title  = {Spectral Minima and Persistent Homology of Flat Tori under Affine Metric Deformations},
      author = {Dehghan Nezhad, Akbar and Janati, Mojtaba},
      year   = {2026},
      doi    = {10.5281/zenodo.23652663},
      url    = {https://doi.org/10.5281/zenodo.23652663}
    }

A machine-readable citation is also provided in `CITATION.cff`.

## License

MIT License. See `LICENSE` for the full text.

## Versioning

The version of record cited in the paper is v1.0.0, archived at DOI 10.5281/zenodo.23652663. Later versions (v1.0.1 onwards) contain naming and metadata fixes only and do not affect the mathematical content or numerical results.

## Contact

- Akbar Dehghan Nezhad — dehghannezhad@iust.ac.ir
- Mojtaba Janati — mojtaba_janati@mathdep.iust.ac.ir

School of Mathematics and Computer Science, Iran University of Science and Technology, Tehran, Iran.