# Spectral Minima and Persistent Homology of Flat Tori under Affine Metric Deformations

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23652663.svg)](https://doi.org/10.5281/zenodo.23652663)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

This repository contains the complete code, data, and reproducibility artifacts
for the paper:

> **Spectral Minima and Persistent Homology of Flat Tori under Affine Metric Deformations**
> Akbar Dehghan Nezhad and Mojtaba Janati
> School of Mathematics and Computer Science, Iran University of Science and Technology
> (2026). Preprint. DOI: [10.5281/zenodo.23652663](https://doi.org/10.5281/zenodo.23652663)

---

## Repository Structure

---

## Requirements

- **Python** 3.10 or newer
- **NumPy** ≥ 1.24
- **SciPy** ≥ 1.10
- **Ripser** ≥ 0.6 (persistent homology)
- **Persim** ≥ 0.3 (bottleneck distance)
- **scikit-learn** ≥ 1.3
- **Matplotlib** ≥ 3.7

Install all dependencies with:

```bash
pip install -r requirements.txt
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
