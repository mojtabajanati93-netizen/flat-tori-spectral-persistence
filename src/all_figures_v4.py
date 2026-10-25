#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 1 (FINAL v4): Generate all 6 figures (Figure 1 to Figure 6).

Confirmed working on:
  - Python 3.10-3.13
  - matplotlib >= 3.5 (uses $...$ for mathtext, not \(...\))
  - numpy, ripser, persim

Fixes applied in this version:
  - All labels use $...$ (matplotlib-native math mode) instead of \(...\)
    which caused literal backslash rendering in the original version.
  - Figure 1 uses G = diag(1..n) and Q with q_1 = 2 to produce an
    interior transverse crossing at eps_* = (n-1)/2.
  - Figures 2 and 3 use the canonical example G = diag(1,2,3),
    Q = diag(2,0,-1), giving a genuine transverse crossing at eps_* = 2/3.
  - Figure 2 now shows the theoretical C_met * delta_eps bound and eps_*.
  - Figure 3 marks eps_* in panel (a); panel (c) shows that the
    bottleneck distance is continuous at eps_*.
  - Safe handling of empty persistence diagrams (safe_bottleneck).
  - Explicit seeds for every random quantity (reproducibility).

Output: output_figures/*.pdf (6 files)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from itertools import product
import ripser
from persim import bottleneck
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight'
})

OUTPUT_DIR = 'figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_SEED = 42

# Canonical example used in Figures 2 and 3
G_CANON = np.diag([1.0, 2.0, 3.0])
Q_CANON = np.diag([2.0, 0.0, -1.0])
EPS_STAR_CANON = 2.0 / 3.0


# ============================================================
# Helper functions
# ============================================================
def generate_normalized_metric(n, seed):
    """Random SPD matrix with trace(Q) = n."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Q = A.T @ A + 0.5 * np.eye(n)
    return Q * (n / np.trace(Q))


def get_candidate_vectors(n, max_norm=5):
    """Branch-and-bound enumeration of {m : |m|_2 <= max_norm, m != 0, m ~ -m}."""
    seen, Ms = set(), []

    def recurse(prefix, remaining_sq):
        if len(prefix) == n:
            if any(x != 0 for x in prefix):
                m = tuple(prefix)
                neg = tuple(-x for x in prefix)
                key = m if m < neg else neg
                if key not in seen:
                    seen.add(key); Ms.append(m)
            return
        mx = int(np.floor(np.sqrt(remaining_sq)))
        for x in range(-mx, mx + 1):
            recurse(prefix + (x,), remaining_sq - x * x)

    recurse((), max_norm ** 2)
    return np.array(Ms, dtype=int).reshape(-1, n) if Ms else np.empty((0, n), dtype=int)


def compute_certified_L(G, Q, eps_range=(0.0, 3.0), n_samples=50):
    """Certified L from Lemma 4.3 (m_0 = e_1)."""
    eps_samples = np.linspace(eps_range[0], eps_range[1], n_samples)
    eigvals = np.array([np.linalg.eigvalsh(G + eps * Q) for eps in eps_samples])
    mu = eigvals.min(); M = eigvals.max()
    if mu <= 1e-12:
        return None
    return max(1, int(np.ceil(np.sqrt(M / mu))))


def compute_lambda_fast(g_eps, Ms):
    g_inv = np.linalg.inv(g_eps)
    return 4 * np.pi ** 2 * np.min(np.einsum('ki,ij,kj->k', Ms, g_inv, Ms))


def compute_heat_trace(g_eps, max_norm=5, t=0.1):
    g_inv = np.linalg.inv(g_eps)
    n = g_eps.shape[0]
    total = 0.0
    for m in product(range(-max_norm, max_norm + 1), repeat=n):
        v = np.array(m)
        total += np.exp(-4 * np.pi ** 2 * t * (v @ g_inv @ v))
    return total


def periodic_distance_matrix(X, g_eps, max_norm=1):
    N, n = X.shape
    shifts = np.array(list(product(range(-max_norm, max_norm + 1), repeat=n)))
    diff = X[:, None, :] - X[None, :, :]
    D = np.full((N, N), np.inf)
    for z in shifts:
        D = np.minimum(D, np.einsum('ijk,kl,ijl->ij', diff + z, g_eps, diff + z))
    return np.sqrt(np.maximum(D, 0))


def get_persistence_diagram(D):
    diagrams = ripser.ripser(D, maxdim=1, distance_matrix=True)['dgms']
    if len(diagrams) > 1 and len(diagrams[1]) > 0:
        return diagrams[1]
    return np.empty((0, 2))


def safe_bottleneck(dgm1, dgm2):
    """Bottleneck distance, robust to empty diagrams."""
    dgm1 = dgm1 if len(dgm1) > 0 else np.empty((0, 2))
    dgm2 = dgm2 if len(dgm2) > 0 else np.empty((0, 2))
    try:
        return float(bottleneck(dgm1, dgm2))
    except Exception:
        if len(dgm1) == 0 and len(dgm2) == 0:
            return 0.0
        dgm = dgm1 if len(dgm2) == 0 else dgm2
        return float(np.max(dgm[:, 1] - dgm[:, 0]) / 2) if len(dgm) > 0 else 0.0


def detect_all_crossings(Q, G, eps_grid, Ms):
    """All transverse crossings of the active minimizer."""
    E = np.zeros((len(eps_grid), len(Ms)))
    for i, eps in enumerate(eps_grid):
        g_inv = np.linalg.inv(G + eps * Q)
        E[i] = 4 * np.pi ** 2 * np.einsum('ki,ij,kj->k', Ms, g_inv, Ms)
    active = np.argmin(E, axis=1)
    changes = np.where(np.diff(active) != 0)[0]
    crossings = []
    for idx in changes:
        m1, m2 = Ms[active[idx]], Ms[active[idx + 1]]

        def f(eps):
            g_inv = np.linalg.inv(G + eps * Q)
            return m1 @ g_inv @ m1 - m2 @ g_inv @ m2

        lo, hi = eps_grid[idx], eps_grid[idx + 1]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if f(lo) * f(mid) <= 0:
                hi = mid
            else:
                lo = mid
        eps_star = 0.5 * (lo + hi)
        g_inv = np.linalg.inv(G + eps_star * Q)
        d1 = -4 * np.pi ** 2 * m1 @ g_inv @ Q @ g_inv @ m1
        d2 = -4 * np.pi ** 2 * m2 @ g_inv @ Q @ g_inv @ m2
        crossings.append((eps_star, abs(d1 - d2)))
    return crossings


# ============================================================
# Figure 1: interior transverse crossing + percentile band
# ============================================================
def fig1_spectral_min():
    """
    For each n in {2, 3, 4}:
      - G = diag(1, 2, ..., n), Q = diag(2, 0, ..., 0)
      - interior transverse crossing at eps_* = (n-1)/2
      - solid line: representative realization
      - shaded band: 5-95 percentile across 50 realizations
      - vertical line: theoretical eps_*
    """
    print("Generating Figure 1 (interior transverse crossing + percentile band)...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    n_real = 50
    eps_grid = np.linspace(0, 3, 200)

    for idx, n in enumerate([2, 3, 4]):
        ax = axes[idx]
        G = np.diag(np.arange(1, n + 1, dtype=float))
        Q_ref = np.zeros((n, n))
        Q_ref[0, 0] = 2.0

        L_ref = compute_certified_L(G, Q_ref, eps_range=(0.0, 3.0))
        Ms_ref = get_candidate_vectors(n, max_norm=max(L_ref, 3))

        lam_ref = np.array([compute_lambda_fast(G + eps * Q_ref, Ms_ref)
                            for eps in eps_grid])

        all_lambdas = np.zeros((n_real, len(eps_grid)))
        for r in range(n_real):
            rng = np.random.default_rng(RANDOM_SEED + idx * 1000 + r)
            Q_r = np.zeros((n, n))
            Q_r[0, 0] = 2.0 + 0.15 * rng.standard_normal()
            if Q_r[0, 0] < 0.5:
                Q_r[0, 0] = 0.5
            for i, eps in enumerate(eps_grid):
                all_lambdas[r, i] = compute_lambda_fast(G + eps * Q_r, Ms_ref)

        lo_band = np.percentile(all_lambdas, 5, axis=0)
        hi_band = np.percentile(all_lambdas, 95, axis=0)

        ax.fill_between(eps_grid, lo_band, hi_band, alpha=0.20, color='blue',
                        label='5-95% band (50 real.)')
        ax.plot(eps_grid, lam_ref, 'b-', linewidth=2,
                label=r'Representative $\Lambda(\varepsilon)$')

        eps_star = (n - 1) / 2.0
        ax.axvline(eps_star, color='red', linestyle='--',
                   label=rf'theoretical $\varepsilon_* = {eps_star:.2f}$')

        ax.set_xlabel(r'$\varepsilon$')
        ax.set_ylabel(r'$\Lambda(\varepsilon)$')
        ax.set_title(rf'$n = {n}$')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_spectral_min.pdf'))
    plt.close()


# ============================================================
# Figure 2: bottleneck distance with C_met bound and eps*
# ============================================================
def fig2_bottleneck():
    """
    Uses the canonical example G = diag(1,2,3), Q = diag(2,0,-1),
    giving a genuine transverse crossing at eps_* = 2/3.
    Shows the theoretical C_met * delta_eps bound.
    """
    print("Generating Figure 2 (with C_met line and eps_*)...")
    n = 3
    G, Q = G_CANON, Q_CANON
    L = compute_certified_L(G, Q, eps_range=(0.0, 2.5))
    Ms = get_candidate_vectors(n, max_norm=max(L, 3))

    rng = np.random.default_rng(RANDOM_SEED + 1000)
    X = rng.random((200, n))

    eps_grid = np.linspace(0, 2.5, 100)
    dists = []
    prev_dgm = None
    for eps in eps_grid:
        g_eps = G + eps * Q
        D = periodic_distance_matrix(X, g_eps)
        dgm = get_persistence_diagram(D)
        dists.append(0.0 if prev_dgm is None else safe_bottleneck(prev_dgm, dgm))
        prev_dgm = dgm

    eigvals_all = np.array([np.linalg.eigvalsh(G + eps * Q) for eps in eps_grid])
    mu = eigvals_all.min(); M = eigvals_all.max()
    q = np.linalg.norm(Q, ord=2)
    C_met = q * np.sqrt(M * n) / (4 * mu)
    d_eps = eps_grid[1] - eps_grid[0]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(eps_grid[1:], dists[1:], 'b-', linewidth=1.5,
            label=r'$d_B(\mathrm{Dgm}_1(\varepsilon),\mathrm{Dgm}_1(\varepsilon+\Delta\varepsilon))$')
    ax.axhline(C_met * d_eps, color='gray', linestyle='--', linewidth=1.5,
               label=rf'$C_{{\mathrm{{met}}}}\Delta\varepsilon \approx {C_met * d_eps:.4f}$')

    crossings = detect_all_crossings(Q, G, eps_grid, Ms)
    eps_marker = crossings[0][0] if crossings else EPS_STAR_CANON
    ax.axvline(eps_marker, color='red', linestyle=':', linewidth=1.5,
               label=rf'$\varepsilon_* \approx {eps_marker:.3f}$')

    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel('Bottleneck distance')
    ax.set_title(rf'Persistence stability ($n={n}$, 200 points)')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_bottleneck.pdf'))
    plt.close()


# ============================================================
# Figure 3: three descriptor components (canonical example)
# ============================================================
def fig3_descriptor_components():
    """Three components of the descriptor, using the canonical example."""
    print("Generating Figure 3 (three descriptor components)...")
    n = 3
    G, Q = G_CANON, Q_CANON
    L = compute_certified_L(G, Q, eps_range=(0.0, 2.5))
    Ms = get_candidate_vectors(n, max_norm=max(L, 3))

    rng = np.random.default_rng(RANDOM_SEED + 1000)
    X = rng.random((200, n))
    eps_grid = np.linspace(0, 2.5, 100)

    D0 = periodic_distance_matrix(X, G)
    ref_dgm = get_persistence_diagram(D0)

    lambdas, heat_traces, p_dists = [], [], []
    for eps in eps_grid:
        g_eps = G + eps * Q
        lambdas.append(compute_lambda_fast(g_eps, Ms))
        heat_traces.append(compute_heat_trace(g_eps, t=0.1))
        D = periodic_distance_matrix(X, g_eps)
        p_dists.append(safe_bottleneck(ref_dgm, get_persistence_diagram(D)))

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(eps_grid, lambdas, 'b-', linewidth=1.8)
    axes[0].axvline(EPS_STAR_CANON, color='red', linestyle=':', alpha=0.7,
                    label=r'theoretical $\varepsilon_* = 2/3$')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].set_ylabel(r'$\Lambda(\varepsilon)$')
    axes[0].set_title(r'(a) Spectral minimum: corner at transverse crossing')

    axes[1].plot(eps_grid, heat_traces, 'g-', linewidth=1.8)
    axes[1].set_ylabel(r'$\Theta_{0.1}(\varepsilon)$')
    axes[1].set_title(r'(b) Heat trace: smooth across the crossing')

    axes[2].plot(eps_grid, p_dists, 'm-', linewidth=1.8)
    axes[2].set_xlabel(r'$\varepsilon$')
    axes[2].set_ylabel(r'$d_B(\mathrm{Dgm}_1,\, \mathrm{Dgm}_1^0)$')
    axes[2].set_title(r'(c) Bottleneck distance to the reference diagram at $\varepsilon=0$')

    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_descriptor_components.pdf'))
    plt.close()


# ============================================================
# Figure 4: multiple deformation paths with explicit seeds
# ============================================================
def fig4_multiple_paths():
    """Three deformation paths with explicit seeds and PD check."""
    print("Generating Figure 4 (multiple explicit paths)...")
    n = 3
    G = np.eye(n)
    eps_grid = np.linspace(0, 3, 200)

    rng_diag = np.random.default_rng(RANDOM_SEED + 2000)
    rng_rank1 = np.random.default_rng(RANDOM_SEED + 3000)

    diag_entries = rng_diag.uniform(0.1, 1.0, n)
    a = rng_rank1.standard_normal(n)
    Q_rank1 = np.outer(a, a) / (a @ a)

    paths = {
        'Random SPD': generate_normalized_metric(n, seed=RANDOM_SEED),
        'Diagonal (positive)': np.diag(diag_entries),
        'Rank-one PSD': Q_rank1,
    }

    L_max = max(compute_certified_L(G, Q, eps_range=(0.0, 3.0))
                for Q in paths.values())
    Ms = get_candidate_vectors(n, max_norm=max(L_max, 3))

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, Q in paths.items():
        min_eig = min(np.linalg.eigvalsh(G + eps * Q).min() for eps in eps_grid)
        if min_eig <= 1e-10:
            print(f"  WARNING: path '{name}' becomes singular; skipped.")
            continue
        lambdas = [compute_lambda_fast(G + eps * Q, Ms) for eps in eps_grid]
        ax.plot(eps_grid, lambdas, linewidth=1.8, label=name)

    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel(r'$\Lambda(\varepsilon)$')
    ax.set_title(r'Spectral minimum for three deformation paths ($n=3$)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_multiple_paths.pdf'))
    plt.close()


# ============================================================
# Figure 5: noise robustness
# ============================================================
def fig5_noise_robustness():
    """Bottleneck distance vs. coordinate noise; 5 reps per level."""
    print("Generating Figure 5 (noise robustness, 5 reps per level)...")
    n = 3
    Q = generate_normalized_metric(n, seed=RANDOM_SEED)
    G = np.eye(n)
    g_eps = G + 1.0 * Q

    rng = np.random.default_rng(RANDOM_SEED + 1000)
    X_clean = rng.random((200, n))
    ref_dgm = get_persistence_diagram(periodic_distance_matrix(X_clean, g_eps))

    sigmas = np.linspace(0, 0.2, 20)
    n_reps = 5
    dists, stds = [], []
    for sigma in sigmas:
        run_d = []
        for _ in range(n_reps):
            X_noisy = (X_clean + sigma * rng.standard_normal((200, n))) % 1.0
            D = periodic_distance_matrix(X_noisy, g_eps)
            run_d.append(safe_bottleneck(ref_dgm, get_persistence_diagram(D)))
        dists.append(np.mean(run_d))
        stds.append(np.std(run_d))

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(sigmas, dists, yerr=stds, fmt='bo-', capsize=5,
                label=rf'Observed ({n_reps} reps)')
    ax.set_xlabel(r'Noise level $\sigma$')
    ax.set_ylabel('Bottleneck distance')
    ax.set_title(rf'Noise robustness ($n={n}$, {n_reps} reps per level)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_noise_robustness.pdf'))
    plt.close()


# ============================================================
# Figure 6: analytic example (full symmetric interval)
# ============================================================
def fig6_analytic_example():
    """Analytic crossing at eps=0 with a corner in Lambda."""
    print("Generating Figure 6 (full analytic example)...")
    eps = np.linspace(-0.5, 0.5, 200)
    E_10 = 4 * np.pi ** 2 / (1 + eps)
    E_01 = 4 * np.pi ** 2 / (1 - eps)
    Lam = np.minimum(E_10, E_01)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(eps, E_10, 'b-', linewidth=1.5, label=r'$E_{(1,0)}(\varepsilon)$')
    ax.plot(eps, E_01, 'g-', linewidth=1.5, label=r'$E_{(0,1)}(\varepsilon)$')
    ax.plot(eps, Lam, 'r--', linewidth=2.5, label=r'$\Lambda(\varepsilon)$')
    ax.plot(0, 4 * np.pi ** 2, 'ro', markersize=8)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.7)
    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel('Eigenvalue')
    ax.set_title(r'Analytic crossing ($n=2$); corner at $\varepsilon_*=0$')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_analytic_example.pdf'))
    plt.close()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Starting Figure Generation (FINAL v4)...\n")
    fig1_spectral_min()
    fig2_bottleneck()
    fig3_descriptor_components()
    fig4_multiple_paths()
    fig5_noise_robustness()
    fig6_analytic_example()
    print(f"\nAll 6 figures saved in: {os.path.abspath(OUTPUT_DIR)}")