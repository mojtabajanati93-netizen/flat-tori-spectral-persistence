#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script 1 (v7): Generate all 6 figures as PDF AND PNG.

Changes from v6:
  - FIG3: Correct crossing point is eps = 19/24 ≈ 0.792 (was wrongly 19/8).
  - FIG3: Dashed individual branches e_2 and e_2-e_3 shown for clarity.
  - FIG3: Legend notes that the crossing is a monotone kink (both one-sided
    derivatives negative), not a local extremum.
  - Everything else identical to v6 (which was verified visually).

Run:
  python all_figures_v7.py
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from pathlib import Path
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

G_CANON = np.diag([1.0, 2.0, 3.0])
Q_CANON = np.diag([2.0, 0.0, -1.0])
EPS_STAR_CANON = 2.0 / 3.0

# Correct crossing point for the rank-one path (analytically verified):
#   e_2 and e_2 - e_3 exchange at eps = 19/24.
#   Both one-sided derivatives of Lambda are negative, so this is a
#   monotone kink, not a local extremum.
EPS_RANK_ONE_CROSS = 19.0 / 24.0


# ============================================================
# Save helper
# ============================================================
def save_figure(fig, basename):
    pdf_path = os.path.join(OUTPUT_DIR, f"{basename}.pdf")
    png_path = os.path.join(OUTPUT_DIR, f"{basename}.png")
    fig.canvas.draw()
    fig.savefig(pdf_path)
    fig.savefig(png_path)
    plt.close(fig)
    pdf_kb = os.path.getsize(pdf_path) / 1024
    png_kb = os.path.getsize(png_path) / 1024
    status = "OK" if pdf_kb > 30 else "!! SMALL -- possibly empty"
    print(f"    -> {basename}.pdf ({pdf_kb:.1f} KB), .png ({png_kb:.1f} KB)  [{status}]")


# ============================================================
# Helpers
# ============================================================
def generate_normalized_metric(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Q = A.T @ A + 0.5 * np.eye(n)
    return Q * (n / np.trace(Q))


def get_candidate_vectors(n, max_norm=5):
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
    eps_samples = np.linspace(eps_range[0], eps_range[1], n_samples)
    eigvals = np.array([np.linalg.eigvalsh(G + eps * Q) for eps in eps_samples])
    mu = eigvals.min(); M = eigvals.max()
    if mu <= 1e-12:
        return None
    return max(1, int(np.ceil(np.sqrt(M / mu))))


def compute_lambda_fast(g_eps, Ms):
    g_inv = np.linalg.inv(g_eps)
    return 4 * np.pi ** 2 * np.min(np.einsum('ki,ij,kj->k', Ms, g_inv, Ms))


def compute_branch(g_eps, m):
    """Evaluate a single spectral branch E_m(eps) = 4*pi^2 * m^T g^{-1} m."""
    g_inv = np.linalg.inv(g_eps)
    return 4 * np.pi ** 2 * (m @ g_inv @ m)


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
# Figure 1
# ============================================================
def fig_analytic_example():
    print("[1/6] Figure 1 (analytic example)...")
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
    save_figure(fig, 'fig_1_analytic')


# ============================================================
# Figure 2 (unchanged from v6)
# ============================================================
def fig_spectral_min():
    print("[2/6] Figure 2 (spectral min + band)...")
    fig, axes = plt.subplots(1, 3, figsize=(13, 8))
    n_real = 50
    eps_grid = np.linspace(0, 3, 200)

    for idx, n in enumerate([2, 3, 4]):
        ax = axes[idx]
        G = np.diag(np.arange(1, n + 1, dtype=float))

        Q_list = []
        max_L = 3
        for r in range(n_real):
            rng = np.random.default_rng(RANDOM_SEED + idx * 1000 + r)
            Q_r = np.zeros((n, n))
            Q_r[0, 0] = max(0.5, 2.0 + 0.15 * rng.standard_normal())
            Q_list.append(Q_r)
            L_r = compute_certified_L(G, Q_r, eps_range=(0.0, 3.0))
            if L_r is not None:
                max_L = max(max_L, L_r)

        Ms_ref = get_candidate_vectors(n, max_norm=max_L)
        print(f"      n={n}: max_L={max_L}, |M|={len(Ms_ref)}")

        Q_ref = Q_list[0]
        lam_ref = np.array([compute_lambda_fast(G + eps * Q_ref, Ms_ref)
                            for eps in eps_grid])

        all_lambdas = np.zeros((n_real, len(eps_grid)))
        for r, Q_r in enumerate(Q_list):
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
    save_figure(fig, 'fig_2_spectral_band')


# ============================================================
# Figure 3  [FIXED: crossing at 19/24, not 19/8]
# ============================================================
def fig_multiple_paths():
    print("[3/6] Figure 3 (multiple paths, FIXED crossing 19/24)...")
    n = 3
    G = np.eye(n)
    eps_grid = np.linspace(0, 3, 300)

    rng_diag = np.random.default_rng(RANDOM_SEED + 2000)
    diag_entries = rng_diag.uniform(0.1, 1.0, n)

    # Rank-one PSD with a = [1, 3, -3], trace normalized to n.
    # Analitically, the branches e_2 and e_2 - e_3 cross at eps = 19/24.
    a = np.array([1.0, 3.0, -3.0])
    Q_rank1 = n * np.outer(a, a) / (a @ a)

    paths = {
        'Random SPD': generate_normalized_metric(n, seed=RANDOM_SEED),
        'Diagonal (positive)': np.diag(diag_entries),
        'Rank-one PSD': Q_rank1,
    }

    L_max = 3
    for Q in paths.values():
        L = compute_certified_L(G, Q, eps_range=(0.0, 3.0))
        if L is not None:
            L_max = max(L_max, L)
    Ms = get_candidate_vectors(n, max_norm=L_max)

    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot the three minima
    for name, Q in paths.items():
        min_eig = min(np.linalg.eigvalsh(G + eps * Q).min() for eps in eps_grid)
        if min_eig <= 1e-10:
            print(f"      WARNING: path '{name}' becomes singular; skipped.")
            continue
        lambdas = [compute_lambda_fast(G + eps * Q, Ms) for eps in eps_grid]
        ax.plot(eps_grid, lambdas, linewidth=1.8, label=name)

    # [NEW] Show the two active branches along the rank-one path,
    # to make the exchange visible.
    e2 = np.array([0, 1, 0])
    e2m3 = np.array([0, 1, -1])
    branches_e2 = [compute_branch(G + eps * Q_rank1, e2) for eps in eps_grid]
    branches_e2m3 = [compute_branch(G + eps * Q_rank1, e2m3) for eps in eps_grid]
    ax.plot(eps_grid, branches_e2, color='gray', linestyle=':', linewidth=1.0,
            alpha=0.7, label=r'$E_{e_2}$ (rank-one)')
    ax.plot(eps_grid, branches_e2m3, color='gray', linestyle='--', linewidth=1.0,
            alpha=0.7, label=r'$E_{e_2-e_3}$ (rank-one)')

    # [FIXED] Crossing at eps = 19/24 (not 19/8)
    ax.axvline(EPS_RANK_ONE_CROSS, color='red', linestyle=':', linewidth=1.8,
               alpha=0.85,
               label=rf'transverse crossing at $\varepsilon = 19/24 \approx {EPS_RANK_ONE_CROSS:.3f}$')

    # Annotate the kink point
    lam_at_cross = 4 * np.pi ** 2 * ((19 + 30*EPS_RANK_ONE_CROSS) /
                                     (19 * (1 + 3*EPS_RANK_ONE_CROSS)))
    ax.scatter([EPS_RANK_ONE_CROSS], [lam_at_cross], color='red', s=50, zorder=5)

    ax.set_xlabel(r'$\varepsilon$')
    ax.set_ylabel(r'$\Lambda(\varepsilon)$')
    ax.set_title(r'Spectral minimum for three deformation paths ($n=3$)')
    ax.legend(loc='upper right', fontsize=9, ncol=1, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 3)

    # Small text annotation about the kink type
    ax.text(1.15, lam_at_cross + 4.5,
            'monotone kink\n(both one-sided\nderivatives negative)',
            fontsize=9, color='red', alpha=0.85,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='red', alpha=0.7))

    plt.tight_layout()
    save_figure(fig, 'fig_3_multiple_paths')


# ============================================================
# Figure 4 (unchanged from v6)
# ============================================================
def fig_bottleneck():
    print("[4/6] Figure 4 (bottleneck + C_met)...")
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

    fig, ax = plt.subplots(figsize=(9, 7))
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
    save_figure(fig, 'fig_4_bottleneck')


# ============================================================
# Figure 5 (unchanged from v6)
# ============================================================
def fig_noise_robustness():
    print("[5/6] Figure 5 (noise robustness)...")
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
    save_figure(fig, 'fig_5_noise')


# ============================================================
# Figure 6 (unchanged from v6)
# ============================================================
def fig_descriptor_components():
    print("[6/6] Figure 6 (descriptor components)...")
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
    save_figure(fig, 'fig_6_descriptor')


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Starting Figure Generation (v7)...\n")
    fig_analytic_example()
    fig_spectral_min()
    fig_multiple_paths()
    fig_bottleneck()
    fig_noise_robustness()
    fig_descriptor_components()
    print(f"\nAll figures saved in: {os.path.abspath(OUTPUT_DIR)}")
    print("\nFiles (PDF + PNG):")
    for f in sorted(Path(OUTPUT_DIR).glob('fig_*')):
        print(f"  {f.name}: {f.stat().st_size:,} bytes")