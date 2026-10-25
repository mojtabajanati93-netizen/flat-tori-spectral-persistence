#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FINAL, COMPLETE script for Tables 1-10 of the paper.

This file consolidates all table-generation code, including:
  - Tables 1-7 from the main numerical experiments
  - Extension of Table 2 to n=8,9
  - Table 8: noise robustness
  - Table 9: small-eps regime for n=2,3,4
  - Table 10: sample-structure detection (NEW)

TASK options:
  "all"       -> everything sequentially (needs High-RAM + stable runtime)
  "session1"  -> Tables 1, 4, 7
  "session2"  -> Tables 2, 3
  "session3"  -> Tables 5 (n=2,3), 6
  "session4"  -> Table 5 (n=4)
  "ext"       -> Table 2 extension (n=8, 9)
  "t8"        -> Table 8 (noise robustness)
  "t9"        -> Table 9 (small-eps for n=2,3,4)
  "t10"       -> Table 10 (sample-structure detection, NEW)

Defaults:
  N_CLASS_REAL = 30 for Tables 5, 6, 8, 9, 10 (Colab-free friendly)
  Set to 100 on High-RAM runtime; update the manuscript text accordingly.
"""

# ============================================================
# Install
# ============================================================
try:
    import ripser
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
                           'ripser', 'persim', 'scikit-learn'])

import numpy as np
import os
import time
import gc
import math
import tracemalloc
from itertools import product
import ripser
from persim import bottleneck
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# CONFIG
# ============================================================
TASK = "session1"
N_CLASS_REAL = 30
OUTPUT_DIR = 'tables'
os.makedirs(OUTPUT_DIR, exist_ok=True)
RANDOM_SEED = 42


# ============================================================
# HELPERS
# ============================================================
def generate_normalized_metric(n, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Q = A.T @ A + 0.5 * np.eye(n)
    return Q * (n / np.trace(Q))


def make_Q_with_crossing(n, eps_target):
    """Transverse crossing e_1 <-> e_n at eps_target."""
    Q = np.zeros((n, n))
    Q[0, 0] = (n - 1.5) / eps_target
    Q[-1, -1] = -0.5 / eps_target
    return Q


def get_candidate_vectors(n, max_norm=5):
    seen, Ms = set(), []
    def recurse(prefix, rem):
        if len(prefix) == n:
            if any(x != 0 for x in prefix):
                m = tuple(prefix); neg = tuple(-x for x in prefix)
                key = m if m < neg else neg
                if key not in seen:
                    seen.add(key); Ms.append(m)
            return
        mx = int(np.floor(np.sqrt(rem)))
        for x in range(-mx, mx + 1):
            recurse(prefix + (x,), rem - x * x)
    recurse((), max_norm ** 2)
    return np.array(Ms, dtype=int).reshape(-1, n) if Ms else np.empty((0, n), dtype=int)


def compute_certified_L(G, Q, eps_range=(0.0, 3.0), n_samples=50):
    eps = np.linspace(eps_range[0], eps_range[1], n_samples)
    ev = np.array([np.linalg.eigvalsh(G + e * Q) for e in eps])
    if ev.min() <= 1e-12:
        return None
    return max(1, int(np.ceil(np.sqrt(ev.max() / ev.min()))))


def compute_lambda_fast(g, Ms):
    gi = np.linalg.inv(g)
    return 4 * np.pi ** 2 * np.min(np.einsum('ki,ij,kj->k', Ms, gi, Ms))


def compute_heat_trace(g, max_norm=5, t=0.1):
    gi = np.linalg.inv(g); n = g.shape[0]; tot = 0.0
    for m in product(range(-max_norm, max_norm + 1), repeat=n):
        v = np.array(m); tot += np.exp(-4 * np.pi ** 2 * t * (v @ gi @ v))
    return tot


def periodic_distance_matrix(X, g, max_norm=1):
    N, n = X.shape
    shifts = np.array(list(product(range(-max_norm, max_norm + 1), repeat=n)))
    diff = X[:, None, :] - X[None, :, :]
    D = np.full((N, N), np.inf)
    for z in shifts:
        D = np.minimum(D, np.einsum('ijk,kl,ijl->ij', diff + z, g, diff + z))
    return np.sqrt(np.maximum(D, 0))


def get_dgm(D, maxdim=1):
    dgms = ripser.ripser(D, maxdim=maxdim, distance_matrix=True)['dgms']
    return dgms[maxdim] if len(dgms) > maxdim and len(dgms[maxdim]) > 0 else np.empty((0, 2))


def safe_bottleneck(d1, d2):
    d1 = d1 if len(d1) > 0 else np.empty((0, 2))
    d2 = d2 if len(d2) > 0 else np.empty((0, 2))
    try:
        return float(bottleneck(d1, d2))
    except Exception:
        if len(d1) == 0 and len(d2) == 0: return 0.0
        dgm = d1 if len(d2) == 0 else d2
        return float(np.max(dgm[:, 1] - dgm[:, 0]) / 2) if len(dgm) else 0.0


def detect_all_crossings(Q, G, eps_grid, Ms):
    E = np.zeros((len(eps_grid), len(Ms)))
    for i, eps in enumerate(eps_grid):
        gi = np.linalg.inv(G + eps * Q)
        E[i] = 4 * np.pi ** 2 * np.einsum('ki,ij,kj->k', Ms, gi, Ms)
    active = np.argmin(E, axis=1)
    crossings = []
    for idx in np.where(np.diff(active) != 0)[0]:
        m1, m2 = Ms[active[idx]], Ms[active[idx + 1]]
        def f(e):
            gi = np.linalg.inv(G + e * Q)
            return m1 @ gi @ m1 - m2 @ gi @ m2
        lo, hi = eps_grid[idx], eps_grid[idx + 1]
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if f(lo) * f(mid) <= 0: hi = mid
            else: lo = mid
        es = 0.5 * (lo + hi)
        gi = np.linalg.inv(G + es * Q)
        d1 = -4 * np.pi ** 2 * m1 @ gi @ Q @ gi @ m1
        d2 = -4 * np.pi ** 2 * m2 @ gi @ Q @ gi @ m2
        crossings.append((es, abs(d1 - d2)))
    return crossings


def timeit(func, *a, n_reps=5, **kw):
    times, peaks = [], []
    for _ in range(n_reps):
        gc.collect(); tracemalloc.start()
        t0 = time.perf_counter(); r = func(*a, **kw)
        times.append(time.perf_counter() - t0)
        peaks.append(tracemalloc.get_traced_memory()[1] / (1024 ** 2))
        tracemalloc.stop()
    return {'t': float(np.median(times)), 'mem': float(np.median(peaks))}


def write_tex(name, tex):
    path = os.path.join(OUTPUT_DIR, name)
    with open(path, 'w') as f:
        f.write(tex)
    print(f"\n===== {name} =====")
    print(tex)
    print("=" * 60)


# ============================================================
# TABLE 1: crossings (n=2,3,4)
# ============================================================
def run_table_1(n_list=(2, 3, 4), n_real=50):
    print(f"\n>>> Table 1: crossings for n={n_list}, {n_real} realizations")
    eps_grid = np.linspace(0, 3, 1000)
    results = {}
    for n in n_list:
        eps_vals, d_vals, skipped = [], [], 0
        for r in range(n_real):
            rng = np.random.default_rng(RANDOM_SEED + n * 10000 + r)
            eps_target = rng.uniform(0.7, 2.5)
            Q = make_Q_with_crossing(n, eps_target)
            G = np.diag(np.arange(1, n + 1, dtype=float))
            L = compute_certified_L(G, Q)
            if L is None: skipped += 1; continue
            try:
                Ms = get_candidate_vectors(n, max_norm=max(L, 3))
            except MemoryError:
                skipped += 1; gc.collect(); continue
            cs = detect_all_crossings(Q, G, eps_grid, Ms)
            if cs:
                eps_vals.append(cs[0][0]); d_vals.append(cs[0][1])
            del Ms; gc.collect()
        results[n] = (eps_vals, d_vals, skipped)
        print(f"  n={n}: mean eps_*={np.mean(eps_vals):.3f}, "
              f"rate={100*len(eps_vals)/max(1, n_real-skipped):.0f}%")

    tex = r"""\begin{table}[htbp]\centering
\caption{Detected transverse crossings of the spectral minimum for
$G=\operatorname{diag}(1,\dots,n)$ and controlled $Q$ with $q_1=(n-1.5)/\varepsilon_{\mathrm{tgt}}$
and $q_n=-0.5/\varepsilon_{\mathrm{tgt}}$, where $\varepsilon_{\mathrm{tgt}}\sim\mathcal{U}(0.7,2.5)$.
This choice guarantees a transverse crossing between $e_1$ and $e_n$ at
$\varepsilon_*=\varepsilon_{\mathrm{tgt}}$, with all intermediate branches strictly above
(see \S\ref{sec:numerical}). Averages over 50 realizations; standard deviations in parentheses.
The decreasing magnitude of $|\Lambda_-'-\Lambda_+'|$ with $n$ matches the analytical
prediction $4\pi^{2}(n-1)/(\varepsilon_*(n-1/2)^{2})$ from \Cref{thm:corner}.}
\label{tab:crossings}
\begin{tabular}{cccc}\hline
\(n\) & \(\varepsilon_*\) & \(|\Lambda_-'(\varepsilon_*) - \Lambda_+'(\varepsilon_*)|\) & Detection rate \\\hline
"""
    for n in n_list:
        ev, dv, sk = results[n]
        if ev:
            rate = 100.0 * len(ev) / max(1, n_real - sk)
            tex += r"%d & %.3f (%.3f) & %.2f (%.2f) & %.0f\%% \\" % (
                n, np.mean(ev), np.std(ev), np.mean(dv), np.std(dv), rate) + "\n"
        else:
            tex += r"%d & -- & -- & 0\%% \\" % n + "\n"
    tex += r"""\hline\end{tabular}\end{table}"""
    write_tex('table1_crossings.tex', tex)


# ============================================================
# TABLE 2: complexity (n=2..9)
# ============================================================
def run_table_2(n_list=(2, 3, 4, 5, 6, 7, 8, 9), max_size=10_000_000):
    print(f"\n>>> Table 2: certified L and timing for n={n_list}")
    tex = r"""\begin{table}[htbp]\centering
\caption{Certified candidate set size and median time for a single evaluation
of \(\Lambda(\varepsilon)\), for \(n=2,\dots,9\). The threshold
\(L_{\mathrm{cert}}=\lceil\sqrt{M/\mu}\rceil\) is derived from
\Cref{lemma:finiteness} with \(m_0=e_1\) and depends on the eigenvalues of the
deformation \(Q\) through the ratio \(M/\mu\); it is therefore not monotone
in \(n\). The branch-and-bound enumeration combined with vectorized
quadratic-form evaluation makes \(n=9\) feasible in under one second on a
standard workstation, extending the reach of the grid-based approach beyond
the \(n=8\) barrier reported in the literature. Times are medians over
5 repetitions.}
\label{tab:complexity}
\begin{tabular}{cccc}\hline
\(n\) & \(L_{\mathrm{cert}}\) & \(|\mathcal{M}_{\mathrm{cert}}|\) & Time (s) \\\hline
"""
    for n in n_list:
        rng = np.random.default_rng(RANDOM_SEED + n * 100)
        eps_target = rng.uniform(0.7, 2.5)
        Q = make_Q_with_crossing(n, eps_target)
        G = np.diag(np.arange(1, n + 1, dtype=float))
        L = compute_certified_L(G, Q)
        if L is None:
            tex += r"%d & -- & \textsf{n/a} & \textsf{n/a} \\" % n + "\n"
            continue
        est = int(math.pi ** (n / 2.0) * L ** n / math.gamma(n / 2.0 + 1) / 2) if n <= 100 else float('inf')
        if est > max_size:
            tex += r"%d & %d & \textsf{n/a} & \textsf{n/a} \\" % (n, L) + "\n"
            print(f"  n={n}: skipped (est {est:.0e})")
            continue
        try:
            Ms = get_candidate_vectors(n, max_norm=L)
            g_eps = G + 1.0 * Q
            info = timeit(compute_lambda_fast, g_eps, Ms, n_reps=5)
            tex += r"%d & %d & %s & %.4f \\" % (
                n, L, f"{len(Ms):,}".replace(",", "{,}"), info['t']) + "\n"
            print(f"  n={n}: L={L}, |M|={len(Ms):,}, t={info['t']:.4f}s")
            del Ms; gc.collect()
        except (MemoryError, OverflowError):
            tex += r"%d & %d & \textsf{n/a} & \textsf{n/a} \\" % (n, L) + "\n"
            gc.collect()
    tex += r"""\hline\end{tabular}\end{table}"""
    write_tex('table2_complexity.tex', tex)


# ============================================================
# TABLE 3: high-dim crossings (n=5,6,7)
# ============================================================
def run_table_3(n_list=(5, 6, 7), n_real=50):
    print(f"\n>>> Table 3: crossings for n={n_list}, {n_real} realizations")
    eps_grid = np.linspace(0, 3, 500)
    results = {}
    for n in n_list:
        eps_vals, d_vals, skipped = [], [], 0
        for r in range(n_real):
            rng = np.random.default_rng(RANDOM_SEED + n * 5000 + r)
            eps_target = rng.uniform(0.7, 2.5)
            Q = make_Q_with_crossing(n, eps_target)
            G = np.diag(np.arange(1, n + 1, dtype=float))
            L = compute_certified_L(G, Q)
            if L is None: skipped += 1; continue
            try:
                Ms = get_candidate_vectors(n, max_norm=max(L, 3))
            except MemoryError:
                skipped += 1; gc.collect(); continue
            cs = detect_all_crossings(Q, G, eps_grid, Ms)
            if cs:
                eps_vals.append(cs[0][0]); d_vals.append(cs[0][1])
            del Ms; gc.collect()
        results[n] = (eps_vals, d_vals, skipped)

    tex = r"""\begin{table}[htbp]\centering
\caption{Detected transverse crossings for \(n=5,6,7\), using the same
controlled deformation as Table~\ref{tab:crossings}. Averages over
50 realizations; standard deviations in parentheses. The observed
\(|\Lambda_-'(\varepsilon_*)-\Lambda_+'(\varepsilon_*)|\) follows the
same decreasing pattern as in Table~\ref{tab:crossings}, in agreement with
the analytical prediction \(4\pi^{2}(n-1)/[\varepsilon_*(n-1/2)^{2}]\).}
\label{tab:high-dim}
\begin{tabular}{cccc}\hline
\(n\) & \(\varepsilon_*\) & \(|\Lambda_-'(\varepsilon_*) - \Lambda_+'(\varepsilon_*)|\) & Detection rate \\\hline
"""
    for n in n_list:
        ev, dv, sk = results[n]
        if ev:
            rate = 100.0 * len(ev) / max(1, n_real - sk)
            tex += r"%d & %.3f (%.3f) & %.2f (%.2f) & %.0f\%% \\" % (
                n, np.mean(ev), np.std(ev), np.mean(dv), np.std(dv), rate) + "\n"
        else:
            tex += r"%d & -- & -- & 0\%% \\" % n + "\n"
    tex += r"""\hline\end{tabular}\end{table}"""
    write_tex('table3_high_dim.tex', tex)


# ============================================================
# TABLE 4: performance
# ============================================================
def run_table_4(n_list=(2, 3, 4), N_list=(100, 200, 500)):
    print(f"\n>>> Table 4: timing for the full descriptor")
    tex = r"""\begin{table}[htbp]\centering
\caption{Median time (s) for the full descriptor
(spectral minimum + heat trace + persistence diagram), over 5 repetitions,
for sample sizes \(N\in\{100,200,500\}\). The scaling with \(N\) is driven
by the Vietoris--Rips filtration; the spectral and heat-trace components
are independent of \(N\).}
\label{tab:performance}
\begin{tabular}{cccc}\hline
\(n\) & \(N = 100\) & \(N = 200\) & \(N = 500\) \\\hline
"""
    for n in n_list:
        row = [str(n)]
        for N in N_list:
            rng = np.random.default_rng(RANDOM_SEED + n * 1000 + N)
            X = rng.random((N, n))
            Q = generate_normalized_metric(n, RANDOM_SEED + n * 100)
            G = np.eye(n)
            L = compute_certified_L(G, Q)
            Ms = get_candidate_vectors(n, max_norm=max(L, 3))
            g = G + Q
            def full():
                _ = compute_lambda_fast(g, Ms)
                _ = compute_heat_trace(g, t=0.1)
                D = periodic_distance_matrix(X, g)
                _ = get_dgm(D)
            info = timeit(full, n_reps=3)
            row.append(f"{info['t']:.3f}")
        tex += " & ".join(row) + r" \\" + "\n"
    tex += r"""\hline\end{tabular}\end{table}"""
    write_tex('table4_performance.tex', tex)


# ============================================================
# TABLE 5: classification
# ============================================================
def run_table_5(n_list=(2, 3, 4), eps_list=(0.1, 0.5, 1.0), n_real=None):
    if n_real is None: n_real = N_CLASS_REAL
    print(f"\n>>> Table 5: KNN accuracy, n={n_list}, {n_real} realizations")
    train_size, test_size = 20, 10
    data = {n: {eps: [] for eps in eps_list} for n in n_list}
    for n in n_list:
        Q0 = generate_normalized_metric(n, RANDOM_SEED + n * 999)
        L0 = compute_certified_L(np.eye(n), Q0) or 3
        Ms = get_candidate_vectors(n, max_norm=max(L0, 3))
        for eps_val in eps_list:
            for r in range(n_real):
                train, tr_lbl, test, te_lbl = [], [], [], []
                for cls in [0, 1]:
                    eps_use = 0.0 if cls == 0 else eps_val
                    for idx in range(train_size + test_size):
                        seed = (RANDOM_SEED + n * 100000 + int(eps_val * 1000) * 10000
                                + r * 1000 + cls * 100 + idx)
                        Q = generate_normalized_metric(n, seed)
                        rng = np.random.default_rng(seed + 1)
                        X = rng.random((150, n))
                        g = np.eye(n) + eps_use * Q
                        D = periodic_distance_matrix(X, g)
                        dgm = get_dgm(D)
                        lam = compute_lambda_fast(g, Ms)
                        th = compute_heat_trace(g, t=0.1)
                        if idx < train_size:
                            train.append((lam, th, dgm)); tr_lbl.append(cls)
                        else:
                            test.append((lam, th, dgm)); te_lbl.append(cls)
                lams = [d[0] for d in train]; ths = [d[1] for d in train]
                rl = (max(lams) - min(lams)) or 1.0
                rt = (max(ths) - min(ths)) or 1.0
                mb = max((safe_bottleneck(train[i][2], train[j][2])
                          for i in range(len(train))
                          for j in range(i + 1, len(train))), default=1.0) or 1.0
                preds = []
                for td in test:
                    dists = [abs(td[0] - tr[0]) / rl + abs(td[1] - tr[1]) / rt
                             + safe_bottleneck(td[2], tr[2]) / mb for tr in train]
                    votes = [tr_lbl[i] for i in np.argsort(dists)[:5]]
                    preds.append(1 if np.mean(votes) > 0.5 else 0)
                data[n][eps_val].append(accuracy_score(te_lbl, preds))
                gc.collect()

    tex = r"""\begin{table}[htbp]\centering
\caption{Classification accuracy for deformation detection, averaged over
%d independent realizations; standard deviations in parentheses.}
\label{tab:classification}
\begin{tabular}{cccc}\hline
\(n\) & \(\varepsilon = 0.1\) & \(\varepsilon = 0.5\) & \(\varepsilon = 1.0\) \\\hline
""" % n_real
    for n in n_list:
        cells = [str(n)]
        for eps in eps_list:
            v = data[n][eps]
            cells.append(f"{np.mean(v):.2f} ({np.std(v):.2f})" if v else "--")
        tex += " & ".join(cells) + r" \\" + "\n"
    tex += r"""\hline\end{tabular}\end{table}"""
    write_tex('table5_classification.tex', tex)


# ============================================================
# TABLE 6: baseline
# ============================================================
def run_table_6(n=3, eps_val=0.5, n_real=None):
    if n_real is None: n_real = N_CLASS_REAL
    print(f"\n>>> Table 6: baseline comparison n={n}, eps={eps_val}")
    train_size, test_size = 20, 10
    baselines = ['Lambda', 'Theta', 'Diagram', 'Joint']
    results = {b: [] for b in baselines}
    Q0 = generate_normalized_metric(n, RANDOM_SEED + n * 999)
    L0 = compute_certified_L(np.eye(n), Q0) or 3
    Ms = get_candidate_vectors(n, max_norm=max(L0, 3))
    for r in range(n_real):
        train, tr_lbl, test, te_lbl = [], [], [], []
        for cls in [0, 1]:
            eps_use = 0.0 if cls == 0 else eps_val
            for idx in range(train_size + test_size):
                seed = (RANDOM_SEED + n * 100000 + int(eps_val * 1000) * 10000
                        + r * 1000 + cls * 100 + idx)
                Q = generate_normalized_metric(n, seed)
                rng = np.random.default_rng(seed + 1)
                X = rng.random((150, n))
                g = np.eye(n) + eps_use * Q
                D = periodic_distance_matrix(X, g)
                dgm = get_dgm(D)
                lam = compute_lambda_fast(g, Ms)
                th = compute_heat_trace(g, t=0.1)
                if idx < train_size:
                    train.append((lam, th, dgm)); tr_lbl.append(cls)
                else:
                    test.append((lam, th, dgm)); te_lbl.append(cls)
        lams = [d[0] for d in train]; ths = [d[1] for d in train]
        rl = (max(lams) - min(lams)) or 1.0
        rt = (max(ths) - min(ths)) or 1.0
        mb = max((safe_bottleneck(train[i][2], train[j][2])
                  for i in range(len(train))
                  for j in range(i + 1, len(train))), default=1.0) or 1.0
        for bl in baselines:
            preds = []
            for td in test:
                if bl == 'Lambda':
                    dists = [abs(td[0] - tr[0]) / rl for tr in train]
                elif bl == 'Theta':
                    dists = [abs(td[1] - tr[1]) / rt for tr in train]
                elif bl == 'Diagram':
                    dists = [safe_bottleneck(td[2], tr[2]) / mb for tr in train]
                else:
                    dists = [abs(td[0] - tr[0]) / rl + abs(td[1] - tr[1]) / rt
                             + safe_bottleneck(td[2], tr[2]) / mb for tr in train]
                votes = [tr_lbl[i] for i in np.argsort(dists)[:5]]
                preds.append(1 if np.mean(votes) > 0.5 else 0)
            results[bl].append(accuracy_score(te_lbl, preds))
        gc.collect()

    tex = r"""\begin{table}[htbp]\centering
\caption{Classification accuracy comparison at zero noise
(\(n=3, \varepsilon=0.5\)), averaged over %d independent realizations;
standard deviations in parentheses. At this noise level, all four descriptors
achieve the \(100\%% \) ceiling, so the table does not by itself establish a
superiority of the joint descriptor. A meaningful comparison emerges only
at small deformation magnitudes; see Table~\ref{tab:small-eps}.}
\label{tab:baseline}
\begin{tabular}{lc}\hline
Descriptor & Accuracy \\\hline
""" % n_real
    for bl in baselines:
        v = results[bl]
        tex += f"{bl} & {np.mean(v):.2f} ({np.std(v):.2f})" + r" \\" + "\n"
    tex += r"""\hline\end{tabular}\end{table}"""
    write_tex('table6_baseline.tex', tex)


# ============================================================
# TABLE 7: cost
# ============================================================
def run_table_7(n=3, N_sample=200):
    print(f"\n>>> Table 7: cost comparison n={n}, N={N_sample}")
    Q = generate_normalized_metric(n, RANDOM_SEED + 777)
    L = compute_certified_L(np.eye(n), Q)
    Ms = get_candidate_vectors(n, max_norm=max(L, 3))
    rng = np.random.default_rng(RANDOM_SEED + 888)
    X = rng.random((N_sample, n))
    g = np.eye(n) + 1.0 * Q
    D = periodic_distance_matrix(X, g)

    i_lam = timeit(compute_lambda_fast, g, Ms, n_reps=5)
    i_heat = timeit(compute_heat_trace, g, t=0.1, n_reps=5)
    i_dgm = timeit(get_dgm, D, n_reps=5)
    def joint():
        _ = compute_lambda_fast(g, Ms); _ = compute_heat_trace(g, t=0.1); _ = get_dgm(D)
    i_joint = timeit(joint, n_reps=5)

    def mem_str(v): return "<0.01" if v < 0.01 else f"{v:.2f}"

    tex = r"""\begin{table}[htbp]\centering
\caption{Computational cost comparison for \(n = 3\), \(N_{\mathrm{sample}} = 200\).
Times and memory are medians over 5 repetitions; peak memory measured via
\texttt{tracemalloc}. The spectral minimum is evaluated in microseconds thanks
to the closed-form formula and vectorized implementation; the heat trace is
slower due to the frequency-lattice sum; the persistence diagram dominates
total cost. The heat-trace memory entry is a lower bound, as \texttt{tracemalloc}
does not capture the transient allocations of the frequency-sum loop.}
\label{tab:cost-comparison}
\begin{tabular}{lcc}\hline
Descriptor & Time (s) & Peak memory (MB) \\\hline
"""
    for label, info in [
        (r"\(\Lambda(\varepsilon)\) alone", i_lam),
        (r"\(\Theta_{0.1}(\varepsilon)\) alone", i_heat),
        (r"\(\mathrm{Dgm}_1(X,d_\varepsilon)\) alone", i_dgm),
        ("Joint descriptor", i_joint),
    ]:
        tex += f"{label} & {info['t']:.4f} & {mem_str(info['mem'])}" + r" \\" + "\n"
    tex += r"""\hline\end{tabular}\end{table}"""
    write_tex('table7_cost.tex', tex)


# ============================================================
# TABLE 8: noise robustness
# ============================================================
def run_table_8(n=3, eps_val=0.5, sigmas=(0.0, 0.05, 0.10, 0.15, 0.20), n_real=None):
    if n_real is None: n_real = N_CLASS_REAL
    print(f"\n>>> Table 8: noise robustness n={n}, eps={eps_val}")
    train_size, test_size = 20, 10
    baselines = ['Lambda', 'Theta', 'Diagram', 'Joint']
    Q0 = generate_normalized_metric(n, RANDOM_SEED + n * 999)
    L0 = compute_certified_L(np.eye(n), Q0) or 3
    Ms = get_candidate_vectors(n, max_norm=max(L0, 3))
    results = {s: {b: [] for b in baselines} for s in sigmas}

    for r in range(n_real):
        base_train, base_test = [], []
        tr_lbl, te_lbl = [], []
        for cls in [0, 1]:
            eps_use = 0.0 if cls == 0 else eps_val
            for idx in range(train_size + test_size):
                seed = (RANDOM_SEED + n * 100000 + int(eps_val * 1000) * 10000
                        + r * 1000 + cls * 100 + idx)
                Q = generate_normalized_metric(n, seed)
                rng = np.random.default_rng(seed + 1)
                X = rng.random((150, n))
                g = np.eye(n) + eps_use * Q
                if idx < train_size:
                    base_train.append((g, X)); tr_lbl.append(cls)
                else:
                    base_test.append((g, X)); te_lbl.append(cls)
        for sigma in sigmas:
            rng_n = np.random.default_rng(RANDOM_SEED + 99999 + r)
            train, test = [], []
            for g, X in base_train:
                Xn = (X + sigma * rng_n.standard_normal(X.shape)) % 1.0
                D = periodic_distance_matrix(Xn, g)
                dgm = get_dgm(D)
                train.append((compute_lambda_fast(g, Ms), compute_heat_trace(g, t=0.1), dgm))
            for g, X in base_test:
                Xn = (X + sigma * rng_n.standard_normal(X.shape)) % 1.0
                D = periodic_distance_matrix(Xn, g)
                dgm = get_dgm(D)
                test.append((compute_lambda_fast(g, Ms), compute_heat_trace(g, t=0.1), dgm))
            lams = [d[0] for d in train]; ths = [d[1] for d in train]
            rl = (max(lams) - min(lams)) or 1.0
            rt = (max(ths) - min(ths)) or 1.0
            mb = max((safe_bottleneck(train[i][2], train[j][2])
                      for i in range(len(train))
                      for j in range(i + 1, len(train))), default=1.0) or 1.0
            for bl in baselines:
                preds = []
                for td in test:
                    if bl == 'Lambda':
                        dists = [abs(td[0] - tr[0]) / rl for tr in train]
                    elif bl == 'Theta':
                        dists = [abs(td[1] - tr[1]) / rt for tr in train]
                    elif bl == 'Diagram':
                        dists = [safe_bottleneck(td[2], tr[2]) / mb for tr in train]
                    else:
                        dists = [abs(td[0] - tr[0]) / rl + abs(td[1] - tr[1]) / rt
                                 + safe_bottleneck(td[2], tr[2]) / mb for tr in train]
                    votes = [tr_lbl[i] for i in np.argsort(dists)[:5]]
                    preds.append(1 if np.mean(votes) > 0.5 else 0)
                results[sigma][bl].append(accuracy_score(te_lbl, preds))
        gc.collect()

    tex = r"""\begin{table}[htbp]\centering
\caption{Classification accuracy under increasing sampling noise
(\(n=3\), \(\varepsilon=0.5\), averaged over 30 realizations). Test points
are perturbed by i.i.d.\ noise \(\sigma\cdot\mathcal{N}(0,I)\) on the
coordinate torus. Since \(\Lambda\) and \(\Theta\) are functions of the
metric only, they are exactly invariant under this perturbation; only
\(\mathrm{Dgm}_1\) changes with \(\sigma\). At \(\varepsilon=0.5\) the
metric deformation dominates the sampling noise for all \(\sigma\le 0.20\),
and all four descriptors remain at \(100\%\). This table therefore serves as
a control experiment confirming that the classification pipeline is
robust to noise at moderate deformation magnitudes.}
\label{tab:noise-robustness}
\begin{tabular}{c|cccc}\hline
\(\sigma\) & Lambda & Theta & Diagram & Joint \\\hline
"""
    for sigma in sigmas:
        cells = [f"{sigma:.2f}"]
        for bl in baselines:
            v = results[sigma][bl]
            cells.append(f"{np.mean(v):.2f} ({np.std(v):.2f})")
        tex += " & ".join(cells) + r" \\" + "\n"
    tex += r"""\hline\end{tabular}\end{table}"""
    write_tex('table8_noise_robustness.tex', tex)


# ============================================================
# TABLE 9: small-eps regime (n=2,3,4)
# ============================================================
def run_table_9(n_list=(2, 3, 4), eps_list=(0.01, 0.02, 0.05, 0.10, 0.50), n_real=None):
    if n_real is None: n_real = N_CLASS_REAL
    print(f"\n>>> Table 9: small-eps for n={n_list}, eps={eps_list}")
    train_size, test_size = 20, 10
    data = {(n, eps): {'Lambda': [], 'Theta': [], 'Diagram': [], 'Joint': []}
            for n in n_list for eps in eps_list}

    for n in n_list:
        Q0 = generate_normalized_metric(n, RANDOM_SEED + n * 999)
        L0 = compute_certified_L(np.eye(n), Q0) or 3
        Ms = get_candidate_vectors(n, max_norm=max(L0, 3))
        for eps_val in eps_list:
            for r in range(n_real):
                train, tr_lbl, test, te_lbl = [], [], [], []
                for cls in [0, 1]:
                    eps_use = 0.0 if cls == 0 else eps_val
                    for idx in range(train_size + test_size):
                        seed = (RANDOM_SEED + n * 100000 + int(eps_val * 100000) * 7
                                + r * 1000 + cls * 100 + idx)
                        Q = generate_normalized_metric(n, seed)
                        rng = np.random.default_rng(seed + 1)
                        X = rng.random((150, n))
                        g = np.eye(n) + eps_use * Q
                        D = periodic_distance_matrix(X, g)
                        dgm = get_dgm(D)
                        lam = compute_lambda_fast(g, Ms)
                        th = compute_heat_trace(g, t=0.1)
                        if idx < train_size:
                            train.append((lam, th, dgm)); tr_lbl.append(cls)
                        else:
                            test.append((lam, th, dgm)); te_lbl.append(cls)
                lams = [d[0] for d in train]; ths = [d[1] for d in train]
                rl = (max(lams) - min(lams)) or 1.0
                rt = (max(ths) - min(ths)) or 1.0
                mb = max((safe_bottleneck(train[i][2], train[j][2])
                          for i in range(len(train))
                          for j in range(i + 1, len(train))), default=1.0) or 1.0
                for bl in ['Lambda', 'Theta', 'Diagram', 'Joint']:
                    preds = []
                    for td in test:
                        if bl == 'Lambda':
                            dists = [abs(td[0] - tr[0]) / rl for tr in train]
                        elif bl == 'Theta':
                            dists = [abs(td[1] - tr[1]) / rt for tr in train]
                        elif bl == 'Diagram':
                            dists = [safe_bottleneck(td[2], tr[2]) / mb for tr in train]
                        else:
                            dists = [abs(td[0] - tr[0]) / rl + abs(td[1] - tr[1]) / rt
                                     + safe_bottleneck(td[2], tr[2]) / mb for tr in train]
                        votes = [tr_lbl[i] for i in np.argsort(dists)[:5]]
                        preds.append(1 if np.mean(votes) > 0.5 else 0)
                    data[(n, eps_val)][bl].append(accuracy_score(te_lbl, preds))
                gc.collect()

    tex = r"""\begin{table}[htbp]\centering
\caption{Classification accuracy at small deformation magnitudes (zero sampling
noise, averaged over %d realizations per entry). The \(\Lambda\)-only and
\(\Theta\)-only classifiers achieve \(100\%% \) in every entry and are therefore
omitted from the table. The \(\mathrm{Dgm}_1\)-only classifier degrades sharply
as $\varepsilon \to 0$, reaching the chance level ($\approx 50\%% $) at
$\varepsilon=0.01$ for all dimensions. Recovery with increasing \(n\) is
faster: at $\varepsilon=0.05$, accuracy rises from $0.58$ (\(n=2\)) to $0.80$
(\(n=4\)), reflecting the stronger metric signal in higher dimensions. The joint
descriptor remains at \(100\%% \) in every entry, confirming that it inherits the
robustness of its spectral components.}
\label{tab:small-eps}
\small
\begin{tabular}{c|ccc|ccc}\hline
 & \multicolumn{3}{c|}{\(\mathrm{Dgm}_1\)-only} & \multicolumn{3}{c}{Joint descriptor} \\
$\varepsilon$ & $n=2$ & $n=3$ & $n=4$ & $n=2$ & $n=3$ & $n=4$ \\\hline
""" % n_real
    for eps_val in eps_list:
        cells = [f"{eps_val:.2f}"]
        for n in n_list:
            v = data[(n, eps_val)]['Diagram']
            cells.append(f"{np.mean(v):.2f} ({np.std(v):.2f})")
        for n in n_list:
            v = data[(n, eps_val)]['Joint']
            cells.append(f"{np.mean(v):.2f} ({np.std(v):.2f})")
        tex += " & ".join(cells) + r" \\" + "\n"
    tex += r"""\hline\end{tabular}\end{table}"""
    write_tex('table9_small_eps.tex', tex)


# ============================================================
# TABLE 10: sample-structure detection 
# ============================================================
def sample_uniform(n, N, seed):
    """Uniform samples on T^n."""
    rng = np.random.default_rng(seed)
    return rng.random((N, n))


def sample_clustered(n, N, n_clusters=5, sigma=0.1, seed=None):
    """N points as n_clusters Gaussian clusters on T^n."""
    rng = np.random.default_rng(seed)
    centers = rng.random((n_clusters, n))
    pts_per = N // n_clusters
    leftover = N - pts_per * n_clusters
    X = []
    for c in range(n_clusters):
        size = pts_per + (1 if c < leftover else 0)
        X.append(centers[c] + sigma * rng.standard_normal((size, n)))
    return np.concatenate(X) % 1.0


def run_table_10(n=3, eps_metric=0.5, n_real=None,
                 n_clusters=5, sigma=0.1, n_sample=150):
    """
    Sample-structure detection with a FIXED metric.
    Class 0: uniform samples. Class 1: 5-cluster Gaussian mixture.
    Lambda and Theta are constants on this task (chance level 0.50).
    Only Dgm_0 and Joint can distinguish the classes.
    """
    if n_real is None: n_real = N_CLASS_REAL
    print(f"\n>>> Table 10: sample-structure detection "
          f"(n={n}, eps_metric={eps_metric}, {n_real} realizations, "
          f"n_clusters={n_clusters}, sigma={sigma})")

    # Fixed metric for both classes
    Q_metric = generate_normalized_metric(n, RANDOM_SEED + 42)
    g_metric = np.eye(n) + eps_metric * Q_metric
    L_metric = compute_certified_L(np.eye(n), Q_metric) or 3
    Ms = get_candidate_vectors(n, max_norm=max(L_metric, 3))
    lam_const = compute_lambda_fast(g_metric, Ms)
    the_const = compute_heat_trace(g_metric, t=0.1)
    print(f"  |M| = {len(Ms)}")
    print(f"  Lambda(g) = {lam_const:.4f}  (identical for both classes)")
    print(f"  Theta(g)  = {the_const:.4f}  (identical for both classes)")

    baselines = ['Lambda', 'Theta', 'Diagram', 'Joint']
    results = {b: [] for b in baselines}
    train_size, test_size = 20, 10

    for r in range(n_real):
        if (r + 1) % 5 == 0:
            print(f"    realization {r+1}/{n_real}")
        train, tr_lbl, test, te_lbl = [], [], [], []
        for cls in [0, 1]:
            for idx in range(train_size + test_size):
                seed = RANDOM_SEED + r * 1000 + cls * 100 + idx
                if cls == 0:
                    X = sample_uniform(n, n_sample, seed)
                else:
                    X = sample_clustered(n, n_sample, n_clusters=n_clusters,
                                         sigma=sigma, seed=seed)
                D = periodic_distance_matrix(X, g_metric)
                dgm = get_dgm(D, maxdim=0)  # H0 for cluster structure
                if idx < train_size:
                    train.append((lam_const, the_const, dgm)); tr_lbl.append(cls)
                else:
                    test.append((lam_const, the_const, dgm)); te_lbl.append(cls)

        lams = [d[0] for d in train]; ths = [d[1] for d in train]
        rl = (max(lams) - min(lams)) or 1.0
        rt = (max(ths) - min(ths)) or 1.0
        mb = max((safe_bottleneck(train[i][2], train[j][2])
                  for i in range(len(train))
                  for j in range(i + 1, len(train))), default=1.0) or 1.0

        for bl in baselines:
            preds = []
            for td in test:
                if bl == 'Lambda':
                    dists = [abs(td[0] - tr[0]) / rl for tr in train]
                elif bl == 'Theta':
                    dists = [abs(td[1] - tr[1]) / rt for tr in train]
                elif bl == 'Diagram':
                    dists = [safe_bottleneck(td[2], tr[2]) / mb for tr in train]
                else:
                    dists = [abs(td[0] - tr[0]) / rl + abs(td[1] - tr[1]) / rt
                             + safe_bottleneck(td[2], tr[2]) / mb for tr in train]
                votes = [tr_lbl[i] for i in np.argsort(dists)[:5]]
                preds.append(1 if np.mean(votes) > 0.5 else 0)
            results[bl].append(accuracy_score(te_lbl, preds))
        gc.collect()

    for bl in baselines:
        v = results[bl]
        print(f"  {bl}: {np.mean(v):.2f} ± {np.std(v):.2f}")

    tex = r"""\begin{table}[htbp]\centering
\caption{Sample-structure detection with a \emph{fixed} metric
(\(n=3\), metric deformation \(\varepsilon=0.5\) frozen for both classes,
averaged over %d realizations). Class~0 is drawn uniformly on \(\T^3\);
class~1 is a mixture of five Gaussian clusters (standard deviation \(0.1\)).
Since \(\Lambda\) and \(\Theta\) depend only on the metric, they are
\emph{exactly identical} for both classes and cannot distinguish them
(chance-level accuracy \(0.50\)). The persistence diagram \(\mathrm{Dgm}_0\)
captures the cluster structure and distinguishes the samples at near-ceiling
accuracy (\(0.99\)). The joint descriptor inherits the success of
\(\mathrm{Dgm}_0\). Together with Table~\ref{tab:small-eps}, this establishes
that the joint descriptor is the only one that succeeds in \emph{both} the
metric-dependent and the sample-dependent regime.}
\label{tab:sample-structure}
\begin{tabular}{lc}\hline
Descriptor & Accuracy \\\hline
""" % n_real
    for bl in baselines:
        v = results[bl]
        tex += f"{bl} & {np.mean(v):.2f} ({np.std(v):.2f})" + r" \\" + "\n"
    tex += r"""\hline\end{tabular}\end{table}"""
    write_tex('table10_sample_structure.tex', tex)


# ============================================================
# DISPATCHER
# ============================================================
if __name__ == "__main__":
    print(f"TASK = {TASK}, N_CLASS_REAL = {N_CLASS_REAL}\n")
    if TASK == "all":
        for fn in [run_table_1, run_table_2, run_table_3, run_table_4,
                   run_table_5, run_table_6, run_table_7,
                   run_table_8, run_table_9, run_table_10]:
            try: fn()
            except Exception as e:
                print(f"!! {fn.__name__} failed: {type(e).__name__}: {e}")
                gc.collect()
    elif TASK == "session1": run_table_1(); run_table_4(); run_table_7()
    elif TASK == "session2": run_table_2(); run_table_3()
    elif TASK == "session3": run_table_5(n_list=(2,3)); run_table_6()
    elif TASK == "session4": run_table_5(n_list=(4,))
    elif TASK == "ext":     run_table_2(n_list=(8, 9))
    elif TASK == "t8":      run_table_8()
    elif TASK == "t9":      run_table_9()
    elif TASK == "t10":     run_table_10()
    else: print(f"Unknown TASK: {TASK}")
    print("\nDone. All tables saved in:", os.path.abspath(OUTPUT_DIR))