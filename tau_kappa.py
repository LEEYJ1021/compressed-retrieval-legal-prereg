#!/usr/bin/env python3
"""Compute tau(B/4,B), kappa and rho_SQ (Proposition 1) from fitted PCA spectra.
Also gives an empirical check on real query-chunk scores."""
import argparse, os
import numpy as np, pandas as pd

BUDGETS = [16, 32, 64, 96, 192, 384, 768]

def fit_pca(X, n_fit, seed):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n_fit, len(X)), replace=False)
    Xs = X[idx].astype(np.float64)
    mu = Xs.mean(0)
    Xc = Xs - mu
    cov = (Xc.T @ Xc) / (len(Xc) - 1)
    lam, V = np.linalg.eigh(cov)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    Ytr = Xc @ V                       # training PCA coordinates
    return mu, lam, V, Ytr, idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--model", required=True, choices=["A", "B"])
    ap.add_argument("--n_fit", type=int, default=100000)
    ap.add_argument("--n_q", type=int, default=1000)
    ap.add_argument("--n_x", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out_full/tau_kappa")
    a = ap.parse_args()

    X = np.load(f"{a.data_dir}/emb/{a.model}/chunks.npy").astype(np.float32)
    Q = np.load(f"{a.data_dir}/emb/{a.model}/queries.npy").astype(np.float32)
    N, D = X.shape
    print(f"[Model {a.model}] chunks {X.shape}, queries {Q.shape}")

    mu, lam, V, Ytr, fit_idx = fit_pca(X, a.n_fit, a.seed)
    lo, hi = Ytr.min(0), Ytr.max(0)          # SQ8 range l_j, u_j (training)
    rng_j = hi - lo
    kappa_j = rng_j / np.sqrt(lam)           # per-coordinate kappa_j
    l2 = lam ** 2
    cum = np.cumsum(l2)
    total = cum[-1]

    # query-side spectrum in the chunk PCA basis (relaxes "q ~ x" in A1)
    Yq_all = (Q.astype(np.float64) - mu) @ V
    lam_q = (Yq_all ** 2).mean(0)

    # held-out chunks for the empirical check
    rest = np.setdiff1d(np.arange(N), fit_idx)
    r = np.random.default_rng(a.seed + 1)
    xi = r.choice(rest if len(rest) >= a.n_x else np.arange(N), size=a.n_x, replace=False)
    qi = r.choice(len(Q), size=min(a.n_q, len(Q)), replace=False)
    Yx = (X[xi].astype(np.float64) - mu) @ V
    Yq = Yq_all[qi]

    rows = []
    for B in BUDGETS:
        d2, d1 = B, B // 4
        if d2 > D:
            continue
        # --- analytic (Eq. 9, 10) ---
        tau = (cum[d2 - 1] - cum[d1 - 1]) / cum[d2 - 1]
        delta = rng_j[:d2] / 255.0
        rho_exact = np.sum(lam[:d2] * delta ** 2 / 12) / np.sum(l2[:d2])   # no A3
        kap = kappa_j[:d2]
        kappa_eff = np.sqrt(12 * 255 ** 2 * rho_exact)                      # kappa solving Eq.9
        rho_nominal = 81 / (12 * 255 ** 2)                                  # kappa = 9
        # query-aware variant: Var(s_d) = sum lam_q*lam_x
        cq = np.cumsum(lam_q * lam)
        tau_q = (cq[d2 - 1] - cq[d1 - 1]) / cq[d2 - 1]
        rho_q = np.sum(lam_q[:d2] * lam[:d2] * (rng_j[:d2] / 255) ** 2 / 12) / cq[d2 - 1]
        # --- empirical check on real scores ---
        s2 = Yq[:, :d2] @ Yx[:, :d2].T
        s1 = Yq[:, :d1] @ Yx[:, :d1].T
        step = rng_j[:d2] / 255.0
        code = np.clip(np.round((Yx[:, :d2] - lo[:d2]) / step), 0, 255)
        Yx_q = lo[:d2] + step * code
        s_sq = Yq[:, :d2] @ Yx_q.T
        tau_emp = np.var(s2 - s1) / np.var(s2)
        rho_emp = np.var(s_sq - s2) / np.var(s2)
        rows.append(dict(
            model=a.model, B=B, d_f32=d1, d_sq8=d2,
            tau=tau, tau_query_aware=tau_q, tau_empirical=tau_emp,
            E_d1=cum[d1 - 1] / total, E_d2=cum[d2 - 1] / total,
            kappa_mean=kap.mean(), kappa_median=np.median(kap), kappa_max=kap.max(),
            kappa_eff=kappa_eff,
            rho_nominal_k9=rho_nominal, rho_exact=rho_exact, rho_query_aware=rho_q,
            rho_empirical=rho_emp,
            rho_exact_dB=10 * np.log10(rho_exact), rho_empirical_dB=10 * np.log10(rho_emp),
            tau_over_rho=tau / rho_exact, bits_win=bool(tau > rho_exact)))
    df = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    df.to_csv(f"{a.out}/tau_kappa_{a.model}.csv", index=False)
    pd.DataFrame({"j": np.arange(1, D + 1), "lambda": lam, "kappa_j": kappa_j}) \
        .to_csv(f"{a.out}/spectrum_{a.model}.csv", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(df[["B", "tau", "tau_query_aware", "tau_empirical", "kappa_median",
              "kappa_max", "kappa_eff", "rho_exact", "rho_empirical",
              "rho_exact_dB", "tau_over_rho"]].round(5).to_string(index=False))
    print(f"\nAll budgets tau > rho_exact: {df.bits_win.all()}")
    print(f"Saved to {a.out}/tau_kappa_{a.model}.csv")

if __name__ == "__main__":
    main()
