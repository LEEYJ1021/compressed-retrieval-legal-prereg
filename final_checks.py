import glob, math, os, sys, faiss, numpy as np, pandas as pd
from evalcore import *
faiss.omp_set_num_threads(8)
N_CH = 160340; SEEDS = [0, 1, 2]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
fmt = lambda t: f"{t[0]:+.4f} [{t[1]:+.4f},{t[2]:+.4f}]"
for model in ["A", "B"]:
    R = f"out_full/supp2/{model}_std"
    src = np.load(f"{R}/src.npy", allow_pickle=True); cont = np.load(f"{R}/cont.npy", allow_pickle=True)
    raw = np.load(f"{R}/raw.npy"); D = {"A": 768, "B": 1024}[model]
    def pm(pattern):
        fs = sorted(glob.glob(f"{R}/{pattern}")); return np.mean([np.load(f) for f in fs], 0) if fs else None
    def src_ret(v):
        return {s: float(v[src == s].mean() / raw[src == s].mean()) for s in SRC3}
    def worst_ci(v, nb=1000):
        rng = np.random.default_rng(20261004); pre = {}
        for s in SRC3:
            m = src == s; _, inv = np.unique(cont[m], return_inverse=True)
            pre[s] = (np.bincount(inv, weights=v[m]), np.bincount(inv, weights=raw[m]))
        reps = np.empty(nb)
        for b in range(nb):
            r = []
            for s in SRC3:
                a, c = pre[s]; k = rng.integers(0, len(a), len(a)); r.append(a[k].sum() / c[k].sum())
            reps[b] = min(r)
        return float(np.percentile(reps, 2.5))
    # ---- 1. worst-source retention
    rows = []
    cfgs = [(f"sq8_B{B}", pm(f"lad_s*_sq8_{B}.npy")) for B in [96, 192, 384, 768]] + \
           [(f"sq4_B{B}", pm(f"lad_s*_sq4_{B}.npy")) for B in [96, 192, 384]] + \
           [(f"pq_m{m}", pm(f"pq_{m}_s*.npy")) for m in [32, 64, 128, 256]] + \
           [(f"sq4m_m{m}", pm(f"sq4m_{m}_s*.npy")) for m in [32, 64, 128, 256]]
    for name, v in cfgs:
        if v is None: continue
        r = src_ret(v); rows.append(dict(model=model, config=name, **{k: round(x, 3) for k, x in r.items()},
                                         worst=round(min(r.values()), 3), worst_ci_lo=round(worst_ci(v), 3)))
    df = pd.DataFrame(rows); df.to_csv(f"{R}/worst_source.csv", index=False); print(f"\n=== worst-source {model} ==="); print(df.to_string(index=False))
    # ---- 2. adjusted memory: charge projection matrix (D*d*4 bytes) to SQ4/SQ8
    X, Q, gold, src2, cont2 = load("data", model)
    P = {m: pm(f"pq_{m}_s*.npy") for m in [16, 32, 64, 128, 256]}
    rows = []
    for m in [16, 32, 64, 128, 256]:
        if D % m: continue
        pq_bytes = N_CH * m + 256 * 4 * D
        for kind, bits in [("sq4", 4), ("sq8", 8)]:
            d = 1
            while d < D and (N_CH * d * bits / 8 + D * d * 4) < pq_bytes: d += 1
            d -= 1
            if d < 1 or d > D: continue
            vs = []
            for s in SEEDS:
                mu, V, fit = fit_basis(X, 100000, s); Xp, Qp = projN(X, mu, V, d), projN(Q, mu, V, d)
                idx = faiss.index_factory(d, "SQ4" if kind == "sq4" else "SQ8", faiss.METRIC_INNER_PRODUCT); idx.train(Xp[fit]); idx.add(Xp)
                vs.append(ndcg10(search(idx, Qp), gold))
            v = np.mean(vs, 0); pqv = P[m]
            e = boot_macro(pqv - v, src, cont, nb=1000)
            rows.append(dict(model=model, m=m, kind=kind, d_adj=d, PQ_ret=round(macro(pqv, src) / macro(raw, src), 4),
                             SC_ret=round(macro(v, src) / macro(raw, src), 4), **{"PQ-SC": fmt(e)}))
            print(model, m, kind, d, flush=True)
    dd = pd.DataFrame(rows); dd.to_csv(f"{R}/adjusted_memory.csv", index=False); print(f"\n=== adjusted memory {model} (PQ - scalar) ==="); print(dd.to_string(index=False))
