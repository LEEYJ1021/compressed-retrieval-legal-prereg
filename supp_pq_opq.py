import argparse, math, faiss, numpy as np, pandas as pd, time
from evalcore import *
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--data_dir", default="data")
ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--opq_max_m", type=int, default=192)
a = ap.parse_args()
X, Q, gold, src, cont = load(a.data_dir, a.model)
N, D = X.shape
_, base = raw_baseline(X, Q, gold, src); print("raw macro", round(base, 4))
MS = [m for m in [16, 32, 64, 96, 128, 192, 256, 384, 512] if D % m == 0]
acc = {}
def add(key, v): acc.setdefault(key, []).append(v)
for seed in range(a.seeds):
    rng = np.random.default_rng(seed)
    tr = np.ascontiguousarray(X[rng.choice(N, 50000, replace=False)])
    mu, V, fit = fit_basis(X, 100000, seed)
    for m in MS:
        t = time.time()
        pq = faiss.index_factory(D, f"PQ{m}x8", faiss.METRIC_INNER_PRODUCT); pq.train(tr); pq.add(X)
        add(("PQ", m), ndcg10(search(pq, Q), gold))
        if m <= a.opq_max_m:
            op = faiss.index_factory(D, f"OPQ{m},PQ{m}x8", faiss.METRIC_INNER_PRODUCT); op.train(tr); op.add(X)
            add(("OPQ", m), ndcg10(search(op, Q), gold))
        d = math.ceil(m + 256 * 4 * D / N)             # SQ8 dim that matches the PQ index size
        if d <= D:
            Xp, Qp = proj(X, mu, V, d), proj(Q, mu, V, d)
            sq = faiss.index_factory(d, "SQ8", faiss.METRIC_INNER_PRODUCT); sq.train(Xp[fit]); sq.add(Xp)
            add(("SQ8", m), ndcg10(search(sq, Qp), gold))
        print(f"seed{seed} m={m} done ({time.time()-t:.0f}s)", flush=True)
rows = []
for m in MS:
    S = {k: np.mean(acc[(k, m)], axis=0) for k in ["PQ", "OPQ", "SQ8"] if (k, m) in acc}
    seedstd = {k: np.std([macro(v, src) for v in acc[(k, m)]]) for k in S}
    r = dict(model=a.model, m=m, d_sq8=math.ceil(m + 256*4*D/N),
             MiB_PQ=(N*m + 256*4*D)/2**20, MiB_OPQ_adj=(N*m + 256*4*D + 4*D*D)/2**20)
    for k in S: r[f"{k}_macro"] = macro(S[k], src); r[f"{k}_seedstd"] = seedstd[k]
    if "SQ8" in S:
        e, lo, hi = boot_macro(S["PQ"] - S["SQ8"], src, cont); r.update(PQ_minus_SQ8=e, PQ_SQ8_lo=lo, PQ_SQ8_hi=hi)
    if "OPQ" in S:
        e, lo, hi = boot_macro(S["OPQ"] - S["PQ"], src, cont); r.update(OPQ_minus_PQ=e, OPQ_PQ_lo=lo, OPQ_PQ_hi=hi)
    rows.append(r)
df = pd.DataFrame(rows); df.to_csv(f"out_full/supp/pq_opq_{a.model}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(df.round(4).to_string(index=False))
