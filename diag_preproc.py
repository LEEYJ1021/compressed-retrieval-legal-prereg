import argparse, os, numpy as np, pandas as pd, faiss
from evalcore import *
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--m", type=int, required=True)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args(); faiss.omp_set_num_threads(a.threads)
R = f"out_full/supp2/{a.model}_std"; OUT = f"out_full/supp2/diag_{a.model}"; os.makedirs(OUT, exist_ok=True)
X, Q, gold, src, cont = load("data", a.model, None); N, D = X.shape
SRC3 = ["contractnli", "cuad", "maud"]
raw = np.load(f"{R}/raw.npy"); base = macro(raw, src)
print("mean ||x||:", float(np.linalg.norm(X[:2000], axis=1).mean()), " mean ||q||:", float(np.linalg.norm(Q[:500], axis=1).mean()), flush=True)
Xr = np.ascontiguousarray(X, dtype="float32"); Qr = np.ascontiguousarray(Q, dtype="float32")
def flat(Xp, Qp):
    ix = faiss.IndexFlatIP(Xp.shape[1]); ix.add(Xp); return ndcg10(search(ix, Qp), gold)
mu, V, fit = fit_basis(X, 100000, a.seed)
_, VU, _ = fit_basis(X, 100000, a.seed, False); zero = np.zeros(D, dtype="float32")
DF = D
try:
    Xc, Qc = projN(X, mu, V, DF), projN(Q, mu, V, DF)
except Exception as e:
    DF = 768; print("full-dim PCA unavailable (", e, ") -> d=768", flush=True)
    Xc, Qc = projN(X, mu, V, DF), projN(Q, mu, V, DF)
Xu, Qu = projN(X, zero, VU, DF), projN(Q, zero, VU, DF)
def sq(spec, Xp, Qp):
    ix = faiss.index_factory(Xp.shape[1], spec, faiss.METRIC_INNER_PRODUCT)
    ix.train(np.ascontiguousarray(Xp[fit])); ix.add(Xp)
    return ndcg10(search(ix, Qp), gold)
def pq(Xp, Qp):
    ix = faiss.IndexPQ(Xp.shape[1], a.m, 8, faiss.METRIC_INNER_PRODUCT)
    ix.pq.cp.niter = 15; ix.pq.cp.seed = 1234 + a.seed
    perm = np.random.default_rng(a.seed).permutation(N)
    ix.train(np.ascontiguousarray(Xp[perm[:100000]])); ix.add(Xp)
    return ndcg10(search(ix, Qp), gold)
def row(name, bytes_pv, v):
    r = {s: float(v[src == s].mean() / raw[src == s].mean()) for s in SRC3}
    return dict(cond=name, B=bytes_pv, macro=macro(v, src) / base, **r, W_point=min(r.values()))
rows = []
rows.append(row("flat float32, raw (sanity: ~1)", 4 * D, flat(Xr, Qr)))
rows.append(row(f"flat float32, PCA centred+norm d={DF}", 4 * DF, flat(Xc, Qc)))
rows.append(row(f"SQ8 raw d={D}", D, sq("SQ8", Xr, Qr)))
rows.append(row(f"SQ4 raw d={D} (same bytes as PQ m={a.m})", D // 2, sq("SQ4", Xr, Qr)))
rows.append(row(f"SQ8 raw+L2 d={D}", D, sq("SQ8", nrm(Xr.copy()), nrm(Qr.copy()))))
rows.append(row(f"SQ4 raw+L2 d={D}", D // 2, sq("SQ4", nrm(Xr.copy()), nrm(Qr.copy()))))
rows.append(row(f"SQ8 PCA uncentred+norm d={DF}", DF, sq("SQ8", Xu, Qu)))
rows.append(row(f"SQ8 PCA centred+norm d={DF} (pipeline)", DF, sq("SQ8", Xc, Qc)))
rows.append(row(f"SQ4 PCA centred+norm d={DF}", DF // 2, sq("SQ4", Xc, Qc)))
rows.append(row(f"PQ m={a.m}, raw (as in supp_all.py)", a.m, pq(Xr, Qr)))
rows.append(row(f"PQ m={a.m}, centred+norm", a.m, pq(Xc, Qc)))
df = pd.DataFrame(rows); df.round(4).to_csv(f"{OUT}/diag_m{a.m}.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(df.round(3).to_string(index=False))
