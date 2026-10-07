import argparse, os, glob, faiss, numpy as np, pandas as pd
from evalcore import *
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--q_emb", required=True)
ap.add_argument("--tag", default="wd_noname"); ap.add_argument("--seeds", default="0,1,2")
a = ap.parse_args(); faiss.omp_set_num_threads(8)
R = f"out_full/supp2/{a.model}_{a.tag}"; os.makedirs(R, exist_ok=True)
X, Q, gold, src, cont = load("data", a.model, a.q_emb); N, D = X.shape
doc = pd.read_parquet("data/chunks.parquet").doc_id.astype(str).values
cidx = {}
for i, d in enumerate(doc): cidx.setdefault(d, []).append(i)
cidx = {d: np.array(v) for d, v in cidx.items()}
qidx = {}
for i, d in enumerate(cont): qidx.setdefault(d, []).append(i)
def within(Xd, Qp):
    I = -np.ones((len(Qp), 10), dtype=np.int64)
    for d, qi in qidx.items():
        ci = cidx[d]; qi = np.array(qi); S = Qp[qi] @ Xd[ci].T; k = min(10, len(ci))
        top = np.argpartition(-S, k - 1, axis=1)[:, :k]
        o = np.take_along_axis(S, top, 1).argsort(1)[:, ::-1]; top = np.take_along_axis(top, o, 1)
        I[qi, :k] = ci[top]
    return ndcg10(I, gold)
raw = within(X, Q); base = macro(raw, src); np.save(f"{R}/raw.npy", raw); print("within-doc raw macro", round(base, 4), flush=True)
def mk(kind, Xp, fit):
    d = Xp.shape[1]
    if kind == "f32": idx = faiss.IndexFlatIP(d)
    else:
        idx = faiss.index_factory(d, {"fp16": "SQfp16", "sq8": "SQ8", "sq4": "SQ4"}[kind], faiss.METRIC_INNER_PRODUCT); idx.train(Xp[fit])
    idx.add(Xp); return idx
FM = {"f32": 4.0, "fp16": 2.0, "sq8": 1.0, "sq4": 0.5}; BUD = [16, 32, 64, 96, 192, 384, 768]
for seed in [int(s) for s in a.seeds.split(",")]:
    mu, V, fit = fit_basis(X, 100000, seed)
    for kind, bpd in FM.items():
        for B in BUD:
            d = int(B / bpd); nm = f"{R}/wd_s{seed}_{kind}_{B}.npy"
            if d < 1 or d > D or os.path.exists(nm): continue
            Xp, Qp = projN(X, mu, V, d), projN(Q, mu, V, d)
            Xd = np.ascontiguousarray(mk(kind, Xp, fit).reconstruct_n(0, N))
            v = within(Xd, Qp); np.save(nm, v); print(f"seed{seed} {kind} B={B} d={d} macro={macro(v, src):.4f}", flush=True)
L = {}
for f in glob.glob(f"{R}/wd_s*_*_*.npy"):
    _, s, k, B = os.path.basename(f)[:-4].split("_"); L.setdefault((k, int(B)), []).append(np.load(f))
rows = [dict(B=B, fmt=k, n_seeds=len(v), ret_mean=np.mean([macro(x, src) / base for x in v]),
             ret_std=np.std([macro(x, src) / base for x in v], ddof=1) if len(v) > 1 else np.nan) for (k, B), v in sorted(L.items(), key=lambda kv: (kv[0][1], kv[0][0]))]
df = pd.DataFrame(rows); df.to_csv(f"{R}/wd_summary.csv", index=False)
pd.set_option("display.width", 250); print(df.pivot(index="B", columns="fmt", values="ret_mean").round(4).to_string())
mv = {k: np.mean(v, 0) for k, v in L.items()}; cr = []
for B in BUD:
    for nm, x, y in [("sq8-f32", "sq8", "f32"), ("fp16-f32", "fp16", "f32"), ("sq4-sq8", "sq4", "sq8")]:
        if (x, B) in mv and (y, B) in mv:
            e, lo, hi = boot_macro(mv[(x, B)] - mv[(y, B)], src, cont, nb=2000); cr.append(dict(B=B, contrast=nm, diff=round(e, 4), lo=round(lo, 4), hi=round(hi, 4)))
dc = pd.DataFrame(cr); dc.to_csv(f"{R}/wd_contrasts.csv", index=False); print(dc.pivot(index="B", columns="contrast", values=["diff", "lo", "hi"]).to_string())
