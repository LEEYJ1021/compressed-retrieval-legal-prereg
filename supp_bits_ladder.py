import argparse, faiss, numpy as np, pandas as pd, time
from evalcore import *
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--data_dir", default="data")
ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--q_emb", default=None)
ap.add_argument("--tag", default="std"); a = ap.parse_args()

X, Q, gold, src, cont = load(a.data_dir, a.model, a.q_emb)
D = X.shape[1]
s_raw, base = raw_baseline(X, Q, gold, src)
print("raw macro", round(base, 4))
FMT = {"f32": (4.0, None), "fp16": (2.0, "SQfp16"), "sq8": (1.0, "SQ8"), "sq4": (0.5, "SQ4")}
BUD = [16, 32, 64, 96, 192, 384, 768]
rows = []
for seed in range(a.seeds):
    mu, V, fit = fit_basis(X, 100000, seed)
    for fmt, (bpd, spec) in FMT.items():
        for B in BUD:
            d = int(B / bpd)
            if d < 1 or d > D: continue
            t = time.time()
            Xp, Qp = proj(X, mu, V, d), projQ(Q, V, d)
            if spec is None: idx = faiss.IndexFlatIP(d)
            else:
                idx = faiss.index_factory(d, spec, faiss.METRIC_INNER_PRODUCT); idx.train(Xp[fit])
            idx.add(Xp)
            m = macro(ndcg10(search(idx, Qp), gold), src)
            rows.append(dict(model=a.model, seed=seed, fmt=fmt, B=B, d=d, macro=m, retention=m / base))
            print(f"seed{seed} {fmt:5s} B={B:4d} d={d:4d} macro={m:.4f} ret={m/base:.3f} ({time.time()-t:.0f}s)", flush=True)
df = pd.DataFrame(rows)
df.to_csv(f"out_full/supp/bits_ladder_{a.model}_{a.tag}.csv", index=False)
g = df.groupby(["B", "fmt"]).retention.agg(["mean", "std"]).round(4).unstack("fmt")
print("\nretention mean/std over seeds\n", g.to_string())
