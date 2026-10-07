import argparse, faiss, numpy as np, pandas as pd
from evalcore import *
ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True)
ap.add_argument("--draws", type=int, default=3); a = ap.parse_args()
X, Q, gold, src, cont = load("data", a.model)
N, D = X.shape
_, base = raw_baseline(X, Q, gold, src)
mu, V, fit = fit_basis(X, 100000, 0)
def run(Xp, Qp):
    idx = faiss.IndexFlatIP(Xp.shape[1]); idx.add(np.ascontiguousarray(Xp, dtype="float32"))
    return ndcg10(search(idx, np.ascontiguousarray(Qp, dtype="float32")), gold)
rows = []
for d in [24, 48, 96, 192, 384]:
    S = {}
    S["pca_dbonly"] = run((X-mu)@V[:, :d], Q@V[:, :d])
    S["pca_both"]   = run((X-mu)@V[:, :d], (Q-mu)@V[:, :d])
    rb, ro = [], []
    for k in range(a.draws):
        R = (np.random.default_rng(100+k).standard_normal((D, d))/np.sqrt(d)).astype("float32")
        ro.append(run(X@R, Q@R)); rb.append(run((X-mu)@R, (Q-mu)@R))
    S["rp_dbonly"], S["rp_both"] = np.mean(ro, 0), np.mean(rb, 0)
    r = dict(model=a.model, d=d, B_f32=4*d)
    for k, v in S.items(): r[k] = macro(v, src)/base
    for name, (x, y) in {"projector|both (pca_both-rp_both)": ("pca_both", "rp_both"),
                         "projector|dbonly (pca_dbonly-rp_dbonly)": ("pca_dbonly", "rp_dbonly"),
                         "centring both-vs-dbonly|PCA": ("pca_both", "pca_dbonly"),
                         "centring both-vs-dbonly|RP": ("rp_both", "rp_dbonly")}.items():
        e, lo, hi = boot_macro(S[x]-S[y], src, cont, nb=1000)
        r[name] = f"{e:+.4f} [{lo:+.4f},{hi:+.4f}]"
    rows.append(r); print(d, "done", flush=True)
df = pd.DataFrame(rows); df.to_csv(f"out_full/supp/centring2_{a.model}.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 40)
print(df.round(3).T.to_string())
