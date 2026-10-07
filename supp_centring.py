import argparse, faiss, numpy as np, pandas as pd
from evalcore import *
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--data_dir", default="data")
ap.add_argument("--draws", type=int, default=3); a = ap.parse_args()
X, Q, gold, src, cont = load(a.data_dir, a.model)
N, D = X.shape
_, base = raw_baseline(X, Q, gold, src)
muC, VC, fit = fit_basis(X, 100000, 0, centre=True)
_,   VU, _   = fit_basis(X, 100000, 0, centre=False)     # uncentred second-moment eigenvectors
zero = np.zeros(D, dtype="float32")
def run(Xp, Qp):
    idx = faiss.IndexFlatIP(Xp.shape[1]); idx.add(np.ascontiguousarray(Xp))
    return ndcg10(search(idx, np.ascontiguousarray(Qp)), gold)
rows, S = [], {}
for d in [24, 48, 96, 192, 384]:
    S["pca_c"] = run(proj(X, muC, VC, d), proj(Q, muC, VC, d))
    S["pca_u"] = run(proj(X, zero, VU, d), proj(Q, zero, VU, d))
    rc, ru = [], []
    for k in range(a.draws):
        R = (np.random.default_rng(100 + k).standard_normal((D, d)) / np.sqrt(d)).astype("float32")
        rc.append(run((X - muC) @ R, (Q - muC) @ R)); ru.append(run(X @ R, Q @ R))
    S["rp_c"], S["rp_u"] = np.mean(rc, 0), np.mean(ru, 0)
    r = dict(model=a.model, d=d, B_f32=4 * d)
    for k, v in S.items(): r[k] = macro(v, src) / base
    for name, (x, y) in {"projector|centred (pca_c-rp_c)": ("pca_c", "rp_c"),
                         "projector|uncentred (pca_u-rp_u)": ("pca_u", "rp_u"),
                         "centring|PCA (pca_c-pca_u)": ("pca_c", "pca_u"),
                         "centring|RP (rp_c-rp_u)": ("rp_c", "rp_u")}.items():
        e, lo, hi = boot_macro(S[x] - S[y], src, cont, nb=1000)
        r[name] = f"{e:+.4f} [{lo:+.4f},{hi:+.4f}]"
    rows.append(r); print(d, "done", flush=True)
df = pd.DataFrame(rows); df.to_csv(f"out_full/supp/centring_{a.model}.csv", index=False)
pd.set_option("display.width", 300); pd.set_option("display.max_colwidth", 40)
print(df.round(3).T.to_string())
