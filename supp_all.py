import argparse, math, os, glob, time, faiss, numpy as np, pandas as pd
from evalcore import *
ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["centring", "ladder", "pq", "opq", "agg"])
ap.add_argument("--model", required=True)
ap.add_argument("--seeds", default="0,1,2,3,4")
ap.add_argument("--q_emb", default=None)
ap.add_argument("--tag", default="std")
ap.add_argument("--ms", default="16,32,64,128,256")
ap.add_argument("--threads", type=int, default=8)
ap.add_argument("--opq_ntrain", type=int, default=20000)
ap.add_argument("--opq_iter", type=int, default=10)
a = ap.parse_args()
faiss.omp_set_num_threads(a.threads)
R = f"out_full/supp2/{a.model}_{a.tag}"; os.makedirs(R, exist_ok=True)
SEEDS = [int(s) for s in a.seeds.split(",")]
N_CH = 160340
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
ex = lambda n: os.path.exists(f"{R}/{n}.npy")
sv = lambda n, v: np.save(f"{R}/{n}.npy", v)
ld = lambda n: np.load(f"{R}/{n}.npy", allow_pickle=True)

if a.stage != "agg":
    X, Q, gold, src, cont = load("data", a.model, a.q_emb)
    N, D = X.shape
    np.save(f"{R}/src.npy", src); np.save(f"{R}/cont.npy", cont)
    if not ex("raw"):
        s, b = raw_baseline(X, Q, gold, src); sv("raw", s); print("raw macro", round(b, 4), flush=True)

def flat(Xp, Qp):
    idx = faiss.IndexFlatIP(Xp.shape[1]); idx.add(Xp); return ndcg10(search(idx, Qp), gold)

# ------------------------------------------------------------ centring (4 conditions, normalised)
if a.stage == "centring":
    base = macro(ld("raw"), src)
    muC, VC, fit = fit_basis(X, 100000, 0, True); _, VU, _ = fit_basis(X, 100000, 0, False)
    zero = np.zeros(D, dtype="float32"); rows = []
    for d in [24, 48, 96, 192, 384]:
        S = {"pca_c": flat(projN(X, muC, VC, d), projN(Q, muC, VC, d)),
             "pca_u": flat(projN(X, zero, VU, d), projN(Q, zero, VU, d))}
        ru, rc = [], []
        for k in range(3):
            Rm = (np.random.default_rng(100 + k).standard_normal((D, d)) / np.sqrt(d)).astype("float32")
            ru.append(flat(nrm(X @ Rm), nrm(Q @ Rm))); rc.append(flat(nrm((X - muC) @ Rm), nrm((Q - muC) @ Rm)))
        S["rp_u"], S["rp_c"] = np.mean(ru, 0), np.mean(rc, 0)
        r = dict(model=a.model, d=d, B_f32=4 * d)
        for k, v in S.items(): r[k + "_ret"] = round(macro(v, src) / base, 4)
        for nm, (x, y) in {"FROZEN pca_c-rp_u": ("pca_c", "rp_u"), "projector|centred": ("pca_c", "rp_c"),
                           "projector|uncentred": ("pca_u", "rp_u"), "centring|PCA": ("pca_c", "pca_u"),
                           "centring|RP": ("rp_c", "rp_u")}.items():
            e, lo, hi = boot_macro(S[x] - S[y], src, cont, nb=1000); r[nm] = f"{e:+.4f} [{lo:+.4f},{hi:+.4f}]"
        rows.append(r); print(d, "done", flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{R}/centring.csv", index=False); print(df.T.to_string())

# ------------------------------------------------------------ bit-width ladder (per-query saved)
if a.stage == "ladder":
    FMT = {"f32": (4.0, None), "fp16": (2.0, "SQfp16"), "sq8": (1.0, "SQ8"), "sq4": (0.5, "SQ4")}
    for seed in SEEDS:
        mu, V, fit = fit_basis(X, 100000, seed)
        for fmt, (bpd, spec) in FMT.items():
            for B in [16, 32, 64, 96, 192, 384, 768]:
                d = int(B / bpd); nm = f"lad_s{seed}_{fmt}_{B}"
                if d < 1 or d > D or ex(nm): continue
                t = time.time(); Xp, Qp = projN(X, mu, V, d), projN(Q, mu, V, d)
                if spec is None: idx = faiss.IndexFlatIP(d)
                else:
                    idx = faiss.index_factory(d, spec, faiss.METRIC_INNER_PRODUCT); idx.train(Xp[fit])
                idx.add(Xp); v = ndcg10(search(idx, Qp), gold); sv(nm, v)
                print(f"seed{seed} {fmt} B={B} d={d} macro={macro(v, src):.4f} ({time.time()-t:.0f}s)", flush=True)

# ------------------------------------------------------------ PQ + PCA+SQ8 at matched memory
if a.stage == "pq":
    for seed in SEEDS:
        perm = np.random.default_rng(seed).permutation(N); tr = np.ascontiguousarray(X[perm[:50000]])
        mu, V, fit = fit_basis(X, 100000, seed)
        for m in [int(x) for x in a.ms.split(",") if D % int(x) == 0]:
            if not ex(f"pq_{m}_s{seed}"):
                t = time.time(); idx = faiss.IndexPQ(D, m, 8, faiss.METRIC_INNER_PRODUCT)
                idx.pq.cp.niter = 15; idx.pq.cp.seed = 1234 + seed; idx.train(tr); t1 = time.time() - t; idx.add(X)
                sv(f"pq_{m}_s{seed}", ndcg10(search(idx, Q), gold))
                print(f"seed{seed} PQ m={m}: train {t1:.0f}s total {time.time()-t:.0f}s", flush=True)
            d = math.ceil(m + 256 * 4 * D / N)
            if d <= D and not ex(f"sqm_{m}_s{seed}"):
                Xp, Qp = projN(X, mu, V, d), projN(Q, mu, V, d)
                sq = faiss.index_factory(d, "SQ8", faiss.METRIC_INNER_PRODUCT); sq.train(Xp[fit]); sq.add(Xp)
                sv(f"sqm_{m}_s{seed}", ndcg10(search(sq, Qp), gold)); print(f"seed{seed} SQ8 m={m} d={d}", flush=True)

# ------------------------------------------------------------ OPQ (reduced settings)
if a.stage == "opq":
    for seed in SEEDS:
        perm = np.random.default_rng(seed).permutation(N); tr = np.ascontiguousarray(X[perm[:a.opq_ntrain]])
        for m in [int(x) for x in a.ms.split(",") if D % int(x) == 0]:
            if ex(f"opq_{m}_s{seed}"): continue
            t = time.time(); op = faiss.OPQMatrix(D, m); op.niter = a.opq_iter; op.niter_pq = 4
            op.niter_pq_0 = 10; op.max_train_points = a.opq_ntrain
            bs = faiss.IndexPQ(D, m, 8, faiss.METRIC_INNER_PRODUCT); bs.pq.cp.niter = 15
            idx = faiss.IndexPreTransform(op, bs); idx.train(tr); t1 = time.time() - t; idx.add(X)
            sv(f"opq_{m}_s{seed}", ndcg10(search(idx, Q), gold))
            print(f"seed{seed} OPQ m={m}: train {t1:.0f}s total {time.time()-t:.0f}s", flush=True)

# ------------------------------------------------------------ aggregation
if a.stage == "agg":
    src, cont, raw = ld("src"), ld("cont"), ld("raw"); base = macro(raw, src)
    D = {"A": 768, "B": 1024}[a.model]
    print(f"=== {a.model} {a.tag}  raw macro {base:.4f} ===")
    L = {}
    for f in glob.glob(f"{R}/lad_s*_*_*.npy"):
        _, s, fmt, B = os.path.basename(f)[:-4].split("_"); L.setdefault((fmt, int(B)), []).append(np.load(f))
    if L:
        rows = []
        for (fmt, B), vs in sorted(L.items(), key=lambda kv: (kv[0][1], kv[0][0])):
            ms = [macro(v, src) / base for v in vs]
            rows.append(dict(B=B, fmt=fmt, n_seeds=len(vs), ret_mean=np.mean(ms), ret_std=np.std(ms, ddof=1) if len(ms) > 1 else np.nan))
        df = pd.DataFrame(rows); df.to_csv(f"{R}/ladder_summary.csv", index=False)
        print(df.pivot(index="B", columns="fmt", values="ret_mean").round(4).to_string())
        print(df.pivot(index="B", columns="fmt", values="ret_std").round(4).to_string())
        mv = {k: np.mean(v, 0) for k, v in L.items()}; rows = []
        for B in [16, 32, 64, 96, 192, 384, 768]:
            for nm, x, y in [("sq8-f32", "sq8", "f32"), ("fp16-f32", "fp16", "f32"), ("sq4-sq8", "sq4", "sq8")]:
                if (x, B) in mv and (y, B) in mv:
                    e, lo, hi = boot_macro(mv[(x, B)] - mv[(y, B)], src, cont, nb=2000)
                    rows.append(dict(B=B, contrast=nm, diff=round(e, 4), lo=round(lo, 4), hi=round(hi, 4)))
        dd = pd.DataFrame(rows); dd.to_csv(f"{R}/ladder_contrasts.csv", index=False)
        print(dd.pivot(index="B", columns="contrast", values=["diff", "lo", "hi"]).to_string())
    rows = []
    for m in [16, 32, 64, 128, 256, 384, 512]:
        g = lambda k: [np.load(f) for f in sorted(glob.glob(f"{R}/{k}_{m}_s*.npy"))]
        P, S, O = g("pq"), g("sqm"), g("opq")
        if not P or not S: continue
        pm, sm = np.mean(P, 0), np.mean(S, 0)
        r = dict(m=m, n_seeds=len(P), MiB_PQ=round((N_CH * m + 256 * 4 * D) / 2**20, 2),
                 PQ_ret=macro(pm, src) / base, SQ8_ret=macro(sm, src) / base,
                 PQ_sd=np.std([macro(v, src) for v in P]), SQ8_sd=np.std([macro(v, src) for v in S]))
        e, lo, hi = boot_macro(pm - sm, src, cont); r["PQ-SQ8"] = f"{e:+.4f} [{lo:+.4f},{hi:+.4f}]"
        if O:
            om = np.mean(O, 0); r["OPQ_ret"] = macro(om, src) / base
            e, lo, hi = boot_macro(om - pm, src, cont); r["OPQ-PQ"] = f"{e:+.4f} [{lo:+.4f},{hi:+.4f}]"
        rows.append(r)
    if rows:
        df = pd.DataFrame(rows); df.to_csv(f"{R}/pq_summary.csv", index=False); print(df.round(4).to_string(index=False))
