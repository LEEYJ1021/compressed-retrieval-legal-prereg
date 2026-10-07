import argparse, glob, math, os, faiss, numpy as np, pandas as pd
from evalcore import *
ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["sq4m", "recall", "rp10", "binary", "mrl"])
ap.add_argument("--model", required=True)
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--threads", type=int, default=8)
a = ap.parse_args()
faiss.omp_set_num_threads(a.threads)
R = f"out_full/supp2/{a.model}_std"; os.makedirs(R, exist_ok=True)
SEEDS = [int(s) for s in a.seeds.split(",")]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
ex = lambda n: os.path.exists(f"{R}/{n}.npy")
sv = lambda n, v: np.save(f"{R}/{n}.npy", v)
ld = lambda n: np.load(f"{R}/{n}.npy")
fmt = lambda t: f"{t[0]:+.4f} [{t[1]:+.4f},{t[2]:+.4f}]"
X, Q, gold, src, cont = load("data", a.model)
N, D = X.shape
if not ex("raw"):
    s, b = raw_baseline(X, Q, gold, src); sv("raw", s)
raw = ld("raw"); base = macro(raw, src)
print("raw macro", round(base, 4), flush=True)

def make_index(kind, Xp, fit):
    d = Xp.shape[1]
    if kind == "f32": idx = faiss.IndexFlatIP(d)
    else:
        spec = {"fp16": "SQfp16", "sq8": "SQ8", "sq4": "SQ4"}[kind]
        idx = faiss.index_factory(d, spec, faiss.METRIC_INNER_PRODUCT); idx.train(Xp[fit])
    idx.add(Xp); return idx

def flat(Xp, Qp):
    idx = faiss.IndexFlatIP(Xp.shape[1]); idx.add(Xp); return ndcg10(search(idx, Qp), gold)

if a.stage == "sq4m":
    seeds = sorted({int(os.path.basename(f)[:-4].split("_s")[-1]) for f in glob.glob(f"{R}/pq_*_s*.npy")})
    cache, rows = {}, []
    for m in [16, 32, 64, 128, 256]:
        if D % m: continue
        d = 2 * math.ceil(m + 256 * 4 * D / N)
        if d > D: continue
        for s in seeds:
            if ex(f"sq4m_{m}_s{s}") or not ex(f"pq_{m}_s{s}"): continue
            if s not in cache: cache[s] = fit_basis(X, 100000, s)
            mu, V, fit = cache[s]
            Xp, Qp = projN(X, mu, V, d), projN(Q, mu, V, d)
            sv(f"sq4m_{m}_s{s}", ndcg10(search(make_index("sq4", Xp, fit), Qp), gold))
            print("sq4m", m, "seed", s, "d", d, flush=True)
        P = [np.load(f) for f in sorted(glob.glob(f"{R}/pq_{m}_s*.npy"))]
        S4 = [np.load(f) for f in sorted(glob.glob(f"{R}/sq4m_{m}_s*.npy"))]
        S8 = [np.load(f) for f in sorted(glob.glob(f"{R}/sqm_{m}_s*.npy"))]
        if not P or not S4: continue
        pm, s4 = np.mean(P, 0), np.mean(S4, 0)
        r = dict(model=a.model, m=m, d_sq4=d, n_seeds_pq=len(P), n_seeds_sq4=len(S4),
                 PQ_ret=macro(pm, src) / base, SQ4_ret=macro(s4, src) / base,
                 PQ_sd=np.std([macro(v, src) for v in P]), SQ4_sd=np.std([macro(v, src) for v in S4]))
        r["PQ-SQ4"] = fmt(boot_macro(pm - s4, src, cont))
        if S8:
            s8 = np.mean(S8, 0); r["SQ8_ret"] = macro(s8, src) / base; r["SQ4-SQ8"] = fmt(boot_macro(s4 - s8, src, cont))
        rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(f"{R}/pq_sq4_summary.csv", index=False); print(df.round(4).to_string(index=False))

if a.stage == "recall":
    def hits(I):
        out = np.zeros((len(I), 3))
        for i, (row, g) in enumerate(zip(I, gold)):
            h = np.isin(row, g); out[i] = [h[:5].any(), h.any(), h.sum() / min(len(g), 10)]
        return out
    mu, V, fit = fit_basis(X, 100000, 0)
    cfg = {}
    idx = faiss.IndexFlatIP(D); idx.add(X); cfg["raw"] = search(idx, Q)
    for B in [64, 96, 128, 192, 384]:
        for kind, d in [("f32", B // 4), ("sq8", B), ("sq4", 2 * B)]:
            if d > D: continue
            Xp, Qp = projN(X, mu, V, d), projN(Q, mu, V, d)
            cfg[f"{kind}_B{B}"] = search(make_index(kind, Xp, fit), Qp); print("recall cfg", kind, B, flush=True)
    perm = np.random.default_rng(0).permutation(N); tr = np.ascontiguousarray(X[perm[:50000]])
    for m in [64, 128]:
        if D % m: continue
        pq = faiss.IndexPQ(D, m, 8, faiss.METRIC_INNER_PRODUCT); pq.pq.cp.niter = 15; pq.pq.cp.seed = 1234
        pq.train(tr); pq.add(X); cfg[f"pq_m{m}"] = search(pq, Q); print("recall cfg pq", m, flush=True)
    H = {k: hits(v) for k, v in cfg.items()}; rows = []
    for k, h in H.items():
        r = dict(model=a.model, config=k, hit5=macro(h[:, 0], src), hit10=macro(h[:, 1], src), cov10=macro(h[:, 2], src))
        r["hit10_ret"] = r["hit10"] / macro(H["raw"][:, 1], src)
        for s in SRC3: r[f"hit10_{s}"] = float(h[src == s, 1].mean())
        if k != "raw": r["hit10-raw"] = fmt(boot_macro(h[:, 1] - H["raw"][:, 1], src, cont, nb=1000))
        rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(f"{R}/recall.csv", index=False); print(df.round(4).to_string(index=False))

if a.stage == "rp10":
    muC, VC, fit = fit_basis(X, 100000, 0, True); rows = []
    for d in [24, 48, 96, 192, 384]:
        if d > D: continue
        pc = flat(projN(X, muC, VC, d), projN(Q, muC, VC, d)); ru, rc = [], []
        for k in range(10):
            Rm = (np.random.default_rng(100 + k).standard_normal((D, d)) / np.sqrt(d)).astype("float32")
            ru.append(flat(nrm(X @ Rm), nrm(Q @ Rm))); rc.append(flat(nrm((X - muC) @ Rm), nrm((Q - muC) @ Rm)))
        mu_u = [macro(v, src) for v in ru]; mu_c = [macro(v, src) for v in rc]
        ru_m, rc_m = np.mean(ru, 0), np.mean(rc, 0)
        r = dict(model=a.model, d=d, pca_c_ret=macro(pc, src) / base,
                 rp_u_ret=np.mean(mu_u) / base, rp_u_sd_draws=np.std(mu_u, ddof=1) / base,
                 rp_c_ret=np.mean(mu_c) / base, rp_c_sd_draws=np.std(mu_c, ddof=1) / base,
                 frozen_gap_draw_min=macro(pc, src) - max(mu_u), frozen_gap_draw_max=macro(pc, src) - min(mu_u))
        r["FROZEN pca_c-rp_u"] = fmt(boot_macro(pc - ru_m, src, cont, nb=1000))
        r["projector|centred"] = fmt(boot_macro(pc - rc_m, src, cont, nb=1000))
        r["centring|RP"] = fmt(boot_macro(rc_m - ru_m, src, cont, nb=1000))
        rows.append(r); print("rp10 d", d, flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{R}/rp10.csv", index=False); print(df.T.to_string())

if a.stage == "binary":
    mu, V, fit = fit_basis(X, 100000, 0); rows = []
    for B in [16, 32, 64, 96, 128]:
        d = 8 * B
        if d > D: continue
        xb = np.ascontiguousarray(np.packbits(((X - mu) @ V[:, :d]) > 0, axis=1))
        qb = np.ascontiguousarray(np.packbits(((Q - mu) @ V[:, :d]) > 0, axis=1))
        idx = faiss.IndexBinaryFlat(d); idx.add(xb); I = idx.search(qb, 10)[1]
        v = ndcg10(I, gold); sv(f"bin_B{B}", v)
        rows.append(dict(model=a.model, B=B, d_bits=d, ret=macro(v, src) / base, macro=macro(v, src)))
        print("binary B", B, rows[-1], flush=True)
    df = pd.DataFrame(rows)
    try:
        ls = pd.read_csv(f"{R}/ladder_summary.csv")
        for f in ["sq8", "sq4"]:
            df[f"{f}_ret_ladder"] = [ls[(ls.B == b) & (ls.fmt == f)].ret_mean.iloc[0] if len(ls[(ls.B == b) & (ls.fmt == f)]) else np.nan for b in df.B]
    except Exception as e: print("ladder summary not merged:", e)
    df.to_csv(f"{R}/binary.csv", index=False); print(df.round(4).to_string(index=False))

if a.stage == "mrl":
    KINDS = {"f32": 4.0, "sq8": 1.0, "sq4": 0.5}; BUD = [16, 32, 64, 96, 192, 384, 768]
    for seed in SEEDS:
        mu, V, fit = fit_basis(X, 100000, seed)
        for method in ["pca", "trunc"]:
            if method == "trunc" and seed != SEEDS[0]: continue
            for kind, bpd in KINDS.items():
                for B in BUD:
                    d = int(B / bpd); nm = f"mrl_{method}_{kind}_{B}_s{seed}"
                    if d < 1 or d > D or ex(nm): continue
                    if method == "pca": Xp, Qp = projN(X, mu, V, d), projN(Q, mu, V, d)
                    else: Xp, Qp = nrm(X[:, :d]), nrm(Q[:, :d])
                    v = ndcg10(search(make_index(kind, Xp, fit), Qp), gold); sv(nm, v)
                    print(nm, f"macro={macro(v, src):.4f}", flush=True)
    def Ls(method, kind, B):
        return [np.load(f) for f in sorted(glob.glob(f"{R}/mrl_{method}_{kind}_{B}_s*.npy"))]
    rows, crow = [], []
    for B in BUD:
        r = dict(model=a.model, B=B)
        for method in ["trunc", "pca"]:
            for kind in KINDS:
                v = Ls(method, kind, B)
                if v: r[f"{method}_{kind}"] = np.mean([macro(x, src) for x in v]) / base
        rows.append(r)
        for kind in KINDS:
            t, p = Ls("trunc", kind, B), Ls("pca", kind, B)
            if t and p: crow.append(dict(B=B, contrast=f"trunc-pca|{kind}", res=fmt(boot_macro(np.mean(t, 0) - np.mean(p, 0), src, cont, nb=1000))))
        t8, t32 = Ls("trunc", "sq8", B), Ls("trunc", "f32", B)
        if t8 and t32: crow.append(dict(B=B, contrast="trunc_sq8-trunc_f32", res=fmt(boot_macro(np.mean(t8, 0) - np.mean(t32, 0), src, cont, nb=1000))))
        p8, p32 = Ls("pca", "sq8", B), Ls("pca", "f32", B)
        if p8 and p32: crow.append(dict(B=B, contrast="pca_sq8-pca_f32", res=fmt(boot_macro(np.mean(p8, 0) - np.mean(p32, 0), src, cont, nb=1000))))
    df = pd.DataFrame(rows); df.to_csv(f"{R}/mrl_summary.csv", index=False); print(df.round(4).to_string(index=False))
    dc = pd.DataFrame(crow); dc.to_csv(f"{R}/mrl_contrasts.csv", index=False); print(dc.pivot(index="B", columns="contrast", values="res").to_string())
