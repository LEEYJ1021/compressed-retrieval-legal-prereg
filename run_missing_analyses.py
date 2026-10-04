#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_missing_analyses.py -- produces the result tables that are still placeholders in the manuscript:
  t8   Table 8   exact-memory PQ vs PCA+SQ8 (Model A; Model B is recomputed as a check against Table 7)
  t9   Table 9   OPQ vs PCA+SQ8 and PQ at B in {16,32,64}
  t10  Table 10  tau(B/4,B), kappa_hat, rho_SQ_hat and measured gains (+ rank correlation)
  t11  Table 11  2x2 centring x projector control, raw-c minus raw, document hit
  t12  Table 12  five-seed stability (PCAsq8 B192/B384, PQ B64, IVF nprobe 32 on raw)
  t14  Table 14  retention R and worst-source W with 95% CI for PQ/PCAsq8/OPQ -> certification

Usage:   python run_missing_analyses.py probe            # inspect data layout first
         python run_missing_analyses.py all [--enc A,B]  # or t8 t9 t10 t11 t12 t14
Outputs: ./out_missing/*.csv, perq_*.npz (per-query cache), tables.md, meta.json
Conventions follow the manuscript: nDCG@10 with binary gains (Eq. 12), macro over
ContractNLI/CUAD/MAUD (Eq. 14), contract-cluster bootstrap (Eq. 16; 2,000 resamples, seed 20261004).
"""
import argparse, math, os, sys, time, json
from collections import Counter
import numpy as np
import pandas as pd
import faiss
from scipy.stats import spearmanr

# ============================ ADAPTER: EDIT THESE TO YOUR FILES ============================
ROOT = os.environ.get("RP_ROOT", ".")
PATHS = dict(
    chunks="data/chunks.parquet",      # one row per chunk, same order as the chunk embeddings
    queries="data/queries.parquet",    # the ANALYSED 6,877 queries (exclusions already applied)
    emb={"A": "data/emb_A.npy", "B": "data/emb_B.npy"},     # (N, D) chunk embeddings
    qemb={"A": "data/qemb_A.npy", "B": "data/qemb_B.npy"},  # (nq, D) query embeddings, same order as queries
)
COLS = dict(
    chunk_doc="doc_id", chunk_src="source", chunk_id=None,   # chunk_id: set only if gold ids are NOT row indices
    q_src="source", q_gold="gold_chunk_idx", q_keep=None,    # q_gold: list of row indices (or chunk ids)
)
# ===========================================================================================

OUT = os.path.join(ROOT, "out_missing")
SRC = ["ContractNLI", "CUAD", "MAUD"]
SRC_NORM = {"contractnli": "ContractNLI", "cuad": "CUAD", "maud": "MAUD", "privacyqa": "PrivacyQA"}
EXPECT_Q = {"ContractNLI": 977, "CUAD": 4034, "MAUD": 1674, "PrivacyQA": 192}
RAW_REF = {  # Table 4 (macro over 3 sources) -- used as a loader sanity check
    "A": {"ContractNLI": .0815, "CUAD": .1159, "MAUD": .0077, "macro": .0684},
    "B": {"ContractNLI": .0876, "CUAD": .1713, "MAUD": .0101, "macro": .0897}}
T6_PQ = {"A": {16: .0124, 32: .0243, 64: .0453, 96: .0532, 192: .0635, 384: .0679, 768: .0685},
         "B": {16: .0099, 32: .0219, 64: .0394}}
T6_SQ = {"A": {16: .0032, 32: .0110, 64: .0233, 96: .0357, 192: .0546, 384: .0659, 768: .0671},
         "B": {16: .0042, 32: .0118, 64: .0274, 96: .0425, 192: .0708, 384: .0875, 768: .0905}}
GAIN_PAPER = {"A": [.0104, .0202, .0288, .0373, .0302, .0126], "B": [.0103, .0232, .0358, .0509, .0448, .0194]}
BUD_PQ = {"A": [16, 32, 64, 96, 192, 384, 768], "B": [16, 32, 64, 128, 256, 512]}
K = 10
P = dict(n_boot=2000, boot_seed=20261004, pca_n=100000, pq_n=50000, ivf_n=40960, nlist=1024, smoke=False, opq_niter=None)
DISC = 1.0 / np.log2(np.arange(2, K + 2))
IDCG = np.cumsum(DISC)
MD = []  # markdown log


# ------------------------------------------------------------------ data
class Data: pass


def _norm(a):
    a = np.ascontiguousarray(a, dtype=np.float32)
    n = np.linalg.norm(a, axis=1, keepdims=True); n[n == 0] = 1
    return a / n


def load(enc):
    ch = pd.read_parquet(os.path.join(ROOT, PATHS["chunks"]))
    qs = pd.read_parquet(os.path.join(ROOT, PATHS["queries"]))
    X = np.load(os.path.join(ROOT, PATHS["emb"][enc])); Q = np.load(os.path.join(ROOT, PATHS["qemb"][enc]))
    assert len(X) == len(ch), f"chunk embeddings {X.shape} vs chunks {len(ch)}"
    assert len(Q) == len(qs), f"query embeddings {Q.shape} vs queries {len(qs)}"
    if COLS["q_keep"]:
        m = qs[COLS["q_keep"]].astype(bool).to_numpy(); qs = qs[m].reset_index(drop=True); Q = Q[m]
    d = Data(); d.X, d.Q = _norm(X), _norm(Q)
    key = lambda s: s.astype(str).str.lower().str.replace(r"[^a-z]", "", regex=True).map(SRC_NORM).to_numpy()
    d.chunk_src = key(ch[COLS["chunk_src"]]); d.q_src = key(qs[COLS["q_src"]])
    d.chunk_doc = pd.factorize(ch[COLS["chunk_doc"]])[0]
    idmap = {c: i for i, c in enumerate(ch[COLS["chunk_id"]])} if COLS["chunk_id"] else None
    d.gold = [np.asarray([idmap[g] for g in gs] if idmap else list(gs), dtype=np.int64) for gs in qs[COLS["q_gold"]]]
    d.gold_set = [set(g.tolist()) for g in d.gold]
    d.q_doc = d.chunk_doc[[g[0] for g in d.gold]]
    for g, qd in zip(d.gold, d.q_doc): assert (d.chunk_doc[g] == qd).all(), "gold chunks of one query span several documents"
    if not P["smoke"]:
        cnt = Counter(d.q_src)
        assert dict(cnt) == EXPECT_Q, f"query counts {dict(cnt)} != {EXPECT_Q}: apply the exclusions of Appendix B"
        assert len(d.q_src) == 6877 and len(d.X) == 160340
    return d


# ------------------------------------------------------------------ retrieval + metrics
def topk_dense(Qm, Xm, bs=256):
    Qm = np.ascontiguousarray(Qm, np.float32); Xm = np.ascontiguousarray(Xm, np.float32)
    out = np.empty((len(Qm), K), np.int64)
    for s in range(0, len(Qm), bs):
        sc = Qm[s:s + bs] @ Xm.T
        part = np.argpartition(-sc, K, axis=1)[:, :K]
        o = np.argsort(-np.take_along_axis(sc, part, 1), 1)
        out[s:s + bs] = np.take_along_axis(part, o, 1)
    return out


def metrics(top, d):
    nd = np.zeros(len(top))
    for i in range(len(top)):
        g = d.gold_set[i]
        h = np.fromiter((c in g for c in top[i]), float, K)
        nd[i] = (h * DISC).sum() / IDCG[min(K, len(g)) - 1]
    dh = (d.chunk_doc[top] == d.q_doc[:, None]).any(1).astype(float)
    return nd, dh


def ev_dense(d, Qm, Xm): return metrics(topk_dense(Qm, Xm), d)


def sq8(Y, l, u):
    delta = (u - l) / 255.0; delta[delta == 0] = 1.0
    return (l + delta * np.clip(np.round((Y - l) / delta), 0, 255)).astype(np.float32)


def fit_pca(X, seed):
    n = min(P["pca_n"], len(X) // 2 if P["smoke"] else P["pca_n"])
    idx = np.random.default_rng(seed).choice(len(X), n, replace=False)
    S = X[idx].astype(np.float64); mu = S.mean(0); Sc = S - mu
    lam, V = np.linalg.eigh(Sc.T @ Sc / (n - 1)); o = np.argsort(lam)[::-1]; lam, V = lam[o], V[:, o]
    Yf = Sc @ V
    return dict(mu=mu.astype(np.float32), lam=lam, V=V.astype(np.float32), idx=idx,
                l=Yf.min(0).astype(np.float32), u=Yf.max(0).astype(np.float32))


# ------------------------------------------------------------------ bootstrap (Eq. 16)
class Boot:
    def __init__(s, d):
        rng = np.random.default_rng(P["boot_seed"]); s.parts = {}; s.draws = {}
        for src in SRC:
            qi = np.where(d.q_src == src)[0]; u, inv = np.unique(d.q_doc[qi], return_inverse=True)
            s.parts[src] = (qi, inv, len(u), np.bincount(inv).astype(float))
            s.draws[src] = rng.integers(0, len(u), size=(P["n_boot"], len(u)))

    def src_boot(s, v):
        v2 = v if v.ndim == 2 else v[:, None]; out = {}
        for src in SRC:
            qi, inv, Kc, cnt = s.parts[src]
            sums = np.zeros((Kc, v2.shape[1])); np.add.at(sums, inv, v2[qi])
            ix = s.draws[src]
            out[src] = sums[ix].sum(1) / cnt[ix].sum(1)[:, None]
        return out

    def macro_boot(s, v): return np.mean([a[:, 0] for a in s.src_boot(v).values()], 0)
    def macro_point(s, v): return float(np.mean([v[s.parts[x][0]].mean() for x in SRC]))
    def src_point(s, v): return {x: float(v[s.parts[x][0]].mean()) for x in SRC}

    def ci(s, v):
        lo, hi = np.percentile(s.macro_boot(v), [2.5, 97.5]); return s.macro_point(v), lo, hi

    def diff(s, v1, v2):
        b = s.macro_boot(v1) - s.macro_boot(v2); lo, hi = np.percentile(b, [2.5, 97.5])
        return s.macro_point(v1) - s.macro_point(v2), lo, hi

    def retention(s, v, vraw):
        m, m0 = s.src_boot(v), s.src_boot(vraw)
        Rs = np.stack([m[x][:, 0] / m0[x][:, 0] for x in SRC], 1)
        Rm = np.mean([m[x][:, 0] for x in SRC], 0) / np.mean([m0[x][:, 0] for x in SRC], 0)
        return Rm, Rs.min(1)


def sgn(x, nd=4): return f"{x:+.{nd}f}".replace("-", "\u2212")
def cis(lo, hi, nd=4): return f"[{sgn(lo, nd)}, {sgn(hi, nd)}]"
def reading(lo, hi, a="PQ higher", b="PCAsq8 higher"): return a if lo > 0 else (b if hi < 0 else "Includes zero")


def emit(name, header, rows, note=""):
    os.makedirs(OUT, exist_ok=True)
    pd.DataFrame(rows, columns=header).to_csv(os.path.join(OUT, name + ".csv"), index=False, encoding="utf-8-sig")
    txt = f"\n### {name}\n| " + " | ".join(header) + " |\n|" + "|".join(["---"] * len(header)) + "|\n"
    txt += "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows) + ("\n" + note if note else "") + "\n"
    print(txt); MD.append(txt)


# ------------------------------------------------------------------ per-encoder context with cache
class Ctx:
    def __init__(s, enc):
        s.enc = enc; s.d = load(enc); s.D = s.d.X.shape[1]; s.N = len(s.d.X)
        s.boot = Boot(s.d); s.meta = {}; s._pca = {}; s._proj = {}
        s.path = os.path.join(OUT, f"perq_{enc}.npz"); s.cache = {}
        if os.path.exists(s.path):
            z = np.load(s.path)
            for k in z.files:
                if k.endswith("|nd"): s.cache[k[:-3]] = (z[k], z[k[:-3] + "|dh"])

    def get(s, name, fn):
        if name not in s.cache:
            t = time.time(); s.cache[name] = fn(); s.meta[name] = dict(seconds=round(time.time() - t, 1))
            os.makedirs(OUT, exist_ok=True)
            np.savez_compressed(s.path, **{f"{k}|{t_}": v for k, (a, b) in s.cache.items() for t_, v in (("nd", a), ("dh", b))})
        return s.cache[name]

    def pca(s, seed=0):
        if seed not in s._pca: s._pca[seed] = fit_pca(s.d.X, seed)
        return s._pca[seed]

    def dmax(s, seed): return s.D if seed == 0 else min(384, s.D)

    def proj(s, seed, dmax):
        key = (seed, dmax)
        if key in s._proj: return s._proj[key]
        p = s.pca(seed); V = p["V"][:, :dmax]; b = p["mu"] @ V
        out = (np.ascontiguousarray(s.d.X @ V - b), np.ascontiguousarray(s.d.Q @ V - b))
        if seed == 0: s._proj[key] = out
        return out

    def raw(s): return s.get("raw", lambda: ev_dense(s.d, s.d.Q, s.d.X))

    def rawc(s):
        def f():
            mu = s.pca(0)["mu"]; return ev_dense(s.d, s.d.Q - mu, s.d.X - mu)
        return s.get("rawc", f)

    def pcaf32(s, dd, seed=0):
        def f():
            Y, Qy = s.proj(seed, s.dmax(seed)); return ev_dense(s.d, Qy[:, :dd], Y[:, :dd])
        return s.get(f"pcaf32_d{dd}_s{seed}", f)

    def pcasq8(s, dd, seed=0):
        def f():
            Y, Qy = s.proj(seed, s.dmax(seed)); p = s.pca(seed)
            return ev_dense(s.d, Qy[:, :dd], sq8(Y[:, :dd], p["l"][:dd], p["u"][:dd]))
        return s.get(f"pcasq8_d{dd}_s{seed}", f)

    def pcau(s, dd):
        def f():
            V = s.pca(0)["V"][:, :dd]; return ev_dense(s.d, s.d.Q @ V, s.d.X @ V)
        return s.get(f"pcau_d{dd}", f)

    def rp(s, dd, centred, draws=3):
        def f():
            mu = s.pca(0)["mu"]; nds, dhs = [], []
            for j in range(draws):
                R = np.random.default_rng(1000 + j).normal(0, 1 / math.sqrt(dd), (s.D, dd)).astype(np.float32)
                b = (mu @ R) if centred else 0
                nd, dh = ev_dense(s.d, s.d.Q @ R - b, s.d.X @ R - b); nds.append(nd); dhs.append(dh)
            return np.mean(nds, 0), np.mean(dhs, 0)
        return s.get(f"{'rpc' if centred else 'rp'}_d{dd}", f)

    def pq(s, m, seed=0, opq=False):
        name = f"{'opq' if opq else 'pq'}{m}_s{seed}"
        def f():
            tr = s.d.X[np.random.default_rng(seed).choice(s.N, min(P["pq_n"], s.N), replace=False)]
            ix = faiss.index_factory(s.D, (f"OPQ{m}," if opq else "") + f"PQ{m}", faiss.METRIC_INNER_PRODUCT)
            base = faiss.downcast_index(ix)
            pqobj = faiss.downcast_index(base.index).pq if opq else base.pq
            pqobj.cp.seed = seed
            if opq:
                o = faiss.downcast_VectorTransform(base.chain.at(0))
                if P["opq_niter"]: o.niter = P["opq_niter"]; o.niter_pq_0 = min(o.niter_pq_0, 5)
            t = time.time(); ix.train(tr); tt = time.time() - t; ix.add(s.d.X)
            if opq:
                s.meta[name + "_train"] = dict(train_n=len(tr), train_seconds=round(tt, 1), niter=int(o.niter), niter_pq=int(o.niter_pq))
            return metrics(ix.search(s.d.Q, K)[1], s.d)
        return s.get(name, f)

    def ivf(s, seed=0, nprobe=32):
        def f():
            tr = s.d.X[np.random.default_rng(seed).choice(s.N, min(P["ivf_n"], s.N), replace=False)]
            q = faiss.IndexFlatIP(s.D); ix = faiss.IndexIVFFlat(q, s.D, P["nlist"], faiss.METRIC_INNER_PRODUCT)
            ix.cp.seed = seed; ix.train(tr); ix.add(s.d.X); ix.nprobe = nprobe
            return metrics(ix.search(s.d.Q, K)[1], s.d)
        return s.get(f"ivf{nprobe}_s{seed}", f)

    def pq_ok(s, m): return m <= s.D and s.D % m == 0 and (not P["smoke"] or m <= 32)

    def sanity(s, force=False):
        nd, _ = s.raw(); ref = RAW_REF[s.enc]; bad = []
        for x in SRC:
            if abs(s.boot.src_point(nd)[x] - ref[x]) > 2e-4: bad.append((x, s.boot.src_point(nd)[x], ref[x]))
        if abs(s.boot.macro_point(nd) - ref["macro"]) > 2e-4: bad.append(("macro", s.boot.macro_point(nd), ref["macro"]))
        if bad and not (force or P["smoke"]):
            sys.exit(f"[Model {s.enc}] raw nDCG@10 does not reproduce Table 4 {bad}. Fix the ADAPTER (embedding order, "
                     f"exclusions, gold chunk mapping, normalisation) before running anything else, or pass --force.")
        print(f"[Model {s.enc}] raw macro nDCG@10 = {s.boot.macro_point(nd):.4f} (Table 4: {ref['macro']})")


def warn_ref(tag, got, ref, tol):
    if abs(got - ref) > tol: print(f"  !! {tag}: got {got:.4f}, manuscript {ref:.4f} (|diff|>{tol}) -- different seed/sample? check before filling tables")


# ------------------------------------------------------------------ Table 8 (and recomputed Table 7)
def t8(cx):
    for enc, c in cx.items():
        rows = []
        for B in [b for b in BUD_PQ[enc] if c.pq_ok(b)]:
            cb = 256 * 4 * c.D; mp = (c.N * B + cb) / 2**20; dm = min(c.D, math.ceil((c.N * B + cb) / c.N)); ms = c.N * dm / 2**20
            (pnd, _), (snd, _) = c.pq(B), c.pcasq8(dm); df, lo, hi = c.boot.diff(pnd, snd)
            if B in T6_PQ[enc]: warn_ref(f"PQ B{B}", c.boot.macro_point(pnd), T6_PQ[enc][B], 1e-3)
            rows.append([f"B{B}", f"{mp:.3f}", f"{dm}{'*' if dm == c.D and (c.N*B+cb)/c.N > c.D else ''} / {ms:.3f}",
                         f"{c.boot.macro_point(pnd):.5f}", f"{c.boot.macro_point(snd):.5f}", sgn(df), cis(lo, hi), reading(lo, hi)])
        emit(f"table8_model{enc}" if enc == "A" else f"table7_recomputed_model{enc}",
             ["PQ budget", "PQ MiB", "PCAsq8 d / MiB", "PQ nDCG@10", "PCAsq8 nDCG@10", "Difference", "95% interval", "Reading"], rows,
             "* d capped at D (PCA+SQ8 cannot exceed D coordinates; its index is then smaller than PQ's)." if enc == "A" else
             "Check: must match Table 7 of the manuscript (differences within ~1e-3).")


# ------------------------------------------------------------------ Table 9 (OPQ)
def t9(cx):
    rows, rows2 = [], []
    for enc, c in cx.items():
        rawnd = c.raw()[0]
        for B in [b for b in (16, 32, 64) if c.pq_ok(b)]:
            (ond, _), (pnd, _), (snd, _) = c.pq(B, opq=True), c.pq(B), c.pcasq8(B)
            d1 = c.boot.diff(ond, snd); d2 = c.boot.diff(ond, pnd)
            rows.append([enc, B, f"{c.boot.macro_point(ond):.4f}", f"{sgn(d1[0])} {cis(d1[1], d1[2])}", f"{sgn(d2[0])} {cis(d2[1], d2[2])}"])
            # memory-matched variant: OPQ stores a DxD float32 rotation in addition to PQ codes + codebook
            mo = (c.N * B + 256 * 4 * c.D + 4 * c.D * c.D) / 2**20; dm = min(c.D, math.ceil(mo * 2**20 / c.N))
            d3 = c.boot.diff(ond, c.pcasq8(dm)[0])
            rows2.append([enc, B, f"{mo:.3f}", dm, f"{sgn(d3[0])} {cis(d3[1], d3[2])}", reading(d3[1], d3[2], "OPQ higher", "PCAsq8 higher")])
    emit("table9", ["Encoder", "B", "OPQ nDCG@10", "OPQ \u2212 PCAsq8 [95% CI]", "OPQ \u2212 PQ [95% CI]"], rows)
    emit("table9_memory_matched", ["Encoder", "B", "OPQ MiB (incl. rotation)", "matched PCAsq8 d", "OPQ \u2212 PCAsq8 [95% CI]", "Reading"], rows2)
    info = {f"{e}:{k}": v for e, c in cx.items() for k, v in c.meta.items() if k.endswith("_train")}
    print("OPQ training record (sample size, iterations, wall-clock) ->", json.dumps(info))
    MD.append("OPQ training record: " + json.dumps(info))


# ------------------------------------------------------------------ Table 10
def t10(cx):
    rows, extra = {}, []
    for enc, c in cx.items():
        p = c.pca(0); lam, l, u = p["lam"], p["l"], p["u"]; cs = np.cumsum(lam**2); res = []
        for B, gp in zip([32, 64, 96, 192, 384, 768], GAIN_PAPER[enc]):
            if B > c.D: continue
            tau = (cs[B - 1] - cs[B // 4 - 1]) / cs[B - 1]
            kap = float(np.mean((u[:B] - l[:B]) / np.sqrt(np.maximum(lam[:B], 1e-30)))); rho = kap**2 / (12 * 255**2)
            g = c.boot.macro_point(c.pcasq8(B)[0]) - c.boot.macro_point(c.pcaf32(B // 4)[0])
            warn_ref(f"gain {enc} B{B}", g, gp, 5e-4); res.append((B, tau, kap, rho, g, gp))
        rows[enc] = res
        r = spearmanr([x[1] for x in res], [x[4] for x in res]).correlation
        extra.append(f"Model {enc}: Spearman(tau, measured gain) = {r:.3f} over {len(res)} budgets; tau > rho_hat at {sum(x[1] > x[3] for x in res)} of {len(res)}")
    out = []
    for i, B in enumerate([x[0] for x in rows[next(iter(rows))]]):
        a = rows.get("A", [None] * 9)[i] if "A" in rows else None; b = rows.get("B", [None] * 9)[i] if "B" in rows else None
        f = lambda z, j, fmt: fmt.format(z[j]) if z else "-"
        out.append([B, f(a, 1, "{:.4f}"), f(b, 1, "{:.4f}"), f"{f(a, 2, '{:.2f}')} / {f(b, 2, '{:.2f}')}",
                    f"{f(a, 3, '{:.2e}')} / {f(b, 3, '{:.2e}')}", sgn(a[4]) if a else "-", sgn(b[4]) if b else "-"])
    emit("table10", ["B", "tau A", "tau B", "kappa_hat A / B", "rho_SQ_hat A / B", "Gain A (recomputed)", "Gain B (recomputed)"], out, "\n".join(extra))


# ------------------------------------------------------------------ Table 11
def t11(cx):
    rows, rows_dh, notes = [], [], []
    H1A = {("A", 192): "+0.0040 [0.0010, 0.0074]", ("A", 384): "+0.0070 [0.0043, 0.0102]"}
    for enc, c in cx.items():
        for B in [b for b in (192, 384, 768, 1536) if b // 4 <= c.D]:
            dd = B // 4; v = {k: f(dd) for k, f in dict(PCA=c.pcaf32, PCAu=c.pcau, RP=lambda x: c.rp(x, False), RPc=lambda x: c.rp(x, True)).items()}
            dif = lambda a, b: (lambda r: f"{sgn(r[0])} {cis(r[1], r[2])}")(c.boot.diff(v[a][0], v[b][0]))
            rows.append([enc, f"{B:,}", dif("RPc", "RP"), dif("PCA", "PCAu"), dif("PCA", "RPc"), dif("PCAu", "RP")])
            if (enc, B) in H1A: print(f"  reference H1a (PCA-RP) {enc} B{B}: manuscript {H1A[(enc, B)]}; here: {dif('PCA', 'RP')}")
            for k, (nd, dh) in v.items(): rows_dh.append([enc, B, k, f"{c.boot.macro_point(nd):.4f}", f"{c.boot.macro_point(dh):.4f}"])
        (rnd, rdh), (cnd, cdh) = c.raw(), c.rawc(); r = c.boot.diff(cnd, rnd)
        notes.append(f"Model {enc}: raw-c \u2212 raw = {sgn(r[0])} {cis(r[1], r[2])}; macro document hit raw-c {c.boot.macro_point(cdh):.4f}, raw {c.boot.macro_point(rdh):.4f}")
    emit("table11", ["Encoder", "B", "RPc \u2212 RP", "PCA \u2212 PCAu", "PCA \u2212 RPc", "PCAu \u2212 RP"], rows, "\n".join(notes))
    emit("table11_dochit_ndcg", ["Encoder", "B", "Config", "macro nDCG@10", "macro doc hit@10"], rows_dh)


# ------------------------------------------------------------------ Table 12
def t12(cx, seeds=range(5)):
    rows, ranges = [], {}
    specs = [("PCAsq8, B=192", lambda c, s: c.pcasq8(192, s), 192, "A,B"), ("PCAsq8, B=384", lambda c, s: c.pcasq8(384, s), 384, "A,B"),
             ("PQ, B=64", lambda c, s: c.pq(64, s), 64, "A,B"), ("IVF-Flat, nprobe 32, raw", lambda c, s: c.ivf(s), 0, "B")]
    for label, fn, _, encs in specs:
        for enc, c in cx.items():
            if enc not in encs: continue
            vals = [c.boot.macro_point(fn(c, sd)[0]) for sd in seeds]
            _, lo, hi = c.boot.ci(fn(c, 0)[0]); ranges[(label, enc)] = max(vals) - min(vals)
            rows.append([label, enc, f"{np.mean(vals):.4f}", f"{min(vals):.4f}\u2013{max(vals):.4f}", f"{(hi - lo) / 2:.4f}"])
    contrasts = {"Table 7 B16": .0019, "Table 7 B32": .0067, "Table 7 B64": .0083, "Table 7 B128": .0082, "Table 7 B256": .0014,
                 "Table 7 B512": .0011, "H1b smallest gain (B32, A)": .0104, "H1b smallest gain (B32, B)": .0103}
    mx = max(ranges.values()); small = [k for k, v in contrasts.items() if v < mx]
    emit("table12", ["Configuration", "Encoder", "Mean over seeds", "Min\u2013max over seeds", "Bootstrap half-width"], rows,
         f"Largest seed range = {mx:.4f}. Reported contrasts smaller than it: {small if small else 'none'}. (seed 0 half-width; seeds vary PCA sample / PQ sample+k-means seed / IVF sample+seed)")


# ------------------------------------------------------------------ Table 14 additions (R, W with CI -> certification)
def t14(cx, tR=0.95, tS=0.90):
    for enc, c in cx.items():
        rnd = c.raw()[0]; cand = []
        def add(label, v, mem_serial, mem_adj): cand.append((label, v, mem_serial, mem_adj))
        mpq = lambda m: (c.N * m + 256 * 4 * c.D) / 2**20
        adjsq = lambda dd: c.N * dd / 2**20 + (4 * c.D * dd + 4 * c.D) / 2**20
        for B in [b for b in BUD_PQ[enc] if c.pq_ok(b)]:
            add(f"PQ m={B}", c.pq(B)[0], mpq(B), mpq(B))
            dm = min(c.D, math.ceil(mpq(B) * 2**20 / c.N)); add(f"PCAsq8 d={dm} (matched to PQ m={B})", c.pcasq8(dm)[0], c.N * dm / 2**20, adjsq(dm))
        for B in [b for b in (16, 32, 64, 96, 192, 384, 768) if b <= c.D]:
            add(f"PCAsq8 B={B}", c.pcasq8(B)[0], c.N * B / 2**20, adjsq(B))
        for B in [b for b in (16, 32, 64) if c.pq_ok(b)]:
            add(f"OPQ m={B}", c.pq(B, opq=True)[0], mpq(B) + 4 * c.D * c.D / 2**20, mpq(B) + 4 * c.D * c.D / 2**20)
        rows = []
        for label, v, ms, ma in sorted(cand, key=lambda x: x[3]):
            Rm, W = c.boot.retention(v, rnd); R0 = c.boot.macro_point(v) / c.boot.macro_point(rnd)
            Rl, Rh = np.percentile(Rm, [2.5, 97.5]); Wl, Wh = np.percentile(W, [2.5, 97.5])
            cert = (Rl >= tR) and (Wl >= tS)
            rows.append([label, f"{ms:.1f}", f"{ma:.1f}", f"{R0:.3f} [{Rl:.3f}, {Rh:.3f}]", f"{W.mean():.3f} [{Wl:.3f}, {Wh:.3f}]", "YES" if cert else "no"])
        best = next((r for r in rows if r[-1] == "YES"), None)
        emit(f"table14_extra_model{enc}", ["Configuration", "MiB serialised", "MiB adjusted", "R [95% CI]", "W [95% CI]", f"Certified (R_L>={tR}, W_L>={tS})"], rows,
             f"Smallest certified configuration (adjusted memory): {best[0] + ', ' + best[2] + ' MiB' if best else 'none (raw is the only certified index)'}")


# ------------------------------------------------------------------ probe + main
def probe():
    print("cwd:", os.getcwd(), "| ROOT:", ROOT)
    for p in ("data", "data_lbrag", "paper_results", "out_full"):
        fp = os.path.join(ROOT, p); print(f"\n[{p}]", sorted(os.listdir(fp))[:60] if os.path.isdir(fp) else "missing")
    for k in ("chunks", "queries"):
        fp = os.path.join(ROOT, PATHS[k])
        if os.path.exists(fp):
            df = pd.read_parquet(fp); print(f"\n{k}: {df.shape}\n{df.dtypes}\n{df.head(2).T}")
        else: print(f"\n{k}: NOT FOUND at {fp}")
    for e in "AB":
        for k in ("emb", "qemb"):
            fp = os.path.join(ROOT, PATHS[k][e]); print(k, e, np.load(fp, mmap_mode="r").shape if os.path.exists(fp) else f"NOT FOUND {fp}")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", nargs="+"); ap.add_argument("--enc", default="A,B")
    ap.add_argument("--threads", type=int, default=8); ap.add_argument("--force", action="store_true"); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--opq-niter", type=int, default=None, help="override OPQ iterations (default: faiss 50)")
    a = ap.parse_args(); faiss.omp_set_num_threads(a.threads)
    P["opq_niter"] = a.opq_niter
    if a.smoke: P.update(smoke=True, n_boot=100, pq_n=2000, ivf_n=2000, nlist=16, opq_niter=2)
    if a.cmd == ["probe"]: return probe()
    os.makedirs(OUT, exist_ok=True); cmds = ["t8", "t9", "t10", "t11", "t12", "t14"] if "all" in a.cmd else a.cmd
    cx = {e: Ctx(e) for e in a.enc.split(",")}
    for c in cx.values(): c.sanity(a.force)
    for name in cmds: print(f"\n===== {name} ====="); globals()[name](cx)
    with open(os.path.join(OUT, "tables.md"), "a", encoding="utf-8") as f: f.write("\n".join(MD))
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(dict(faiss=faiss.__version__, numpy=np.__version__, boot=P, meta={e: c.meta for e, c in cx.items()}), f, indent=1, default=str)


if __name__ == "__main__":
    main()
