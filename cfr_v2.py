#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cfr.py -- Compression Fragility of dense Retrieval over long documents
단계: explore(탐색) -> freeze(가설 동결) -> confirm(새 모델/새 데이터 확증)
"""
from __future__ import annotations
import argparse, gc, hashlib, json, pickle, re, sys, time, warnings
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd
import faiss
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
warnings.filterwarnings("ignore")

MET = ["ndcg", "recall", "mrr", "dochit", "p1", "hit10"]
MI = {m: i for i, m in enumerate(MET)}
LOG2 = np.log2(np.arange(2, 13))

HYP_SPEC = {
 "_decision_rule": "Universal claims (all budgets / all sources) are supported only if EVERY non-excluded row passes Holm-adjusted p<alpha. Sources with fewer than min_clusters contracts are excluded from the verdict and from macro averages (reported per source only).",
 "H1a": "At equal bytes/vector, PCA (float32, d=B/4) > random projection (float32, d=B/4) in macro nDCG@10, all budgets.",
 "H1b": "At equal bytes/vector (B>=32), PCA+SQ8 (d=B) > PCA float32 (d=B/4), all budgets.",
 "H1c": "At B<=64 bytes/vector, OPQ (m=B) > PCA+SQ8 (d=B).",
 "H2a": "DIAGNOSTIC, not a hypothesis: flip probability vs z = margin*sqrt(d); reported as model fit (R2, calibration bins).",
 "H2b": "Adding source indicators to the z-only model raises deviance-R2 by less than delta_r2_max: supported only if the upper limit of the contract-cluster bootstrap CI of delta_R2 is below delta_r2_max.",
 "H3a": "Identification share (oracle-global)/oracle at full precision is larger for the focus source than for every other non-excluded source.",
 "H3b": "RP d=48 raises the identification share relative to full precision in every non-excluded source.",
 "H4":  "Cross-source top-1 confusion rate of global dense search rises under compression (RPf32 d=48 vs raw) in every non-excluded source. typed_vs_xsrc.csv reports typed-pool gain next to the confusion rate descriptively (typed gain is a pool-restriction effect by construction).",
 "H5":  "Two-stage retrieval (doc-level BM25 top-3 -> dense chunk ranking) > global dense, macro nDCG. Scope: queries name the contract (template), so this is the effect when the query identifies the document.",
}

# ------------------------------------------------------------------ args
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data"); p.add_argument("--model", default="A")
    p.add_argument("--out", default="out_cfr")
    p.add_argument("--stage", choices=["explore", "freeze", "confirm"], default="explore")
    p.add_argument("--prereg", default="prereg.json"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--B", type=int, default=4000)
    p.add_argument("--budgets", default="16,32,64,96,192,384,768,1536")
    p.add_argument("--rp_draws", type=int, default=3)
    p.add_argument("--max_q_per_source", type=int, default=0)
    p.add_argument("--ext_reps", default="raw,PCAsq8_B192")
    p.add_argument("--skip_opq", action="store_true"); p.add_argument("--skip_ann", action="store_true")
    p.add_argument("--nlist", type=int, default=1024); p.add_argument("--threads", type=int, default=0)
    p.add_argument("--focus_source", default="contractnli")
    p.add_argument("--delta_r2_max", type=float, default=0.02); p.add_argument("--alpha", type=float, default=0.05); p.add_argument("--min_clusters", type=int, default=20)
    p.add_argument("--make_emb", default=""); p.add_argument("--q_prefix", default="")
    p.add_argument("--d_prefix", default=""); p.add_argument("--device", default="cuda")
    return p.parse_args()

FROZEN = ["B", "budgets", "rp_draws", "max_q_per_source", "ext_reps", "nlist",
          "focus_source", "delta_r2_max", "alpha", "min_clusters"]
def sha_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def freeze(a):
    spec = {"created_utc": datetime.now(timezone.utc).isoformat(), "explore_model": a.model,
            "script_sha256": sha_file(__file__), "params": {k: getattr(a, k) for k in FROZEN},
            "hypotheses": HYP_SPEC}
    Path(a.prereg).write_text(json.dumps(spec, ensure_ascii=False, indent=1, sort_keys=True))
    print("frozen ->", a.prereg, "| file sha256 =", sha_file(a.prereg))
    print("지금 git commit/tag (가능하면 OSF 업로드) 한 뒤에 confirm 을 실행하세요.")

def load_prereg(a):
    s = json.loads(Path(a.prereg).read_text())
    if s["script_sha256"] != sha_file(__file__):
        sys.exit("스크립트가 동결 이후 수정됨 -> 확증 불가. 동결 버전으로 되돌리세요.")
    for k, v in s["params"].items(): setattr(a, k, v)
    if s["explore_model"] == a.model:
        print("[경고] 탐색에 쓴 모델과 동일 -> 독립 확증 아님")
    return s

# ------------------------------------------------------------------ io
def _tab(d, stem):
    for e in (".parquet", ".pkl", ".jsonl"):
        f = Path(d) / f"{stem}{e}"
        if f.exists():
            return pd.read_parquet(f) if e == ".parquet" else pd.read_pickle(f) if e == ".pkl" else pd.read_json(f, lines=True)
    raise FileNotFoundError(f"{d}/{stem}.(parquet|pkl|jsonl)")

def make_emb(a):
    from sentence_transformers import SentenceTransformer
    d = Path(a.data_dir); ch, qu = _tab(d, "chunks"), _tab(d, "queries")
    m = SentenceTransformer(a.make_emb, device=a.device)
    o = d / "emb" / a.model; o.mkdir(parents=True, exist_ok=True)
    enc = lambda t, pre: m.encode([pre + x for x in t], batch_size=64, normalize_embeddings=True, show_progress_bar=True)
    np.save(o / "chunks.npy", enc(ch.text.astype(str).tolist(), a.d_prefix).astype("float32"))
    np.save(o / "queries.npy", enc(qu.text.astype(str).tolist(), a.q_prefix).astype("float32"))

class Ctx: pass

def _nrm(A): return (A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)).astype("float32")

def load_ctx(a):
    d = Path(a.data_dir); c = Ctx()
    ch, qu = _tab(d, "chunks").reset_index(drop=True), _tab(d, "queries").reset_index(drop=True)
    X = np.load(d / "emb" / a.model / "chunks.npy").astype("float32")
    Q = np.load(d / "emb" / a.model / "queries.npy").astype("float32")
    assert len(ch) == len(X) and len(qu) == len(Q), "행 수 불일치(정렬 가드)"
    names = sorted(ch.source.astype(str).unique()); sc = {s: i for i, s in enumerate(names)}
    sidx = ch.source.astype(str).map(sc).to_numpy()
    dcode, _ = pd.factorize(ch.source.astype(str) + "||" + ch.doc_id.astype(str), sort=True)
    order = np.lexsort((np.arange(len(ch)), dcode, sidx))
    inv = np.empty_like(order); inv[order] = np.arange(len(order))
    X = _nrm(X[order]); ch = ch.iloc[order].reset_index(drop=True)
    dc, dn = pd.factorize(ch.source.astype(str) + "||" + ch.doc_id.astype(str))
    c.doc_of_chunk = dc.astype(np.int64); c.n_docs = len(dn)
    c.doc_start = np.r_[0, np.flatnonzero(np.diff(c.doc_of_chunk)) + 1]
    c.doc_end = np.r_[c.doc_start[1:], len(ch)]
    csrc = ch.source.astype(str).map(sc).to_numpy()
    c.chunk_src = csrc
    c.doc_src = csrc[c.doc_start]
    c.src_start = np.array([c.doc_start[c.doc_src == s].min() for s in range(len(names))])
    c.src_end = np.array([c.doc_end[c.doc_src == s].max() for s in range(len(names))])
    gold = [inv[np.asarray(g, dtype=int)] for g in qu.gold_chunks]
    keep = np.array([len(g) > 0 for g in gold])
    _seen, _dup = set(), np.zeros(len(qu), bool)
    for i, (s_, t_, g_) in enumerate(zip(qu.source.astype(str), qu.text.astype(str), gold)):
        k_ = (s_, t_, tuple(sorted(int(x) for x in g_)))
        if k_ in _seen: _dup[i] = True
        _seen.add(k_)
    print(f"[data] 동일 질의문+동일 정답 중복 제거: {int((_dup & keep).sum())}건")
    keep &= ~_dup
    if a.max_q_per_source:
        rng = np.random.default_rng(a.seed); sel = np.zeros(len(qu), bool)
        for s in names:
            ii = np.flatnonzero((qu.source.astype(str) == s).to_numpy() & keep)
            sel[rng.choice(ii, min(len(ii), a.max_q_per_source), replace=False)] = True
        keep &= sel
    ki = np.flatnonzero(keep)
    c.queries = qu.iloc[ki].reset_index(drop=True); c.gold = [gold[i] for i in ki]
    c.Q = _nrm(Q[ki]); c.X = X; c.chunks = ch; c.texts = ch.text.astype(str).tolist()
    c.src_names = names; c.qsrc = c.queries.source.astype(str).map(sc).to_numpy()
    c.qdoc = np.array([c.doc_of_chunk[g[0]] for g in c.gold])
    c.N, c.D, c.nq = len(X), X.shape[1], len(c.queries)
    print(f"[data] N={c.N} D={c.D} nq={c.nq} docs={c.n_docs} sources={names}")
    return c

# ------------------------------------------------------------------ metrics / engine
def topk_idx(s, k=10):
    if len(s) <= k: return np.argsort(-s)
    p = np.argpartition(-s, k - 1)[:k]; return p[np.argsort(-s[p])]

def rank_in(s, idx):
    if isinstance(idx, slice): return topk_idx(s[idx]) + idx.start
    return idx[topk_idx(s[idx])]

def q_metrics(r, gold, gdoc, c):
    k = len(r); rel = np.isin(r, gold)
    dcg = float((rel / LOG2[:k]).sum()); idcg = float((1.0 / LOG2[:min(10, len(gold))]).sum())
    f = np.flatnonzero(rel)
    return np.array([dcg / idcg, rel.sum() / len(gold), 1.0 / (f[0] + 1) if len(f) else 0.0,
                     float((c.doc_of_chunk[r] == gdoc).any()), float(rel[0]), float(rel.any())])

def dense_iter(Xh, Qh, bs=256):
    for a in range(0, len(Qh), bs):
        yield np.arange(a, min(a + bs, len(Qh))), Qh[a:a + bs] @ Xh.T

class BM25:
    def __init__(self, texts, stop=None, k1=1.2, b=0.75):
        self.cv = CountVectorizer(lowercase=True, stop_words=stop, token_pattern=r"(?u)\b\w\w+\b", dtype=np.float32)
        tf = self.cv.fit_transform(texts).tocsr(); n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1 + (n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(1)).ravel(); tf = tf.tocoo()
        w = idf[tf.col] * tf.data * (k1 + 1) / (tf.data + k1 * (1 - b + b * dl[tf.row] / dl.mean()))
        self.W = sparse.csr_matrix((w.astype(np.float32), (tf.row, tf.col)), shape=tf.shape).T.tocsr()
    def score(self, qt): return self.cv.transform(qt) @ self.W

def build_bm25_doc(c):
    docs = c.chunks.groupby(c.doc_of_chunk)["text"].agg(" ".join).sort_index().tolist()
    bm = BM25(docs); qs = c.queries.text.astype(str).tolist()
    M = np.zeros((c.nq, c.n_docs), np.float32)
    for a in range(0, c.nq, 512): M[a:a + 512] = bm.score(qs[a:a + 512]).toarray()
    return M

_rk = lambda x: (-x).argsort().argsort() + 1

def run_modes(c, it, modes, bm25_doc=None, rng=None, ks=(1, 3, 5)):
    s1 = ["dense"] + (["bm25", "rrf"] if bm25_doc is not None else [])
    tsn = [f"ts_{n}_k{k}" for n in s1 for k in ks] if "ts" in modes else []
    names = [m for m in modes if m != "ts"] + tsn
    out = {n: np.zeros((c.nq, len(MET))) for n in names}; N = c.N
    if "global" in modes: out["xsrc"] = np.zeros((c.nq, 1))
    for idxs, S in it:
        for j, qi in enumerate(idxs):
            s, g, gd, sr = S[j], c.gold[qi], c.qdoc[qi], c.qsrc[qi]
            if "global" in modes:
                rg = rank_in(s, slice(0, N)); out["global"][qi] = q_metrics(rg, g, gd, c)
                out["xsrc"][qi, 0] = float(c.chunk_src[rg[0]] != sr)
            if "typed" in modes: out["typed"][qi] = q_metrics(rank_in(s, slice(c.src_start[sr], c.src_end[sr])), g, gd, c)
            if "oracle" in modes: out["oracle"][qi] = q_metrics(rank_in(s, slice(c.doc_start[gd], c.doc_end[gd])), g, gd, c)
            if "randpool" in modes:
                k = c.src_end[sr] - c.src_start[sr]
                pool = np.union1d(rng.choice(N, min(k, N), replace=False), g)
                out["randpool"][qi] = q_metrics(rank_in(s, pool), g, gd, c)
            if tsn:
                dd = np.maximum.reduceat(s, c.doc_start); cand = {"dense": dd}
                if bm25_doc is not None:
                    cand["bm25"] = bm25_doc[qi]; cand["rrf"] = 1 / (60 + _rk(bm25_doc[qi])) + 1 / (60 + _rk(dd))
                for nme, sc_ in cand.items():
                    order = np.argsort(-sc_)[:max(ks)]
                    for k in ks:
                        idx = np.concatenate([np.arange(c.doc_start[d], c.doc_end[d]) for d in order[:k]])
                        out[f"ts_{nme}_k{k}"][qi] = q_metrics(rank_in(s, idx), g, gd, c)
    return out

def lists_metrics(c, R):
    o = np.zeros((c.nq, len(MET)))
    for qi in range(c.nq):
        r = R[qi][R[qi] >= 0]; o[qi] = q_metrics(r, c.gold[qi], c.qdoc[qi], c)
    return o

# ------------------------------------------------------------------ cluster bootstrap
class Boot:
    def __init__(self, c, B, seed, min_clusters=0):
        rng = np.random.default_rng(seed); self.c = c; self.parts = {}
        for s, nm in enumerate(c.src_names):
            m = np.flatnonzero(c.qsrc == s)
            if len(m) == 0: continue
            docs, pos = np.unique(c.qdoc[m], return_inverse=True)
            W = rng.multinomial(len(docs), np.full(len(docs), 1 / len(docs)), size=B).astype(np.float64)
            self.parts[nm] = (m, pos, len(docs), W)
        self.core = [k for k, v in self.parts.items() if v[2] >= min_clusters] or list(self.parts)
    def draws(self, v):
        r = {}
        for nm, (m, pos, nd, W) in self.parts.items():
            sd = np.bincount(pos, weights=v[m], minlength=nd); nn = np.bincount(pos, minlength=nd)
            r[nm] = (W @ sd) / np.maximum(W @ nn, 1e-12)
        r["macro"] = np.mean([r[k] for k in self.core], axis=0); return r
    def point(self, v):
        r = {nm: float(v[m].mean()) for nm, (m, *_ ) in self.parts.items()}
        r["macro"] = float(np.mean([r[k] for k in self.core])); return r

ci = lambda d: tuple(np.percentile(d, [2.5, 97.5]))
def p_gt(d): return (np.sum(d <= 0) + 1) / (len(d) + 1)
def p_two(d): return min(1.0, 2 * min((np.sum(d <= 0) + 1) / (len(d) + 1), (np.sum(d >= 0) + 1) / (len(d) + 1)))
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for i, k in enumerate(o): run = max(run, (m - i) * p[k]); adj[k] = min(1, run)
    return adj

# ------------------------------------------------------------------ representations
class Builder:
    def __init__(self, c, seed, threads):
        self.c, self.seed = c, seed; rng = np.random.default_rng(seed)
        self.tr = rng.choice(c.N, min(c.N, 100_000), replace=False)
        Xs = c.X[self.tr]; self.mu = Xs.mean(0); Xc = Xs - self.mu
        w, V = np.linalg.eigh((Xc.T @ Xc) / len(Xs)); self.V = V[:, ::-1].astype("float32")
        self.threads = threads or faiss.omp_get_max_threads()
    def proj(self, kind, d, draw):
        if kind == "RP":
            R = (np.random.default_rng([self.seed, d, draw]).standard_normal((self.c.D, d)) / np.sqrt(d)).astype("float32")
            return lambda A: A @ R
        return lambda A: (A - self.mu) @ self.V[:, :d]
    def make(self, fam, B, draw=0):
        c = self.c; X, Q, N, D = c.X, c.Q, c.N, c.D; IP = faiss.METRIC_INNER_PRODUCT
        if fam == "raw":
            idx = faiss.IndexFlatIP(D); idx.add(X); return X, Q, idx, D
        if fam in ("PQ", "OPQ"):
            idx = faiss.IndexPQ(D, B, 8, IP) if fam == "PQ" else faiss.index_factory(D, f"OPQ{B},PQ{B}x8", IP)
            idx.train(np.ascontiguousarray(X[self.tr[:50000]])); idx.add(X)
            return idx.reconstruct_n(0, N), Q, idx, D
        kind = "RP" if fam.startswith("RP") else "PCA"; f32 = fam.endswith("f32"); d = B // 4 if f32 else B
        fn = self.proj(kind, d, draw); Xp, Qp = _nrm(fn(X)), _nrm(fn(c.Q))
        if f32:
            idx = faiss.IndexFlatIP(d); idx.add(Xp); return Xp, Qp, idx, d
        idx = faiss.IndexScalarQuantizer(d, faiss.ScalarQuantizer.QT_8bit, IP)
        idx.train(np.ascontiguousarray(Xp[self.tr])); idx.add(Xp)
        return idx.reconstruct_n(0, N), Qp, idx, d

def time_index(idx, Qh, threads, n=200):
    faiss.omp_set_num_threads(1); q = np.ascontiguousarray(Qh[:n]); t = []
    for i in range(n):
        t0 = time.perf_counter(); idx.search(q[i:i + 1], 10); t.append((time.perf_counter() - t0) * 1e3)
    faiss.omp_set_num_threads(threads); return float(np.median(t)), float(np.percentile(t, 95))

mem_mib = lambda idx: faiss.serialize_index(idx).size / 2 ** 20

def cfg_names(budgets, D, skip_opq):
    o = ["raw"]
    for B in budgets:
        if B % 4 == 0 and 1 <= B // 4 <= D: o += [f"RPf32_B{B}", f"PCAf32_B{B}"]
        if B <= D: o += [f"RPsq8_B{B}", f"PCAsq8_B{B}"]
        if B <= D and D % B == 0: o += [f"PQ_B{B}"] + ([] if skip_opq or B > 64 else [f"OPQ_B{B}"])
    return o

def eval_named(c, bld, name, ext, a, cdir, bm25_doc):
    modes = ["global", "typed", "oracle"] + (["ts"] if name in ext else [])
    cp = cdir / f"{name}.pkl"
    if cp.exists():
        r = pickle.load(open(cp, "rb"))
        if set(r["modes"]) == set(modes) and "xsrc" in r["perq"]: return r
    fam, B = ("raw", None) if name == "raw" else (name.rsplit("_B", 1)[0], int(name.rsplit("_B", 1)[1]))
    nd = a.rp_draws if fam.startswith("RP") else 1; acc = None
    for dr in range(nd):
        Xh, Qh, idx, d = bld.make(fam, B, dr)
        if dr == 0:
            mem = mem_mib(idx); lat = time_index(idx, Qh, bld.threads); dd = d
        res = run_modes(c, dense_iter(Xh, Qh), modes, bm25_doc, np.random.default_rng(a.seed + 99 + dr))
        acc = res if acc is None else {k: acc[k] + res[k] for k in acc}
        del Xh, Qh, idx; gc.collect()
    r = dict(name=name, fam=fam, B=B, d=dd, mem=mem, lat=lat[0], lat95=lat[1], modes=modes,
             perq={k: v / nd for k, v in acc.items()})
    pickle.dump(r, open(cp, "wb")); print(f"  {name:14s} d={dd:4d} mem={mem:8.1f}MiB lat={lat[0]:.2f}ms "
          f"ndcg={r['perq']['global'][:, 0].mean():.4f}"); return r

def bm25_rec(c, stop):
    bm = BM25(c.texts, stop); qs = c.queries.text.astype(str).tolist()
    def it():
        for a in range(0, c.nq, 256):
            yield np.arange(a, min(a + 256, c.nq)), bm.score(qs[a:a + 256]).toarray()
    return dict(name="BM25" + ("_sw" if stop else ""), fam="BM25", B=None, d=None, mem=None, lat=None, lat95=None,
                modes=["global", "typed", "oracle"], perq=run_modes(c, it(), ["global", "typed", "oracle"]))

# ------------------------------------------------------------------ analyses
def frontier_table(recs, boot):
    names = [n for n in recs if recs[n]["mem"] is not None]
    dr = {n: boot.draws(recs[n]["perq"]["global"][:, 0]) for n in names}
    nd = np.stack([dr[n]["macro"] for n in names]); mem = np.array([recs[n]["mem"] for n in names])
    dom = np.zeros_like(nd, bool)
    for i in range(len(names)):
        bm = mem <= mem[i]; st = mem < mem[i]
        dom[i] = ((nd[bm] >= nd[i]) & ((nd[bm] > nd[i]) | st[bm][:, None])).any(0)
    rows = []
    for i, n in enumerate(names):
        r = recs[n]; ref = dr["raw"]["macro"]; ret = dr[n]["macro"] / ref
        pt = boot.point(r["perq"]["global"][:, 0]); row = dict(name=n, fam=r["fam"], B_bytes=r["B"], d=r["d"],
              mem_mib=r["mem"], lat_ms=r["lat"], lat_p95=r["lat95"], ndcg_macro=pt["macro"],
              lo=ci(dr[n]["macro"])[0], hi=ci(dr[n]["macro"])[1], retention=float(np.mean(ret)),
              ret_lo=ci(ret)[0], ret_hi=ci(ret)[1], pareto_prob=1 - dom[i].mean(),
              dochit_macro=boot.point(r["perq"]["global"][:, 3])["macro"])
        row.update({f"ndcg_{k}": v for k, v in pt.items() if k != "macro"}); rows.append(row)
    return pd.DataFrame(rows)

CONTR = [("PCAf32", "RPf32"), ("PCAsq8", "PCAf32"), ("RPsq8", "RPf32"), ("OPQ", "PCAsq8"),
         ("PQ", "PCAsq8"), ("OPQ", "PQ"), ("OPQ", "RPf32"), ("PCAsq8", "RPf32")]
def contrasts(recs, boot, budgets):
    rows = []
    for a_, b_ in CONTR:
        for B in budgets:
            na, nb = f"{a_}_B{B}", f"{b_}_B{B}"
            if na not in recs or nb not in recs: continue
            va, vb = recs[na]["perq"]["global"][:, 0], recs[nb]["perq"]["global"][:, 0]
            d = boot.draws(va)["macro"] - boot.draws(vb)["macro"]
            pa, pb = boot.point(va), boot.point(vb); ps = [pa[k] - pb[k] for k in boot.core]
            loso = [np.mean([x for j, x in enumerate(ps) if j != i]) for i in range(len(ps))]
            rows.append(dict(a=a_, b=b_, B=B, diff=pa["macro"] - pb["macro"], ci_lo=ci(d)[0], ci_hi=ci(d)[1],
                             p_gt=p_gt(d), p_two=p_two(d), loso_min=min(loso), loso_max=max(loso)))
    return pd.DataFrame(rows)

def idshare_draws(rec, boot):
    G = boot.draws(rec["perq"]["global"][:, 0]); O = boot.draws(rec["perq"]["oracle"][:, 0])
    return {k: (O[k] - G[k]) / np.maximum(O[k], 1e-12) for k in boot.parts}, G, O

def decomposition(recs, boot):
    rows = []; ref = recs["raw"]; _, Gr, Or = idshare_draws(ref, boot)
    for n in recs:
        if n != "raw" and not (recs[n]["fam"] in ("RPf32", "PCAf32")): continue
        sh, G, O = idshare_draws(recs[n], boot)
        for s in boot.parts:
            dG = Gr[s] - G[s]; Lloc = Or[s] - O[s]; Lid = dG - Lloc
            rows.append(dict(rep=n, d=recs[n]["d"], source=s, ndcg_global=G[s].mean(), ndcg_oracle=O[s].mean(),
                             id_share=sh[s].mean(), id_lo=ci(sh[s])[0], id_hi=ci(sh[s])[1],
                             loss_total=dG.mean(), loss_localization=Lloc.mean(), loss_identification=Lid.mean(),
                             lid_lo=ci(Lid)[0], lid_hi=ci(Lid)[1]))
    return pd.DataFrame(rows)

def compute_margins(c, bs=256):
    n = c.nq; m = np.zeros(n); sg = np.zeros(n); same = np.zeros(n)
    for idxs, S in dense_iter(c.X, c.Q, bs):
        for j, qi in enumerate(idxs):
            s, g = S[j], c.gold[qi]; sg[qi] = s[g].max(); sv = s[g].copy(); s[g] = -np.inf
            a = int(s.argmax()); m[qi] = sg[qi] - s[a]; same[qi] = float(c.doc_of_chunk[a] == c.qdoc[qi]); s[g] = sv
    return dict(margin=m, s_gold=sg, comp_same_doc=same)

def query_features(c):
    rows = []
    for qi, t in enumerate(c.queries.text.astype(str)):
        tk = re.findall(r"\w+", t); n = max(len(tk), 1)
        ncap = sum(1 for w in tk[1:] if w[0].isupper()); ndig = sum(1 for w in tk if any(ch.isdigit() for ch in w))
        qs = {w.lower() for w in tk}; ov = 0.0
        for g in c.gold[qi][:5]:
            cs = {w.lower() for w in re.findall(r"\w+", c.texts[g])}; ov = max(ov, len(qs & cs) / max(len(qs), 1))
        rows.append((np.log(n), (ncap + ndig) / n, ov))
    f = np.array(rows); sz = (c.doc_end - c.doc_start)[c.qdoc]
    return pd.DataFrame(dict(qlen=f[:, 0], id_density=f[:, 1], overlap=f[:, 2], log_doc_chunks=np.log(sz),
                             log_n_gold=np.log([len(g) for g in c.gold])))

def margin_analysis(c, recs, mg, a):
    import statsmodels.api as sm
    base = recs["raw"]["perq"]["global"][:, MI["p1"]] == 1; rows = []
    for d in (384, 192, 96, 48):
        nm = f"RPf32_B{4 * d}"
        if nm not in recs: continue
        rows.append(pd.DataFrame(dict(flip=1 - recs[nm]["perq"]["global"][:, MI["p1"]], z=mg["margin"] * np.sqrt(d),
                  src=[c.src_names[i] for i in c.qsrc], doc=c.qdoc, d=d))[base])
    if not rows: return None
    L = pd.concat(rows, ignore_index=True)
    _nd = L.groupby("src")["doc"].nunique(); L = L[L.src.isin(_nd[_nd >= a.min_clusters].index)].reset_index(drop=True)
    grp = pd.factorize(L["doc"])[0]
    X0 = sm.add_constant(L[["z"]]); fam = sm.families.Binomial()
    X1 = pd.concat([X0, pd.get_dummies(L["src"], prefix="src", drop_first=True).astype(float)], axis=1)
    m0 = sm.GLM(L["flip"], X0, family=fam).fit(cov_type="cluster", cov_kwds={"groups": grp})
    m1 = sm.GLM(L["flip"], X1, family=fam).fit(cov_type="cluster", cov_kwds={"groups": grp})
    r2 = lambda m: 1 - m.deviance / m.null_deviance
    names = list(m1.params.index); dum = [i for i, n in enumerate(names) if n.startswith("src_")]
    R = np.zeros((len(dum), len(names)))
    for k, i in enumerate(dum): R[k, i] = 1
    pw = float(m1.wald_test(R, scalar=True).pvalue) if dum else np.nan
    bz = float(m0.params["z"]); pz = float(m0.pvalues["z"]); p1s = pz / 2 if bz < 0 else 1 - pz / 2
    res = dict(n_rows=len(L), coef_z=bz, p_z_one_sided=p1s, r2_z_only=r2(m0), r2_with_source=r2(m1),
               delta_r2=r2(m1) - r2(m0), wald_source_p=pw)
    # H2b: 계약서(문서) 단위 클러스터 부트스트랩으로 delta_R2 의 CI
    rng = np.random.default_rng(a.seed + 11); ud = np.unique(grp)
    by = {g_: np.flatnonzero(grp == g_) for g_ in ud}
    A0, A1, yy = X0.to_numpy(float), X1.to_numpy(float), L["flip"].to_numpy(float); dr2 = []
    for _ in range(300):
        ii = np.concatenate([by[g_] for g_ in rng.choice(ud, len(ud), replace=True)])
        try:
            f0 = sm.GLM(yy[ii], A0[ii], family=fam).fit(); f1 = sm.GLM(yy[ii], A1[ii], family=fam).fit()
            dr2.append(r2(f1) - r2(f0))
        except Exception: pass
    res.update(dr2_lo=float(np.percentile(dr2, 2.5)) if dr2 else np.nan,
               dr2_hi=float(np.percentile(dr2, 97.5)) if dr2 else np.nan, dr2_nboot=len(dr2))
    L["zbin"] = pd.qcut(L.z, 10, duplicates="drop").astype(str)
    bins = L.groupby(["zbin", "src"]).flip.agg(["mean", "count"]).reset_index()
    return res, bins

def fragility_ml(c, recs, mg, qf, reps, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score, brier_score_loss
    from sklearn.inspection import permutation_importance
    F = qf.copy(); F["margin"] = mg["margin"]; F["s_gold"] = mg["s_gold"]; F["comp_same_doc"] = mg["comp_same_doc"]
    for i, s in enumerate(c.src_names): F[f"src_{s}"] = (c.qsrc == i).astype(float)
    base = recs["raw"]["perq"]["global"][:, MI["hit10"]] == 1; rows, imps = [], []
    for nm in reps:
        if nm not in recs: continue
        y = (recs[nm]["perq"]["global"][:, MI["hit10"]] < 0.5).astype(int)[base]
        Fb, g = F[base].reset_index(drop=True), c.qdoc[base]
        if y.sum() < 30 or len(np.unique(g)) < 5: print(f"  [ML] {nm}: 양성 부족, 건너뜀"); continue
        gkf = GroupKFold(n_splits=min(5, len(np.unique(g)))); res = {"hgb": [], "lr_margin": []}; im = []
        for tr, te in gkf.split(Fb, y, g):
            if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2: continue
            h = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, random_state=seed).fit(Fb.iloc[tr], y[tr])
            ph = h.predict_proba(Fb.iloc[te])[:, 1]; res["hgb"].append((roc_auc_score(y[te], ph), brier_score_loss(y[te], ph)))
            l = LogisticRegression(max_iter=1000).fit(Fb.iloc[tr][["margin"]], y[tr]); pl = l.predict_proba(Fb.iloc[te][["margin"]])[:, 1]
            res["lr_margin"].append((roc_auc_score(y[te], pl), brier_score_loss(y[te], pl)))
            pi = permutation_importance(h, Fb.iloc[te], y[te], scoring="roc_auc", n_repeats=5, random_state=seed)
            im.append(pi.importances_mean)
        for k, v in res.items():
            if v: v = np.array(v); rows.append(dict(rep=nm, model=k, n=len(y), pos_rate=y.mean(), auc=v[:, 0].mean(),
                                                    auc_sd=v[:, 0].std(), brier=v[:, 1].mean()))
        if im:
            for f_, v in zip(Fb.columns, np.mean(im, 0)): imps.append(dict(rep=nm, feature=f_, perm_importance_auc=v))
    return pd.DataFrame(rows), pd.DataFrame(imps)

def run_ann(c, bld, a, boot):
    rows = []; thr = bld.threads
    for fam, B in [("raw", None), ("PCAf32", 768), ("RPf32", 768)]:
        Xh, Qh, flat, d = bld.make(fam, B, 0); nm = "raw" if fam == "raw" else f"{fam}_B{B}"
        _, I0 = flat.search(Qh, 10); m0 = lists_metrics(c, I0); base = boot.draws(m0[:, 0])
        variants = [("Flat", "-", flat)]
        quant = faiss.IndexFlatIP(d); ivf = faiss.IndexIVFFlat(quant, d, a.nlist, faiss.METRIC_INNER_PRODUCT)
        ivf.train(np.ascontiguousarray(Xh[bld.tr[:40 * a.nlist]])); ivf.add(Xh)
        for npb in (8, 32, 128): variants.append(("IVF", f"nprobe={npb}", (ivf, npb)))
        hn = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT); hn.hnsw.efConstruction = 128; hn.add(Xh)
        variants.append(("HNSW", "efS=64", (hn, 64)))
        for kind, par, obj in variants:
            idx = obj if kind == "Flat" else obj[0]
            if kind == "IVF": idx.nprobe = obj[1]
            if kind == "HNSW": idx.hnsw.efSearch = obj[1]
            _, I = idx.search(Qh, 10); mt = lists_metrics(c, I); dd = boot.draws(mt[:, 0])
            ratio = dd["macro"] / base["macro"]; lat = time_index(idx, Qh, thr)[0]
            ov = np.mean([len(set(I[i]) & set(I0[i])) / 10 for i in range(c.nq)])
            rows.append(dict(rep=nm, index=kind, param=par, ndcg_macro=mt[:, 0].mean(), ratio_to_flat=float(np.mean(ratio)),
                             r_lo=ci(ratio)[0], r_hi=ci(ratio)[1], overlap10=ov, lat_ms=lat, mem_mib=mem_mib(idx)))
        del Xh, Qh, flat, ivf, hn; gc.collect()
    df = pd.DataFrame(rows); return df

# ------------------------------------------------------------------ hypotheses
def test_hypotheses(recs, boot, ctr, mres, a, label):
    rows = []; budgets = [int(x) for x in a.budgets.split(",")]
    small = {k for k, v in boot.parts.items() if v[2] < a.min_clusters}
    if small: print(f"[판정 제외] 계약서 수 < {a.min_clusters}: {sorted(small)}")
    def add(h, desc, est, lo, hi, p, ex=False):
        rows.append(dict(id=h, test=desc, estimate=est, ci_lo=lo, ci_hi=hi, p_one_sided=p, excluded=ex))
    def cr(x, y, B):
        r = ctr[(ctr.a == x) & (ctr.b == y) & (ctr.B == B)] if len(ctr) else ctr
        return None if len(r) == 0 else r.iloc[0]
    for h, x, y, Bs in [("H1a", "PCAf32", "RPf32", budgets), ("H1b", "PCAsq8", "PCAf32", [b for b in budgets if b >= 32]),
                        ("H1c", "OPQ", "PCAsq8", [b for b in budgets if b <= 64])]:
        for B in Bs:
            r = cr(x, y, B)
            if r is not None: add(f"{h}[B={B}]", f"{x}-{y}", r["diff"], r.ci_lo, r.ci_hi, r.p_gt)
    if "raw" in recs:
        if a.focus_source in boot.parts and a.focus_source not in small:
            sh, _, _ = idshare_draws(recs["raw"], boot)
            for s in boot.parts:
                if s != a.focus_source:
                    d = sh[a.focus_source] - sh[s]
                    add(f"H3a[{a.focus_source}>{s}]", "id_share diff", d.mean(), *ci(d), p_gt(d), ex=(s in small))
            if "RPf32_B192" in recs:
                sh2, _, _ = idshare_draws(recs["RPf32_B192"], boot)
                for s in boot.parts:
                    d = sh2[s] - sh[s]; add(f"H3b[{s}]", "id_share(d=48)-raw", d.mean(), *ci(d), p_gt(d), ex=(s in small))
        if "RPf32_B192" in recs:
            xr = boot.draws(recs["raw"]["perq"]["xsrc"][:, 0]); xc = boot.draws(recs["RPf32_B192"]["perq"]["xsrc"][:, 0])
            for s in boot.parts:
                d = xc[s] - xr[s]; add(f"H4[{s}]", "xsrc_top1(RPf32 d=48)-raw", d.mean(), *ci(d), p_gt(d), ex=(s in small))
        pr = recs["raw"]["perq"]
        if "ts_bm25_k3" in pr:
            d = boot.draws(pr["ts_bm25_k3"][:, 0])["macro"] - boot.draws(pr["global"][:, 0])["macro"]
            add("H5", "twostage_bm25_k3-global macro", d.mean(), *ci(d), p_gt(d))
    df = pd.DataFrame(rows)
    if len(df) == 0: return df
    m = df.p_one_sided.notna() & ~df.excluded
    df["p_holm"] = np.nan; df.loc[m, "p_holm"] = holm(df.loc[m, "p_one_sided"].values)
    df["decision"] = np.where(df.p_holm < a.alpha, "supported", "not supported")
    df.loc[df.excluded, "decision"] = "excluded(few clusters)"; df["label"] = label
    if mres is not None:
        lo, hi = mres["dr2_lo"], mres["dr2_hi"]
        dec = "supported" if hi < a.delta_r2_max else ("not supported" if lo >= a.delta_r2_max else "inconclusive")
        df = pd.concat([df, pd.DataFrame([dict(id="H2b", test="upper CI of delta_R2 < thr", estimate=mres["delta_r2"],
              ci_lo=lo, ci_hi=hi, decision=dec, excluded=False, label=label)])], ignore_index=True)
    df["excluded"] = df["excluded"].fillna(False).astype(bool)
    return df

def universal(hy):
    rows = []
    for h in ["H1a", "H1b", "H1c", "H3a", "H3b", "H4"]:
        sel = hy.id.str.startswith(h + "["); s = hy[sel & ~hy.excluded]; nex = int((sel & hy.excluded).sum())
        if len(s) == 0:
            rows.append(dict(id=h, n_rows=0, n_pass=0, n_excluded=nex, decision="untested")); continue
        npass = int((s.decision == "supported").sum())
        rows.append(dict(id=h, n_rows=len(s), n_pass=npass, n_excluded=nex,
                         decision="supported" if npass == len(s) else "not supported"))
    for h in ["H5", "H2b"]:
        s = hy[hy.id == h]
        rows.append(dict(id=h, n_rows=len(s), n_pass=int((s.decision == "supported").sum()), n_excluded=0,
                         decision=s.decision.iloc[0] if len(s) else "untested"))
    return pd.DataFrame(rows)

def typed_xsrc_table(recs, boot):
    rows = []
    for n, r in recs.items():
        pq = r["perq"]
        if "xsrc" not in pq or "typed" not in pq: continue
        g = boot.draws(pq["global"][:, 0]); t = boot.draws(pq["typed"][:, 0]); x = boot.draws(pq["xsrc"][:, 0])
        for s in list(boot.parts) + ["macro"]:
            d = t[s] - g[s]
            rows.append(dict(rep=n, source=s, ndcg_global=g[s].mean(), ndcg_typed=t[s].mean(), typed_gain=d.mean(),
                             gain_lo=ci(d)[0], gain_hi=ci(d)[1], xsrc_top1=x[s].mean(), x_lo=ci(x[s])[0], x_hi=ci(x[s])[1]))
    return pd.DataFrame(rows)

# ------------------------------------------------------------------ main
def main():
    a = parse()
    if a.threads: faiss.omp_set_num_threads(a.threads)
    if a.make_emb: return make_emb(a)
    if a.stage == "freeze": return freeze(a)
    label = "EXPLORATORY"
    if a.stage == "confirm": load_prereg(a); label = "CONFIRMATORY"
    out = Path(a.out) / f"{a.stage}_{a.model}"; out.mkdir(parents=True, exist_ok=True)
    cdir = out / "cache"; cdir.mkdir(exist_ok=True)
    c = load_ctx(a); budgets = [int(x) for x in a.budgets.split(",")]
    boot = Boot(c, a.B, a.seed + 7, a.min_clusters); bld = Builder(c, a.seed, a.threads)
    small = {k: len(np.unique(c.qdoc[c.qsrc == i])) for i, k in enumerate(c.src_names)}
    print("[주의] 소스별 클러스터(계약서) 수:", small, "-> 20개 미만이면 CI 신뢰도 낮음")
    ext = set(a.ext_reps.split(",")); bm25_doc = build_bm25_doc(c) if ext else None
    recs = {}
    print("[A] 표현 평가")
    for n in cfg_names(budgets, c.D, a.skip_opq): recs[n] = eval_named(c, bld, n, ext, a, cdir, bm25_doc if n in ext else None)
    dh = recs["raw"]["perq"]["global"][:, MI["dochit"]].mean()
    chance = np.mean(1 - (1 - (c.doc_end - c.doc_start)[c.qdoc] / c.N) ** 10)
    assert dh > 3 * chance, f"정렬 가드 실패: raw dochit={dh:.3f}, chance={chance:.3f} -> 텍스트-벡터 대응 확인"
    recs["BM25"] = bm25_rec(c, None); recs["BM25_sw"] = bm25_rec(c, "english")
    ft = frontier_table(recs, boot); ft.to_csv(out / "frontier.csv", index=False)
    ctr = contrasts(recs, boot, budgets); ctr.to_csv(out / "matched_contrasts.csv", index=False)
    dec = decomposition(recs, boot); dec.to_csv(out / "decomposition.csv", index=False)
    bm = pd.DataFrame([dict(rep=k, source=s, **{m_: v for m_, v in zip(["ndcg", "dochit"], [
        boot.point(recs[k]["perq"]["global"][:, 0])[s], boot.point(recs[k]["perq"]["global"][:, 3])[s]])})
        for k in ("raw", "BM25", "BM25_sw") for s in list(boot.parts) + ["macro"]]); bm.to_csv(out / "bm25_vs_dense.csv", index=False)
    if ext:
        rows = []
        for n in ext & set(recs):
            for mode in recs[n]["perq"]:
                p = boot.point(recs[n]["perq"][mode][:, 0]); rows.append(dict(rep=n, mode=mode, **p))
        pd.DataFrame(rows).to_csv(out / "pools_and_twostage.csv", index=False)
    print("[B] 마진 모형 / ML")
    mg = compute_margins(c); qf = query_features(c); mres = None
    r = margin_analysis(c, recs, mg, a)
    if r:
        mres, bins = r; pd.DataFrame([mres]).to_csv(out / "margin_model.csv", index=False); bins.to_csv(out / "margin_bins.csv", index=False)
    def _pick(fam, tgt):
        cs = [(abs(np.log(recs[n]["B"] / tgt)), n) for n in recs if recs[n]["fam"] == fam and recs[n]["B"]]
        return min(cs)[1] if cs else None
    ml_reps = [x for x in (_pick("PCAf32", 384), _pick("RPf32", 384), _pick("PQ", 32)) if x]
    print("  [ML] 대상 구성:", ml_reps)
    ml, imp = fragility_ml(c, recs, mg, qf, ml_reps, a.seed)
    if ml.empty: ml = pd.DataFrame(columns=["rep", "model", "n", "pos_rate", "auc", "auc_sd", "brier"])
    if imp.empty: imp = pd.DataFrame(columns=["rep", "feature", "perm_importance_auc"])
    ml.to_csv(out / "fragility_ml.csv", index=False); imp.to_csv(out / "fragility_importance.csv", index=False)
    if not a.skip_ann:
        print("[C] ANN 층"); run_ann(c, bld, a, boot).to_csv(out / "ann_layer.csv", index=False)
    hy = test_hypotheses(recs, boot, ctr, mres, a, label); hy.to_csv(out / "hypotheses.csv", index=False)
    uv = universal(hy); uv.to_csv(out / "hypotheses_universal.csv", index=False)
    typed_xsrc_table(recs, boot).to_csv(out / "typed_vs_xsrc.csv", index=False)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for f, g in ft.groupby("fam"): ax.scatter(g.mem_mib, g.ndcg_macro, label=f, s=30 + 60 * g.pareto_prob)
        ax.set_xscale("log"); ax.set_xlabel("index memory (MiB)"); ax.set_ylabel("macro nDCG@10 (marker ~ Pareto prob.)")
        ax.legend(); fig.tight_layout(); fig.savefig(out / "frontier.png", dpi=200)
    except Exception as e: print("plot skip:", e)
    print(f"\n[{label}] 완료 -> {out}"); print(hy[["id", "estimate", "p_holm", "decision"]].to_string(index=False))
    print("\n[가설 단위 판정: 모든 비제외 행이 Holm 통과해야 supported]"); print(uv.to_string(index=False))

if __name__ == "__main__":
    main()