#!/usr/bin/env python3
"""LegalBench-RAG(복구 qrels)로 압축 차원 / 출처별 격차 / 인덱스 패밀리 / BM25를 평가.
nDCG@10은 청크 grade(2/1), dochit@10은 정답 계약서의 청크가 top10에 하나라도 있는지."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np, pandas as pd, faiss
from rp_benchmark_scenario_D import (project, c32, l2n, make_index, time_single, mem_mib,
                                     embed_texts, Lexical, boot_ci, paired, holm, log)

K = 10
DISC = 1.0 / np.log2(np.arange(2, K + 2))
ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data_lbrag"); ap.add_argument("--out", default="out_H")
ap.add_argument("--model", default="BAAI/bge-base-en-v1.5"); ap.add_argument("--device", default="cuda")
ap.add_argument("--dims", type=int, nargs="+", default=[384, 192, 96, 48])
ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--skip-index", action="store_true")
a = ap.parse_args()
D, OUT = Path(a.data), Path(a.out); OUT.mkdir(exist_ok=True)

# ---------- 데이터 ----------
cdf = pd.read_parquet(D / "chunks.parquet")
qdf = pd.read_parquet(D / "queries_clean.parquet")
qrels = json.load(open(D / "qrels_clean.json"))
qdf = qdf[qdf.keep].reset_index(drop=True)
SRC = ["privacy_qa", "contractnli", "maud", "cuad"]
csrc = cdf.src.map({s: i for i, s in enumerate(SRC)}).values
qsrc = qdf.src.map({s: i for i, s in enumerate(SRC)}).values
doc_id = cdf.doc.astype("category").cat.codes.values
nq, N = len(qdf), len(cdf)
rel = [{int(c): g for c, g in qrels[q].items()} for q in qdf.qid]
reldocs = [set(doc_id[list(r)]) for r in rel]
idcg = np.array([sum((2.0 ** g - 1) * DISC[i] for i, g in enumerate(sorted(r.values(), reverse=True)[:K])) for r in rel])
log(f"chunks={N} queries={nq} per-source q={dict(qdf.src.value_counts())}")

def evaluate(ret):
    nd, hit, dh = np.zeros(nq), np.zeros(nq), np.zeros(nq)
    for i in range(nq):
        g = np.array([rel[i].get(int(c), 0) for c in ret[i, :K]])
        nd[i] = ((2.0 ** g - 1) * DISC).sum() / idcg[i]
        hit[i] = (g > 0).any()
        dh[i] = len(set(doc_id[ret[i, :K]]) & reldocs[i]) > 0
    return nd, hit, dh

# ---------- 임베딩 ----------
tag = hashlib.md5(a.model.encode()).hexdigest()[:8]
X = embed_texts(cdf.text.tolist(), a.model, cache=OUT / f"chunks_{tag}.npy", device=a.device, batch=128)
Q = embed_texts(qdf["query"].tolist(), a.model, cache=OUT / f"queries_{tag}.npy", device=a.device, batch=128)
d0 = X.shape[1]
XQ = np.vstack([X, Q]); log(f"embeddings {X.shape} {Q.shape}")

def search(Zd, Zq, k=K, typed=False):
    if not typed:
        ix = faiss.IndexFlatL2(Zd.shape[1]); ix.add(c32(Zd)); return ix.search(c32(Zq), k)[1]
    ret = np.zeros((len(Zq), k), dtype=np.int64)
    for s in range(len(SRC)):
        gid, qi = np.where(csrc == s)[0], np.where(qsrc == s)[0]
        ix = faiss.IndexFlatL2(Zd.shape[1]); ix.add(c32(Zd[gid])); ret[qi] = gid[ix.search(c32(Zq[qi]), k)[1]]
    return ret

# ---------- H1/H2: 차원 x 출처 ----------
res = {}  # (mode,d) -> dict(nd,hit,dh) (시드 평균)
for mode in ("global", "typed_oracle"):
    for d in [d0] + a.dims:
        acc = []
        for s in range(1 if d == d0 else a.seeds):
            Z = XQ if d == d0 else project(XQ, "gauss", d, s)
            acc.append(evaluate(search(Z[:N], Z[N:], typed=(mode != "global"))))
        res[(mode, d)] = {k: np.mean([x[j] for x in acc], 0) for j, k in enumerate(["nd", "hit", "dh"])}
        res[(mode, d)]["seed_sd"] = float(np.std([x[0].mean() for x in acc])) if len(acc) > 1 else 0.0
        log(f"  {mode} d={d}: nDCG={res[(mode, d)]['nd'].mean():.4f} dochit={res[(mode, d)]['dh'].mean():.3f}")

rows = []
rng = np.random.default_rng(0)
for (mode, d), r in res.items():
    base = res[(mode, d0)]["nd"]
    for s, sn in enumerate(SRC + ["ALL"]):
        m = (qsrc == s) if sn != "ALL" else np.ones(nq, bool)
        mu, lo, hi = boot_ci(r["nd"][m], a.boot, 0)
        idx = rng.integers(0, m.sum(), (a.boot, m.sum()))
        ratio = r["nd"][m][idx].mean(1) / np.maximum(base[m][idx].mean(1), 1e-12)
        rows.append(dict(mode=mode, d=d, source=sn, n_q=int(m.sum()), ndcg=mu, ci_lo=lo, ci_hi=hi,
                         hit=r["hit"][m].mean(), dochit=r["dh"][m].mean(), seed_sd=r["seed_sd"] if sn == "ALL" else np.nan,
                         retention_pct=100 * np.mean(ratio), ret_lo=100 * np.percentile(ratio, 2.5), ret_hi=100 * np.percentile(ratio, 97.5)))
tab = pd.DataFrame(rows); tab.to_csv(OUT / "H1_dimension_by_source.csv", index=False)

# 격차 Δ, E (출처 4개, 같은 부트스트랩 표본을 모든 차원에 공유)
B = a.boot; rb = np.random.default_rng(1)
idx_s = [rb.integers(0, (qsrc == s).sum(), (B, (qsrc == s).sum())) for s in range(4)]
grow_rows = []
for mode in ("global", "typed_oracle"):
    dl, dE = {}, {}
    for d in [d0] + a.dims:
        mb = np.stack([res[(mode, d)]["nd"][qsrc == s][idx_s[s]].mean(1) for s in range(4)], 1)
        dl[d] = mb.max(1) - mb.min(1); dE[d] = 1 - dl[d] / np.maximum(mb.max(1), 1e-12)
    for d in [d0] + a.dims:
        g = dl[d] - dl[d0]
        p = float(min(1, 2 * min((g <= 0).mean(), (g >= 0).mean()))) if d != d0 else np.nan
        grow_rows.append(dict(mode=mode, d=d, delta=dl[d].mean(), d_lo=np.percentile(dl[d], 2.5), d_hi=np.percentile(dl[d], 97.5),
                              E=dE[d].mean(), E_lo=np.percentile(dE[d], 2.5), E_hi=np.percentile(dE[d], 97.5),
                              growth_vs_full=g.mean(), g_lo=np.percentile(g, 2.5), g_hi=np.percentile(g, 97.5), p_growth=p))
pd.DataFrame(grow_rows).to_csv(OUT / "H2_disparity.csv", index=False)

# 출처별 paired: 각 차원 vs 원차원 (Holm)
pr = []
for d in a.dims:
    for s, sn in enumerate(SRC):
        m = qsrc == s
        pr.append(dict(d=d, source=sn, **paired(res[("global", d)]["nd"][m], res[("global", d0)]["nd"][m], a.boot, 0)))
pdf = pd.DataFrame(pr); pdf["p_w_holm"] = holm(pdf.p_w.values); pdf.to_csv(OUT / "H2_paired_vs_full_by_source.csv", index=False)

# ---------- H3: 인덱스 패밀리 (같은 표현 기준) ----------
if not a.skip_index:
    fam = []
    for rname, Z in [("raw", XQ), (f"gauss{d0 // 4}", project(XQ, "gauss", d0 // 4, 0))]:
        Zd, Zq = c32(Z[:N]), c32(Z[N:])
        ex = search(Zd, Zq)
        nd_ex = evaluate(ex)[0]
        for kind in ["flat", "ivf", "hnsw", "ivfpq", "sq8"]:
            idx, bt = make_index(kind, Zd)
            for npb in ([None] if kind in ("flat", "hnsw") else [1, 4, 16, 64]):
                if npb: idx.nprobe = npb
                ms, p95 = time_single(idx, Zq, nq=100, k=K)
                I = idx.search(Zq, K)[1]
                nd = evaluate(I)[0]
                rec = np.mean([len(set(I[i]) & set(ex[i])) / K for i in range(nq)])
                pp = paired(nd, nd_ex, a.boot, 0)
                fam.append(dict(rep=rname, index=kind, nprobe=npb, recall_vs_exact=rec, ndcg=nd.mean(), vs_flat_same_rep=pp["diff"],
                                p_w=pp["p_w"], lat_ms=ms, lat_p95=p95, mem_MiB=mem_mib(idx), build_s=bt))
                log(f"  {rname}/{kind}/{npb}: recall={rec:.3f} ndcg={nd.mean():.4f} lat={ms:.2f}ms")
    pd.DataFrame(fam).to_csv(OUT / "H3_index_families_matched.csv", index=False)

# ---------- H4: BM25 (실제 qrels이므로 순환 아님) ----------
lex = Lexical(cdf.text.tolist())
ret = np.zeros((nq, K), dtype=np.int64)
qt = qdf["query"].tolist()
for s in range(0, nq, 64):
    Qm = (lex.cv.transform(qt[s:s + 64]) > 0).astype(np.float32).tocsr()
    S = (Qm @ lex.W.T).toarray()
    part = np.argpartition(-S, K, 1)[:, :K]
    ret[s:s + 64] = np.take_along_axis(part, np.argsort(-np.take_along_axis(S, part, 1), 1), 1)
ndb, hb, dhb = evaluate(ret)
dense = res[("global", d0)]
rows = []
for sn_i, sn in enumerate(SRC + ["ALL"]):
    m = (qsrc == sn_i) if sn != "ALL" else np.ones(nq, bool)
    pp = paired(ndb[m], dense["nd"][m], a.boot, 0)
    rows.append(dict(source=sn, bm25_ndcg=ndb[m].mean(), dense_ndcg=dense["nd"][m].mean(), bm25_minus_dense=pp["diff"],
                     ci_lo=pp["ci_lo"], ci_hi=pp["ci_hi"], dz=pp["dz"], p_w=pp["p_w"],
                     bm25_dochit=dhb[m].mean(), dense_dochit=dense["dh"][m].mean()))
pd.DataFrame(rows).to_csv(OUT / "H4_bm25_vs_dense.csv", index=False)
log("done ->", OUT.resolve())