# POST HOC (not confirmatory): per-source contract-cluster bootstrap contrasts; frozen script cfr_v2.py untouched
import pickle, glob, os, sys, numpy as np, pandas as pd
B = 2000; S = ["contractnli", "cuad", "maud"]; rng = np.random.default_rng(20261004)
q = pd.read_parquet("data/queries.parquet"); ch = pd.read_parquet("data/chunks.parquet"); doc = ch.doc_id.to_numpy()
q["gold"] = q.gold_chunks.map(lambda g: tuple(sorted(int(x) for x in g)))
q = q.drop_duplicates(["text", "gold"]).reset_index(drop=True)
print("nq", len(q), "(기대 6877)")
qdoc = doc[[g[0] for g in q.gold]]; src = q.source.to_numpy(); parts = {}
for s in S:
    m = np.flatnonzero(src == s); d, pos = np.unique(qdoc[m], return_inverse=True)
    W = rng.multinomial(len(d), np.full(len(d), 1 / len(d)), size=B).astype(float)
    parts[s] = (m, pos, len(d), W); print(s, "계약서", len(d), "질의", len(m))
def drw(v):
    o = {}
    for s, (m, pos, nd, W) in parts.items():
        sd = np.bincount(pos, weights=v[m], minlength=nd); nn = np.bincount(pos, minlength=nd)
        o[s] = (W @ sd) / np.maximum(W @ nn, 1e-12)
    o["macro"] = np.mean([o[s] for s in S], axis=0); return o
ci = lambda x: np.percentile(x, [2.5, 97.5])
fmt = lambda x: "%+.4f[%+.4f,%+.4f]%s" % (x.mean(), *ci(x), "*" if (ci(x)[0] > 0 or ci(x)[1] < 0) else " ")
key = lambda x: (x.split("_B")[0], int(x.split("_B")[1]) if "_B" in x else 0)
for M, dn in (("A", "out_full/explore_A"), ("B", "out_full/confirm_B")):
    L = lambda n: pickle.load(open(f"{dn}/cache/{n}.pkl", "rb"))["perq"]["global"][:, 0]
    fr = pd.read_csv(f"{dn}/frontier.csv").set_index("name"); raw = L("raw")
    if len(raw) != len(q) or not all(abs(raw[src == s].mean() - fr.loc["raw", "ndcg_" + s]) < 1e-4 for s in S):
        sys.exit(f"[중단] 모델 {M}: perq와 질의 정렬이 맞지 않음")
    names = sorted({os.path.basename(p)[:-4] for p in glob.glob(f"{dn}/cache/*.pkl")
                    if os.path.basename(p).split("_B")[0] in ("raw", "PQ", "PCAsq8") or "PCAf32_B1536" in p}, key=key)
    D = {n: drw(L(n)) for n in names}
    def show(a, b):
        if a in D and b in D:
            print(f"{a:12s}-{b:12s} | " + " | ".join(fmt(D[a][k] - D[b][k]) for k in S + ["macro"]))
    print(f"\n=== 모델 {M} (정렬 확인 통과) 차이 a-b, 열={S}+macro; *=CI가 0 제외(다중비교 보정 없음) ===")
    for n in names:
        if n.startswith("PQ_B"): show(n, n.replace("PQ", "PCAsq8"))
    show("PCAsq8_B384", "PCAf32_B1536"); show("PCAsq8_B384", "raw"); show("PCAsq8_B768", "raw")
    print("\n보존율(raw 대비) 출처별 [95% CI]와 최악 출처 보존율")
    for n in names:
        if n.split("_B")[0] not in ("PQ", "PCAsq8"): continue
        R = {s: D[n][s] / D["raw"][s] for s in S}; w = np.min([R[s] for s in S], axis=0)
        print(f"{n:12s} " + " ".join(f"{s[:4]} {R[s].mean():.3f}[{ci(R[s])[0]:.3f},{ci(R[s])[1]:.3f}]" for s in S)
              + f" | worst {w.mean():.3f}[{ci(w)[0]:.3f},{ci(w)[1]:.3f}]")
