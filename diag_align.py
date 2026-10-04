import re, sqlite3, faiss, numpy as np
from scipy.stats import spearmanr
from sklearn.feature_extraction.text import TfidfVectorizer

TN = ["shipping_order","warehouse_policy","customer_requirement"]
con = sqlite3.connect("logistics_knowledge.db")
rows = con.execute("select rowid, doc_id, doc_type, content from documents").fetchall()

def nat(s):  # 자연 정렬 키 (SO-2 < SO-10)
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]

def signals(texts, V, rng, n=400):
    T = TfidfVectorizer(sublinear_tf=True).fit_transform(texts)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    S = rng.choice(len(texts), min(n, len(texts)), replace=False)
    ES = V[S] @ V.T; ES[np.arange(len(S)), S] = -9
    TS = (T[S] @ T.T).toarray(); TS[np.arange(len(S)), S] = -9
    nn = ES.argmax(1)
    f1 = TS[np.arange(len(S)), nn].mean()                              # NN의 tfidf 코사인
    k = 10
    eo = np.argpartition(-ES, k, 1)[:, :k]; to = np.argpartition(-TS, k, 1)[:, :k]
    f2 = np.mean([len(set(a) & set(b)) / k for a, b in zip(eo, to)])   # top-10 겹침
    m = np.ones_like(ES, bool); m[np.arange(len(S)), S] = False
    f3 = spearmanr(ES[m][:200000], TS[m][:200000])[0]                  # 쌍별 상관
    return np.array([f1, f2, f3])

rng = np.random.default_rng(0)
for tn in TN:
    idx = faiss.read_index(f"faiss_logistics.index.{tn}")
    V = idx.reconstruct_n(0, idx.ntotal)
    low = lambda r: tn.split("_")[0][:4] in str(r[2]).lower()
    sub = [r for r in rows if low(r)]
    print(f"\n=== {tn}: docs={len(sub)} vecs={len(V)} ===")
    print("벡터 통계: mean|x|=%.4f, 평균 쌍 코사인=%.4f, 중복벡터 수=%d" % (
        np.abs(V).mean(),
        float(((V/np.linalg.norm(V,axis=1,keepdims=True))[:300] @ (V/np.linalg.norm(V,axis=1,keepdims=True))[:300].T).mean()),
        len(V) - len({v.tobytes() for v in V})))
    cands = {
        "rowid": sorted(sub, key=lambda r: r[0]),
        "docid_str": sorted(sub, key=lambda r: str(r[1])),
        "docid_nat": sorted(sub, key=lambda r: nat(r[1])),
        "rowid_rev": sorted(sub, key=lambda r: -r[0]),
    }
    # null: rowid 순서에서 텍스트를 무작위 셔플
    base = [r[3] for r in cands["rowid"]]
    null = np.array([signals([base[i] for i in rng.permutation(len(base))], V, rng) for _ in range(15)])
    mu, sd = null.mean(0), null.std(0) + 1e-9
    print("null  mean:", np.round(mu, 4), " sd:", np.round(sd, 4))
    for name, c in cands.items():
        if len(c) != len(V): continue
        s = signals([r[3] for r in c], V, rng)
        print(f"{name:10s} signals={np.round(s,4)}  z={np.round((s-mu)/sd,1)}")
