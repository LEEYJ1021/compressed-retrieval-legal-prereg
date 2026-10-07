import json, numpy as np, pandas as pd, faiss
SRC3 = ["contractnli", "cuad", "maud"]
DISC = 1.0 / np.log2(np.arange(2, 12))

def load(data_dir, model, q_emb=None):
    ch = pd.read_parquet(f"{data_dir}/chunks.parquet")
    qr = pd.read_parquet(f"{data_dir}/queries.parquet")
    X = np.load(f"{data_dir}/emb/{model}/chunks.npy").astype("float32")
    Q = np.load(q_emb or f"{data_dir}/emb/{model}/queries.npy").astype("float32")
    assert len(ch) == len(X) and len(qr) == len(Q)
    gold = []
    for g in qr.gold_chunks:
        if isinstance(g, str): g = json.loads(g)
        gold.append(np.asarray(list(g), dtype=np.int64))
    src = qr.source.str.lower().str.replace("_", "").values
    doc = ch.doc_id.values
    cont = np.array([str(doc[g[0]]) if len(g) else "none" for g in gold])
    keep = np.array([(s in SRC3) and len(g) > 0 for s, g in zip(src, gold)])
    gold = [g for g, k in zip(gold, keep) if k]
    print("queries kept", keep.sum(), "/", len(keep), flush=True)
    return X, Q[keep], gold, src[keep], cont[keep]

def search(index, Q, k=10):
    return index.search(np.ascontiguousarray(Q, dtype="float32"), k)[1]

def ndcg10(I, gold):
    out = np.zeros(len(I))
    for i, (row, g) in enumerate(zip(I, gold)):
        hit = np.isin(row, g)
        out[i] = (hit * DISC).sum() / DISC[:min(10, len(g))].sum()
    return out

def macro(v, src):
    return float(np.mean([v[src == s].mean() for s in SRC3]))

def boot_macro(v, src, cont, nb=2000, seed=20261004):
    rng = np.random.default_rng(seed); pre = {}
    for s in SRC3:
        m = src == s
        _, inv = np.unique(cont[m], return_inverse=True)
        pre[s] = (np.bincount(inv, weights=v[m]), np.bincount(inv))
    reps = np.empty(nb)
    for b in range(nb):
        t = 0.0
        for s in SRC3:
            sm, n = pre[s]; k = rng.integers(0, len(n), len(n)); t += sm[k].sum() / n[k].sum()
        reps[b] = t / 3
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return macro(v, src), float(lo), float(hi)

def fit_basis(X, n_fit, seed, centre=True):
    rng = np.random.default_rng(seed)
    fit = rng.choice(len(X), size=min(n_fit, len(X)), replace=False)
    Xs = X[fit].astype(np.float64)
    mu = Xs.mean(0) if centre else np.zeros(X.shape[1])
    Xc = Xs - mu
    lam, V = np.linalg.eigh(Xc.T @ Xc / len(Xc))
    o = np.argsort(lam)[::-1]
    return mu.astype("float32"), V[:, o].astype("float32"), fit

def nrm(A):
    A = np.asarray(A, dtype="float32")
    n = np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    return np.ascontiguousarray(A / n, dtype="float32")

def projN(A, mu, V, d):
    """Frozen-script PCA: centre, project, L2-normalise (chunks AND queries)."""
    return nrm((A - mu) @ V[:, :d])

def raw_baseline(X, Q, gold, src):
    idx = faiss.IndexFlatIP(X.shape[1]); idx.add(X)
    s = ndcg10(search(idx, Q), gold)
    return s, macro(s, src)
