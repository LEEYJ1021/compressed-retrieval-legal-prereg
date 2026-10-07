import faiss, numpy as np
from evalcore import *
X, Q, gold, src, cont = load("data", "M")
_, base = raw_baseline(X, Q, gold, src)
mu, V, fit = fit_basis(X, 100000, 0)
_, V0, _ = fit_basis(X, 100000, 0, centre=False)
D = X.shape[1]; z = np.zeros(D, dtype="float32")
def run(Xp, Qp):
    idx = faiss.IndexFlatIP(Xp.shape[1]); idx.add(Xp)
    return macro(ndcg10(search(idx, Qp), gold), src) / base
for d in [96, 192, 384, D]:
    print(d, "PCA centred+norm", round(run(projN(X, mu, V, d), projN(Q, mu, V, d)), 3),
          "| PCA uncentred+norm", round(run(projN(X, z, V0, d), projN(Q, z, V0, d)), 3),
          "| trunc+norm", round(run(nrm(X[:, :d]), nrm(Q[:, :d])), 3), flush=True)
