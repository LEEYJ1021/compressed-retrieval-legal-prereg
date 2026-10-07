import sys, faiss, numpy as np
from evalcore import *
model = sys.argv[1]
X, Q, gold, src, cont = load("data", model)
D = X.shape[1]
_, base = raw_baseline(X, Q, gold, src)
mu, V, fit = fit_basis(X, 100000, 0)
def run(Xp, Qp):
    idx = faiss.IndexFlatIP(Xp.shape[1]); idx.add(np.ascontiguousarray(Xp, dtype="float32"))
    return macro(ndcg10(search(idx, np.ascontiguousarray(Qp, dtype="float32")), gold), src) / base
for d in [48, 96, 192, 384, D]:
    both = run((X-mu)@V[:, :d], (Q-mu)@V[:, :d])
    dbonly = run((X-mu)@V[:, :d], Q@V[:, :d])
    nocen = run(X@V[:, :d], Q@V[:, :d])
    print(f"d={d:4d} both-centred {both:.3f} | db-only {dbonly:.3f} | no centring {nocen:.3f}", flush=True)
