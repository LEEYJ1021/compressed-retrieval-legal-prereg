import sys, faiss, numpy as np
from evalcore import *
model = sys.argv[1]
X, Q, gold, src, cont = load("data", model)
D = X.shape[1]; _, base = raw_baseline(X, Q, gold, src)
mu, V, fit = fit_basis(X, 100000, 0)
def run(Xp, Qp):
    idx = faiss.IndexFlatIP(Xp.shape[1]); idx.add(np.ascontiguousarray(Xp, dtype="float32"))
    return macro(ndcg10(search(idx, np.ascontiguousarray(Qp, dtype="float32")), gold), src) / base
for d in [48, 96, 192, 384, D]:
    a = run((X-mu)@V[:, :d], (Q-mu)@V[:, :d])
    b = run(projN(X, mu, V, d), projN(Q, mu, V, d))
    print(f"d={d:4d} centred, no norm {a:.3f} | centred + L2-norm (frozen) {b:.3f}", flush=True)
