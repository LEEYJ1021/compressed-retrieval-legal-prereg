import sys, glob, numpy as np, pandas as pd
from evalcore import macro, boot_macro
model = sys.argv[1]; out = f"out_full/supp/pq/{model}"
src, cont, raw = np.load(f"{out}/src.npy"), np.load(f"{out}/cont.npy"), np.load(f"{out}/raw.npy")
base = macro(raw, src); D = {"A": 768, "B": 1024}[model]; N = 160340
def get(kind, m): return [np.load(f) for f in sorted(glob.glob(f"{out}/{kind}_{m}_s*.npy"))]
rows = []
for m in [16, 32, 64, 128, 256]:
    P, S, O = get("pq", m), get("sq8", m), get("opq", m)
    if not P or not S: continue
    pm, sm = np.mean(P, 0), np.mean(S, 0)
    r = dict(model=model, m=m, n_seeds=len(P), MiB_PQ=round((N*m + 256*4*D)/2**20, 2),
             PQ_ret=macro(pm, src)/base, SQ8_ret=macro(sm, src)/base,
             PQ_seedstd=np.std([macro(v, src) for v in P]), SQ8_seedstd=np.std([macro(v, src) for v in S]))
    e, lo, hi = boot_macro(pm - sm, src, cont, nb=2000); r["PQ-SQ8"] = f"{e:+.4f} [{lo:+.4f},{hi:+.4f}]"
    if O:
        om = np.mean(O, 0); r["OPQ_ret"] = macro(om, src)/base
        e, lo, hi = boot_macro(om - pm, src, cont, nb=2000); r["OPQ-PQ"] = f"{e:+.4f} [{lo:+.4f},{hi:+.4f}]"
    rows.append(r)
df = pd.DataFrame(rows); pd.set_option("display.width", 250)
print(df.round(4).to_string(index=False)); df.to_csv(f"out_full/supp/pq_summary_{model}.csv", index=False)
