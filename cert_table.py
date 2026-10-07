import glob, os, math, numpy as np, pandas as pd
from evalcore import SRC3, macro
S = "out_full/supp2"; OUT = "out_full/paper"; os.makedirs(OUT, exist_ok=True)
N = 160340; D = {"A": 768, "B": 1024}; NB = 4000
def pm(R, pat):
    fs = sorted(glob.glob(f"{R}/{pat}"))
    return (np.mean([np.load(f) for f in fs], 0), len(fs)) if fs else (None, 0)
rows = []
for model in "AB":
    R = f"{S}/{model}_std"; d0 = D[model]
    src = np.load(f"{R}/src.npy", allow_pickle=True); cont = np.load(f"{R}/cont.npy", allow_pickle=True)
    raw = np.load(f"{R}/raw.npy")
    cl = {}
    for s in SRC3:
        m = src == s; _, inv = np.unique(cont[m], return_inverse=True); cl[s] = (m, inv, int(inv.max()) + 1)
    def agg(v):
        return {s: (np.bincount(cl[s][1], weights=v[cl[s][0]], minlength=cl[s][2]),
                    np.bincount(cl[s][1], minlength=cl[s][2])) for s in SRC3}
    Ar = agg(raw); base = macro(raw, src); mx = 384 if model == "A" else 512
    C = [("raw", "raw", None, N * 4 * d0 / 2**20, False)]
    for B in [96, 192, 384, 768]:
        C.append(("sq8", f"sq8_B{B}", f"lad_s*_sq8_{B}.npy", (N * B + 4 * d0 * B + 4 * d0) / 2**20, False))
    for B in [96, 192, 384]:
        C.append(("sq4", f"sq4_B{B}", f"lad_s*_sq4_{B}.npy", (N * B + 4 * d0 * 2 * B + 4 * d0) / 2**20, False))
    for m in [32, 64, 128, 256, mx]:
        if d0 % m: continue
        dd = math.ceil(m + 256 * 4 * d0 / N)
        C.append(("pq", f"pq_m{m}", f"pq_{m}_s*.npy", (N * m + 256 * 4 * d0) / 2**20, False))
        if dd <= d0: C.append(("sq8", f"sq8m_m{m}", f"sqm_{m}_s*.npy", (N * dd + 4 * d0 * dd + 4 * d0) / 2**20, True))
        if 2 * dd <= d0: C.append(("sq4", f"sq4m_m{m}", f"sq4m_{m}_s*.npy", (N * dd + 4 * d0 * 2 * dd + 4 * d0) / 2**20, True))
    for kind, name, pat, mib, matched in C:
        if kind == "raw": v, n = raw, 0
        else:
            v, n = pm(R, pat)
            if v is None: continue
        Av = agg(v); rng = np.random.default_rng(20261004); Rb = np.empty(NB); Wb = np.empty(NB)
        for b in range(NB):
            nv, nr, rs = [], [], []
            for s in SRC3:
                sv, cnt = Av[s]; sr, _ = Ar[s]; K = len(cnt); k = rng.integers(0, K, K)
                c = cnt[k].sum(); a = sv[k].sum(); r = sr[k].sum()
                nv.append(a / c); nr.append(r / c); rs.append(a / r)
            Rb[b] = np.mean(nv) / np.mean(nr); Wb[b] = min(rs)
        sr_ = {s: float(v[src == s].mean() / raw[src == s].mean()) for s in SRC3}
        rows.append(dict(model=model, kind=kind, config=name, matched=matched, n_seeds=n, MiB_adj=round(mib, 1),
                         R=macro(v, src) / base, R_lo=np.percentile(Rb, 2.5), R_hi=np.percentile(Rb, 97.5),
                         W_point=min(sr_.values()), W_boot_mean=Wb.mean(), W_lo=np.percentile(Wb, 2.5),
                         contractnli=sr_["contractnli"], cuad=sr_["cuad"], maud=sr_["maud"]))
        print(model, name, flush=True)
df = pd.DataFrame(rows); df["certified"] = (df.R_lo >= 0.95) & (df.W_lo >= 0.90)
df.round(4).to_csv(f"{OUT}/certification.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(df.round(3).to_string(index=False))
nc = int(df[df.kind != "raw"].certified.sum())
print("\nCERTIFIED compressed configurations:", nc, "->", "Case II (some certified)" if nc else "Case I (none certified)")
print(df[(df.kind != "raw") & df.certified][["model", "config", "MiB_adj", "R_lo", "W_lo"]].round(3).to_string(index=False))
