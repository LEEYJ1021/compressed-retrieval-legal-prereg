#!/usr/bin/env python3
import re, os, math, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT, OUT = "out_full/supp2", "out_full/paper"; os.makedirs(OUT, exist_ok=True)
MODELS = ["A", "B"]; BUD = [16, 32, 64, 96, 192, 384, 768]
FMTS = ["f32", "fp16", "sq8", "sq4"]
LAB = {"f32": "float32", "fp16": "fp16", "sq8": "8-bit", "sq4": "4-bit"}
COL = {"f32": "#444444", "fp16": "#0072B2", "sq8": "#D55E00", "sq4": "#009E73"}
MK = {"f32": "o", "fp16": "s", "sq8": "^", "sq4": "D"}
LS = {"f32": "-", "fp16": "--", "sq8": "-.", "sq4": ":"}
MNAME = {"A": "Model A", "B": "Model B"}
DIMS = {"A": 768, "B": 1024}; N_CH = 160340

rd = lambda m, f, tag="std": pd.read_csv(f"{ROOT}/{m}_{tag}/{f}")
def pci(s):
    x = re.findall(r"[-+]?\d+\.\d+", s); return float(x[0]), float(x[1]), float(x[2])
def sg(x, p=4):                       # signed number in LaTeX
    return f"{x:+.{p}f}".replace("-", "$-$")
def cell(d, lo, hi, p=4):             # "diff [lo, hi]" ; dagger if CI contains 0
    t = f"{sg(d,p)} [{sg(lo,p)}, {sg(hi,p)}]"
    return t + ("$^\\dagger$" if lo <= 0 <= hi else "")
def tex(path, colspec, header, body, caption, label, notes):
    L = ["\\begin{table}[t]", "\\centering\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in body:
        L.append("\\midrule" if isinstance(r, str) and r == "MID" else (r if isinstance(r, str) else " & ".join(r) + " \\\\"))
    L += ["\\bottomrule", "\\end{tabular}", f"\\par\\smallskip\\footnotesize {notes}", "\\end{table}"]
    open(path, "w").write("\n".join(L)); print("wrote", path)

# ---------------- Table 11 : ladder retention ----------------
body, num = [], []
for m in MODELS:
    s = rd(m, "ladder_summary.csv")
    body += ["MID" if body else "MID", f"\\multicolumn{{5}}{{l}}{{\\textit{{{MNAME[m]}}}}} \\\\"]
    for B in BUD:
        vals = {}
        for f in FMTS:
            q = s[(s.B == B) & (s.fmt == f)]
            if len(q): vals[f] = (q.ret_mean.iloc[0], q.ret_std.iloc[0])
        best = max(v[0] for v in vals.values())
        row = [str(B)]
        for f in FMTS:
            if f not in vals: row.append("--"); continue
            mu, sd = vals[f]; t = f"{mu:.3f} ({sd:.3f})"
            row.append(f"\\textbf{{{t}}}" if mu == best else t)
            num.append(dict(model=m, B=B, fmt=f, ret_mean=mu, ret_std=sd))
        body.append(row)
pd.DataFrame(num).to_csv(f"{OUT}/table11_values.csv", index=False)
tex(f"{OUT}/table11.tex", "lcccc", ["B (bytes/vec)", "float32", "fp16", "8-bit", "4-bit"], body[1:],
    "Retention of exact-search macro nDCG@10 at equal bytes per vector for four PCA code formats.",
    "tab:ladder",
    "Mean (SD) over five PCA-fit seeds; retention = macro nDCG@10 / exact-search macro nDCG@10 "
    "(Model A 0.0684, Model B 0.0897). Dimensions are B/4, B/2, B and 2B. Bold: best format per row. "
    "4-bit is undefined at B=768 (2B exceeds the embedding dimension). Post hoc [P]; seeds vary the PCA fit sample only.")

# ---------------- Table 12 : ladder contrasts ----------------
CN = [("sq8-f32", "8-bit $-$ float32"), ("fp16-f32", "fp16 $-$ float32"), ("sq4-sq8", "4-bit $-$ 8-bit")]
body = []
for m in MODELS:
    c = rd(m, "ladder_contrasts.csv")
    body += ["MID", f"\\multicolumn{{4}}{{l}}{{\\textit{{{MNAME[m]}}}}} \\\\"]
    for B in BUD:
        row = [str(B)]
        for k, _ in CN:
            q = c[(c.B == B) & (c.contrast == k)]
            row.append(cell(q["diff"].iloc[0], q.lo.iloc[0], q.hi.iloc[0]) if len(q) else "--")
        body.append(row)
tex(f"{OUT}/table12.tex", "lccc", ["B"] + [n for _, n in CN], body[1:],
    "Paired differences in macro nDCG@10 between code formats at equal bytes.", "tab:ladder_contrasts",
    "Difference of seed-averaged per-query scores with 95\\% contract-cluster bootstrap interval (2,000 resamples). "
    "$^\\dagger$ interval contains zero. Intervals exclude seed variability. Post hoc [P].")

# ---------------- Table 13 : centring ----------------
D_LIST = [24, 48, 96, 192, 384]
RET = [("pca_c_ret", "PCA, centred (frozen)"), ("pca_u_ret", "PCA, uncentred"),
       ("rp_u_ret", "RP, uncentred (frozen)"), ("rp_c_ret", "RP, centred")]
CON = [("FROZEN pca_c-rp_u", "Frozen contrast: PCA$_c$ $-$ RP$_u$"), ("projector|centred", "Projector effect, both centred"),
       ("projector|uncentred", "Projector effect, both uncentred"), ("centring|PCA", "Centring effect, PCA"),
       ("centring|RP", "Centring effect, RP")]
body = []
for m in MODELS:
    c = rd(m, "centring.csv").set_index("d")
    body += ["MID", f"\\multicolumn{{6}}{{l}}{{\\textit{{{MNAME[m]}}}}} \\\\"]
    for k, n in RET: body.append([n] + [f"{c.loc[d, k]:.3f}" for d in D_LIST])
    body.append("MID")
    for k, n in CON:
        body.append([n] + [cell(*pci(c.loc[d, k]), p=4) for d in D_LIST])
    fr, rc = [pci(c.loc[d, "FROZEN pca_c-rp_u"]) for d in D_LIST], [pci(c.loc[d, "centring|RP"]) for d in D_LIST]
    body.append(["RP-centring share of frozen gap"] +
                [f"{100*b[0]/a[0]:.0f}\\%" if (a[1] > 0 or a[2] < 0) else "--" for a, b in zip(fr, rc)])
tex(f"{OUT}/table13.tex", "lccccc", ["d (float32 dims)"] + [str(d) for d in D_LIST], body[1:],
    "Projector $\\times$ centring cross, with all conditions L2-normalised as in the frozen pipeline.",
    "tab:centring",
    "Upper rows: retention. Lower rows: differences in macro nDCG@10 with 95\\% contract-cluster bootstrap intervals "
    "(1,000 resamples). $^\\dagger$ interval contains zero. Centring subtracts the PCA-fit mean from chunks and queries. "
    "Share shown only where the frozen contrast excludes zero. One PCA fit (seed 0) and three RP draws; draw variability "
    "is not in the intervals. Post hoc [P].")

# ---------------- Table 14 : PQ / OPQ ----------------
body = []
for m in MODELS:
    p = rd(m, "pq_summary.csv"); D = DIMS[m]
    body += ["MID", f"\\multicolumn{{8}}{{l}}{{\\textit{{{MNAME[m]}}}}} \\\\"]
    for _, r in p.iterrows():
        mm = int(r.m); d = math.ceil(mm + 256 * 4 * D / N_CH)
        pq = pci(r["PQ-SQ8"]); row = [str(mm), f"{r.MiB_PQ:.1f}", str(d), f"{r.PQ_ret:.3f}", f"{r.SQ8_ret:.3f}", cell(*pq)]
        if "OPQ_ret" in p.columns and not pd.isna(r.OPQ_ret):
            row += [f"{r.OPQ_ret:.3f}", cell(*pci(r["OPQ-PQ"]))]
        else: row += ["--", "--"]
        body.append(row)
tex(f"{OUT}/table14.tex", "cccccccc",
    ["PQ $m$", "MiB", "SQ8 $d$", "PQ", "PCA+SQ8", "PQ $-$ SQ8", "OPQ", "OPQ $-$ PQ"], body[1:],
    "Product quantisation versus PCA+8-bit scalar quantisation at matched index size, and OPQ versus PQ.",
    "tab:pq",
    "Retention columns are means over seeds; differences are macro nDCG@10 with 95\\% contract-cluster bootstrap intervals. "
    "PQ and PCA+SQ8: three seeds (seed also changes the 50,000-chunk training sample and k-means seed). OPQ: one seed, "
    "20,000 training chunks, 10 iterations; its rotation matrix (4D$^2$ bytes) is not counted in MiB. "
    "PCA projection-matrix memory is not charged to SQ8. $^\\dagger$ interval contains zero. Post hoc [P].")

# ---------------- figures ----------------
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 7.5, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
def xaxis(ax, label):
    ax.set_xscale("log", base=2); ax.set_xticks(BUD); ax.set_xticklabels([str(b) for b in BUD])
    ax.minorticks_off(); ax.set_xlabel(label); ax.grid(alpha=.25, lw=.5)

# Fig 3c
fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.0), sharey=True)
for ax, m in zip(axs, MODELS):
    s = rd(m, "ladder_summary.csv")
    for f in FMTS:
        q = s[s.fmt == f].sort_values("B")
        ax.errorbar(q.B, q.ret_mean, yerr=q.ret_std, color=COL[f], marker=MK[f], ls=LS[f], lw=1.3, ms=4,
                    capsize=2, label=LAB[f])
    ax.axhline(1.0, color="gray", lw=.7, ls=":")
    xaxis(ax, "Bytes per vector (B)"); ax.set_ylim(-0.02, 1.08); ax.set_title(MNAME[m], fontsize=9)
axs[0].set_ylabel("Retention of exact-search macro nDCG@10"); axs[0].legend(title="Code format", frameon=False, loc="upper left")
fig.tight_layout(); [fig.savefig(f"{OUT}/fig3c.{e}", dpi=600) for e in ("pdf", "png")]; plt.close(fig)

# Fig 3d : rows = x in bytes / x in richer-representation dimensions
CF = {"sq8-f32": 1.0, "fp16-f32": 0.5, "sq4-sq8": 2.0}          # richer-side dims = B * factor
CC = {"sq8-f32": COL["sq8"], "fp16-f32": COL["fp16"], "sq4-sq8": COL["sq4"]}
CL = {"sq8-f32": "8-bit $-$ float32", "fp16-f32": "fp16 $-$ float32", "sq4-sq8": "4-bit $-$ 8-bit"}
fig, axs = plt.subplots(2, 2, figsize=(7.4, 5.6), sharey="row")
for j, m in enumerate(MODELS):
    c = rd(m, "ladder_contrasts.csv")
    for i in range(2):
        ax = axs[i, j]
        for k in CF:
            q = c[c.contrast == k].sort_values("B"); x = q.B * (CF[k] if i == 1 else 1)
            ax.plot(x, q["diff"], color=CC[k], marker=MK[{"sq8-f32":"sq8","fp16-f32":"fp16","sq4-sq8":"sq4"}[k]],
                    lw=1.3, ms=4, label=CL[k]); ax.fill_between(x, q.lo, q.hi, color=CC[k], alpha=.18, lw=0)
            ib = q["diff"].values.argmax(); ax.plot(x.values[ib], q["diff"].values[ib], marker="*", ms=11,
                                                    color=CC[k], mec="k", mew=.5, ls="none", zorder=5)
        ax.axhline(0, color="k", lw=.7)
        ax.set_xscale("log", base=2); ax.minorticks_off(); ax.grid(alpha=.25, lw=.5)
        ticks = BUD if i == 0 else [8, 16, 32, 64, 96, 192, 384, 768, 1536]
        ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks], fontsize=7)
        ax.set_xlabel("Bytes per vector (B)" if i == 0 else "Dimensions of the richer representation")
        if i == 0: ax.set_title(MNAME[m], fontsize=9)
axs[0, 0].set_ylabel("Gain in macro nDCG@10"); axs[1, 0].set_ylabel("Gain in macro nDCG@10")
axs[0, 0].legend(frameon=False, loc="upper right")
fig.tight_layout(); [fig.savefig(f"{OUT}/fig3d.{e}", dpi=600) for e in ("pdf", "png")]; plt.close(fig)
print("done")
