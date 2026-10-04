"""patch_cfr.py : cfr.py 에 합의된 수정을 적용한다.
실행: cd ~/Research/IMDS_R2 && python patch_cfr.py   (cfr.py.bak 에 원본을 백업)
각 치환은 '정확히 1회 일치'가 아니면 중단한다.
"""
import re, shutil, sys
from pathlib import Path

P = Path("cfr.py")
src = P.read_text(encoding="utf-8")
shutil.copy(P, "cfr.py.bak")


def rep(s, old, new, label):
    n = s.count(old)
    if n != 1:
        sys.exit(f"[중단] {label}: 일치 {n}회 (1회여야 함). cfr.py 가 올린 버전과 다릅니다.")
    return s.replace(old, new)


def replace_def(s, name, new):
    m = re.search(rf"^def {name}\(.*?(?=^(?:def |class |# -{{10}}))", s, re.S | re.M)
    if not m:
        sys.exit(f"[중단] def {name} 를 찾지 못함")
    return s[:m.start()] + new.strip("\n") + "\n\n" + s[m.end():]


# ---------------------------------------------------------------- 1. HYP_SPEC
HYP_NEW = r'''HYP_SPEC = {
 "_decision_rule": "Universal claims (all budgets / all sources) are supported only if EVERY non-excluded row passes Holm-adjusted p<alpha. Sources with fewer than min_clusters contracts are excluded from the verdict and from macro averages (reported per source only).",
 "H1a": "At equal bytes/vector, PCA (float32, d=B/4) > random projection (float32, d=B/4) in macro nDCG@10, all budgets.",
 "H1b": "At equal bytes/vector (B>=32), PCA+SQ8 (d=B) > PCA float32 (d=B/4), all budgets.",
 "H1c": "At B<=64 bytes/vector, OPQ (m=B) > PCA+SQ8 (d=B).",
 "H2a": "DIAGNOSTIC, not a hypothesis: flip probability vs z = margin*sqrt(d); reported as model fit (R2, calibration bins).",
 "H2b": "Adding source indicators to the z-only model raises deviance-R2 by less than delta_r2_max: supported only if the upper limit of the contract-cluster bootstrap CI of delta_R2 is below delta_r2_max.",
 "H3a": "Identification share (oracle-global)/oracle at full precision is larger for the focus source than for every other non-excluded source.",
 "H3b": "RP d=48 raises the identification share relative to full precision in every non-excluded source.",
 "H4":  "Cross-source top-1 confusion rate of global dense search rises under compression (RPf32 d=48 vs raw) in every non-excluded source. typed_vs_xsrc.csv reports typed-pool gain next to the confusion rate descriptively (typed gain is a pool-restriction effect by construction).",
 "H5":  "Two-stage retrieval (doc-level BM25 top-3 -> dense chunk ranking) > global dense, macro nDCG. Scope: queries name the contract (template), so this is the effect when the query identifies the document.",
}
'''
m = re.search(r"^HYP_SPEC = \{.*?^\}\n", src, re.S | re.M)
if not m:
    sys.exit("[중단] HYP_SPEC 블록을 찾지 못함")
src = src[:m.start()] + HYP_NEW + src[m.end():]

# ---------------------------------------------------------------- 2. args / frozen
src = rep(src, r'''p.add_argument("--alpha", type=float, default=0.05)''',
          r'''p.add_argument("--alpha", type=float, default=0.05); p.add_argument("--min_clusters", type=int, default=20)''', "arg min_clusters")
src = rep(src, r'''"focus_source", "delta_r2_max", "alpha"]''', r'''"focus_source", "delta_r2_max", "alpha", "min_clusters"]''', "FROZEN")

# ---------------------------------------------------------------- 3. load_ctx: 중복 제거, chunk_src
src = rep(src, r'''    keep = np.array([len(g) > 0 for g in gold])
''', r'''    keep = np.array([len(g) > 0 for g in gold])
    _seen, _dup = set(), np.zeros(len(qu), bool)
    for i, (s_, t_, g_) in enumerate(zip(qu.source.astype(str), qu.text.astype(str), gold)):
        k_ = (s_, t_, tuple(sorted(int(x) for x in g_)))
        if k_ in _seen: _dup[i] = True
        _seen.add(k_)
    print(f"[data] 동일 질의문+동일 정답 중복 제거: {int((_dup & keep).sum())}건")
    keep &= ~_dup
''', "dedupe")
src = rep(src, r'''    csrc = ch.source.astype(str).map(sc).to_numpy()
''', r'''    csrc = ch.source.astype(str).map(sc).to_numpy()
    c.chunk_src = csrc
''', "chunk_src")

# ---------------------------------------------------------------- 4. run_modes: 교차 출처 top-1 혼동
src = rep(src, r'''    out = {n: np.zeros((c.nq, len(MET))) for n in names}; N = c.N
''', r'''    out = {n: np.zeros((c.nq, len(MET))) for n in names}; N = c.N
    if "global" in modes: out["xsrc"] = np.zeros((c.nq, 1))
''', "xsrc init")
src = rep(src, r'''            if "global" in modes: out["global"][qi] = q_metrics(rank_in(s, slice(0, N)), g, gd, c)
''', r'''            if "global" in modes:
                rg = rank_in(s, slice(0, N)); out["global"][qi] = q_metrics(rg, g, gd, c)
                out["xsrc"][qi, 0] = float(c.chunk_src[rg[0]] != sr)
''', "xsrc compute")

# ---------------------------------------------------------------- 5. randpool 제거, 캐시 가드
src = rep(src, r'''(["randpool", "ts"] if name in ext else [])''', r'''(["ts"] if name in ext else [])''', "randpool off")
src = rep(src, r'''if set(r["modes"]) == set(modes): return r''', r'''if set(r["modes"]) == set(modes) and "xsrc" in r["perq"]: return r''', "cache guard")

# ---------------------------------------------------------------- 6. Boot: 작은 출처는 macro 에서 제외
src = rep(src, r'''    def __init__(self, c, B, seed):''', r'''    def __init__(self, c, B, seed, min_clusters=0):''', "Boot sig")
src = rep(src, r'''            self.parts[nm] = (m, pos, len(docs), W)
''', r'''            self.parts[nm] = (m, pos, len(docs), W)
        self.core = [k for k, v in self.parts.items() if v[2] >= min_clusters] or list(self.parts)
''', "Boot core")
src = rep(src, r'''        r["macro"] = np.mean([r[k] for k in self.parts], axis=0); return r''',
          r'''        r["macro"] = np.mean([r[k] for k in self.core], axis=0); return r''', "Boot macro draws")
src = rep(src, r'''        r["macro"] = float(np.mean(list(r.values()))); return r''',
          r'''        r["macro"] = float(np.mean([r[k] for k in self.core])); return r''', "Boot macro point")
src = rep(src, r'''ps = [pa[k] - pb[k] for k in boot.parts]''', r'''ps = [pa[k] - pb[k] for k in boot.core]''', "loso core")

# ---------------------------------------------------------------- 7. margin_analysis (+ H2b 클러스터 부트스트랩 CI)
MARGIN_NEW = r'''
def margin_analysis(c, recs, mg, a):
    import statsmodels.api as sm
    base = recs["raw"]["perq"]["global"][:, MI["p1"]] == 1; rows = []
    for d in (384, 192, 96, 48):
        nm = f"RPf32_B{4 * d}"
        if nm not in recs: continue
        rows.append(pd.DataFrame(dict(flip=1 - recs[nm]["perq"]["global"][:, MI["p1"]], z=mg["margin"] * np.sqrt(d),
                  src=[c.src_names[i] for i in c.qsrc], doc=c.qdoc, d=d))[base])
    if not rows: return None
    L = pd.concat(rows, ignore_index=True); grp = pd.factorize(L["doc"])[0]
    X0 = sm.add_constant(L[["z"]]); fam = sm.families.Binomial()
    X1 = pd.concat([X0, pd.get_dummies(L["src"], prefix="src", drop_first=True).astype(float)], axis=1)
    m0 = sm.GLM(L["flip"], X0, family=fam).fit(cov_type="cluster", cov_kwds={"groups": grp})
    m1 = sm.GLM(L["flip"], X1, family=fam).fit(cov_type="cluster", cov_kwds={"groups": grp})
    r2 = lambda m: 1 - m.deviance / m.null_deviance
    names = list(m1.params.index); dum = [i for i, n in enumerate(names) if n.startswith("src_")]
    R = np.zeros((len(dum), len(names)))
    for k, i in enumerate(dum): R[k, i] = 1
    pw = float(m1.wald_test(R, scalar=True).pvalue) if dum else np.nan
    bz = float(m0.params["z"]); pz = float(m0.pvalues["z"]); p1s = pz / 2 if bz < 0 else 1 - pz / 2
    res = dict(n_rows=len(L), coef_z=bz, p_z_one_sided=p1s, r2_z_only=r2(m0), r2_with_source=r2(m1),
               delta_r2=r2(m1) - r2(m0), wald_source_p=pw)
    # H2b: 계약서(문서) 단위 클러스터 부트스트랩으로 delta_R2 의 CI
    rng = np.random.default_rng(a.seed + 11); ud = np.unique(grp)
    by = {g_: np.flatnonzero(grp == g_) for g_ in ud}
    A0, A1, yy = X0.to_numpy(float), X1.to_numpy(float), L["flip"].to_numpy(float); dr2 = []
    for _ in range(300):
        ii = np.concatenate([by[g_] for g_ in rng.choice(ud, len(ud), replace=True)])
        try:
            f0 = sm.GLM(yy[ii], A0[ii], family=fam).fit(); f1 = sm.GLM(yy[ii], A1[ii], family=fam).fit()
            dr2.append(r2(f1) - r2(f0))
        except Exception: pass
    res.update(dr2_lo=float(np.percentile(dr2, 2.5)) if dr2 else np.nan,
               dr2_hi=float(np.percentile(dr2, 97.5)) if dr2 else np.nan, dr2_nboot=len(dr2))
    L["zbin"] = pd.qcut(L.z, 10, duplicates="drop").astype(str)
    bins = L.groupby(["zbin", "src"]).flip.agg(["mean", "count"]).reset_index()
    return res, bins
'''
src = replace_def(src, "margin_analysis", MARGIN_NEW)

# ---------------------------------------------------------------- 8. test_hypotheses 전면 교체 + 보조 함수
HYP_FUNC = r'''
def test_hypotheses(recs, boot, ctr, mres, a, label):
    rows = []; budgets = [int(x) for x in a.budgets.split(",")]
    small = {k for k, v in boot.parts.items() if v[2] < a.min_clusters}
    if small: print(f"[판정 제외] 계약서 수 < {a.min_clusters}: {sorted(small)}")
    def add(h, desc, est, lo, hi, p, ex=False):
        rows.append(dict(id=h, test=desc, estimate=est, ci_lo=lo, ci_hi=hi, p_one_sided=p, excluded=ex))
    def cr(x, y, B):
        r = ctr[(ctr.a == x) & (ctr.b == y) & (ctr.B == B)] if len(ctr) else ctr
        return None if len(r) == 0 else r.iloc[0]
    for h, x, y, Bs in [("H1a", "PCAf32", "RPf32", budgets), ("H1b", "PCAsq8", "PCAf32", [b for b in budgets if b >= 32]),
                        ("H1c", "OPQ", "PCAsq8", [b for b in budgets if b <= 64])]:
        for B in Bs:
            r = cr(x, y, B)
            if r is not None: add(f"{h}[B={B}]", f"{x}-{y}", r["diff"], r.ci_lo, r.ci_hi, r.p_gt)
    if "raw" in recs:
        if a.focus_source in boot.parts and a.focus_source not in small:
            sh, _, _ = idshare_draws(recs["raw"], boot)
            for s in boot.parts:
                if s != a.focus_source:
                    d = sh[a.focus_source] - sh[s]
                    add(f"H3a[{a.focus_source}>{s}]", "id_share diff", d.mean(), *ci(d), p_gt(d), ex=(s in small))
            if "RPf32_B192" in recs:
                sh2, _, _ = idshare_draws(recs["RPf32_B192"], boot)
                for s in boot.parts:
                    d = sh2[s] - sh[s]; add(f"H3b[{s}]", "id_share(d=48)-raw", d.mean(), *ci(d), p_gt(d), ex=(s in small))
        if "RPf32_B192" in recs:
            xr = boot.draws(recs["raw"]["perq"]["xsrc"][:, 0]); xc = boot.draws(recs["RPf32_B192"]["perq"]["xsrc"][:, 0])
            for s in boot.parts:
                d = xc[s] - xr[s]; add(f"H4[{s}]", "xsrc_top1(RPf32 d=48)-raw", d.mean(), *ci(d), p_gt(d), ex=(s in small))
        pr = recs["raw"]["perq"]
        if "ts_bm25_k3" in pr:
            d = boot.draws(pr["ts_bm25_k3"][:, 0])["macro"] - boot.draws(pr["global"][:, 0])["macro"]
            add("H5", "twostage_bm25_k3-global macro", d.mean(), *ci(d), p_gt(d))
    df = pd.DataFrame(rows)
    if len(df) == 0: return df
    m = df.p_one_sided.notna() & ~df.excluded
    df["p_holm"] = np.nan; df.loc[m, "p_holm"] = holm(df.loc[m, "p_one_sided"].values)
    df["decision"] = np.where(df.p_holm < a.alpha, "supported", "not supported")
    df.loc[df.excluded, "decision"] = "excluded(few clusters)"; df["label"] = label
    if mres is not None:
        lo, hi = mres["dr2_lo"], mres["dr2_hi"]
        dec = "supported" if hi < a.delta_r2_max else ("not supported" if lo >= a.delta_r2_max else "inconclusive")
        df = pd.concat([df, pd.DataFrame([dict(id="H2b", test="upper CI of delta_R2 < thr", estimate=mres["delta_r2"],
              ci_lo=lo, ci_hi=hi, decision=dec, excluded=False, label=label)])], ignore_index=True)
    df["excluded"] = df["excluded"].fillna(False).astype(bool)
    return df

def universal(hy):
    rows = []
    for h in ["H1a", "H1b", "H1c", "H3a", "H3b", "H4"]:
        sel = hy.id.str.startswith(h + "["); s = hy[sel & ~hy.excluded]; nex = int((sel & hy.excluded).sum())
        if len(s) == 0:
            rows.append(dict(id=h, n_rows=0, n_pass=0, n_excluded=nex, decision="untested")); continue
        npass = int((s.decision == "supported").sum())
        rows.append(dict(id=h, n_rows=len(s), n_pass=npass, n_excluded=nex,
                         decision="supported" if npass == len(s) else "not supported"))
    for h in ["H5", "H2b"]:
        s = hy[hy.id == h]
        rows.append(dict(id=h, n_rows=len(s), n_pass=int((s.decision == "supported").sum()), n_excluded=0,
                         decision=s.decision.iloc[0] if len(s) else "untested"))
    return pd.DataFrame(rows)

def typed_xsrc_table(recs, boot):
    rows = []
    for n, r in recs.items():
        pq = r["perq"]
        if "xsrc" not in pq or "typed" not in pq: continue
        g = boot.draws(pq["global"][:, 0]); t = boot.draws(pq["typed"][:, 0]); x = boot.draws(pq["xsrc"][:, 0])
        for s in list(boot.parts) + ["macro"]:
            d = t[s] - g[s]
            rows.append(dict(rep=n, source=s, ndcg_global=g[s].mean(), ndcg_typed=t[s].mean(), typed_gain=d.mean(),
                             gain_lo=ci(d)[0], gain_hi=ci(d)[1], xsrc_top1=x[s].mean(), x_lo=ci(x[s])[0], x_hi=ci(x[s])[1]))
    return pd.DataFrame(rows)
'''
src = replace_def(src, "test_hypotheses", HYP_FUNC)

# ---------------------------------------------------------------- 9. main 수정
src = rep(src, r'''boot = Boot(c, a.B, a.seed + 7); bld''', r'''boot = Boot(c, a.B, a.seed + 7, a.min_clusters); bld''', "Boot call")
src = rep(src, r'''    ml, imp = fragility_ml(c, recs, mg, qf, ["PCAf32_B384", "RPf32_B384", "OPQ_B32"], a.seed)
''', r'''    def _pick(fam, tgt):
        cs = [(abs(np.log(recs[n]["B"] / tgt)), n) for n in recs if recs[n]["fam"] == fam and recs[n]["B"]]
        return min(cs)[1] if cs else None
    ml_reps = [x for x in (_pick("PCAf32", 384), _pick("RPf32", 384), _pick("PQ", 192)) if x]
    print("  [ML] 대상 구성:", ml_reps)
    ml, imp = fragility_ml(c, recs, mg, qf, ml_reps, a.seed)
    if ml.empty: ml = pd.DataFrame(columns=["rep", "model", "n", "pos_rate", "auc", "auc_sd", "brier"])
    if imp.empty: imp = pd.DataFrame(columns=["rep", "feature", "perm_importance_auc"])
''', "ML reps")
src = rep(src, r'''hy = test_hypotheses(recs, boot, ctr, mres, a, label); hy.to_csv(out / "hypotheses.csv", index=False)''',
          r'''hy = test_hypotheses(recs, boot, ctr, mres, a, label); hy.to_csv(out / "hypotheses.csv", index=False)
    uv = universal(hy); uv.to_csv(out / "hypotheses_universal.csv", index=False)
    typed_xsrc_table(recs, boot).to_csv(out / "typed_vs_xsrc.csv", index=False)''', "hy write")
src = rep(src, r'''print(hy[["id", "estimate", "p_holm", "decision"]].to_string(index=False))''',
          r'''print(hy[["id", "estimate", "p_holm", "decision"]].to_string(index=False))
    print("\n[가설 단위 판정: 모든 비제외 행이 Holm 통과해야 supported]"); print(uv.to_string(index=False))''', "print")

P.write_text(src, encoding="utf-8")
print("패치 완료. 백업: cfr.py.bak")
