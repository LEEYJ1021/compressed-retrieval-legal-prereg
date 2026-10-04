"""verify_qrels.py : 4개 출처의 qrels를 원본 span과 대조한다.
실행: cd ~/Research/IMDS_R2 && source ~/venvs/rpbench/bin/activate && python verify_qrels.py
"""
import json, os, re, sys
import numpy as np, pandas as pd

BASE = os.path.expanduser("~/Research/IMDS_R2/data_probe/awinml__legalbench-rag")
DATA = "data_lbrag"
SOURCES = ["cuad", "contractnli", "maud", "privacy_qa"]


def dkey(p):
    """'maud/Foo.txt' -> 'maud/foo' (마지막 두 경로 요소, 소문자, 확장자 제거)"""
    parts = p.replace("\\", "/").split("/")[-2:]
    return os.path.splitext("/".join(parts))[0].lower()


ch = pd.read_parquet(f"{DATA}/chunks.parquet").reset_index(drop=True)
q = pd.read_parquet(f"{DATA}/queries.parquet").set_index("qid")
Q = {"qrels(원본)": json.load(open(f"{DATA}/qrels.json"))}
if os.path.exists(f"{DATA}/qrels_fixed.json"):
    Q["qrels_fixed"] = json.load(open(f"{DATA}/qrels_fixed.json"))

# 원본 txt 파일 위치
files = {}
for root, _, fs in os.walk(BASE):
    for f in fs:
        if f.endswith(".txt"):
            files[dkey(os.path.join(root, f))] = os.path.join(root, f)
print("원본 txt 파일:", len(files))

_c = {}
def ftext(k):
    if k not in _c:
        _c[k] = open(files[k], encoding="utf-8", errors="ignore", newline="").read()
    return _c[k]

ch["dk"] = ch.doc.map(dkey)
print("청크 문서 중 원본 파일 없음:", int((~ch.dk.isin(files)).groupby(ch.src).sum().sum()))

# ---- [1] 청크 좌표: 저장된 start/end 검사, 어긋나면 텍스트로 복원 ----
S = ch.start.to_numpy().astype(np.int64).copy()
E = ch.end.to_numpy().astype(np.int64).copy()
ok_stored = np.zeros(len(ch), bool)
restored = np.zeros(len(ch), bool)
failed = np.zeros(len(ch), bool)
for d, g in ch.groupby("dk", sort=False):
    if d not in files:
        failed[g.index] = True
        continue
    raw = ftext(d); cur = 0
    for i, t, s, e in zip(g.index, g.text, g.start, g.end):
        if raw[int(s):int(e)] == t:
            ok_stored[i] = True
            cur = int(s) + 1
            continue
        p = raw.find(t, cur)
        if p < 0:
            m = re.compile(r"\s+".join(map(re.escape, t.split()))).search(raw, cur) if t.split() else None
            if m: S[i], E[i] = m.start(), m.end(); restored[i] = True; cur = m.start() + 1
            else: failed[i] = True
        else:
            S[i], E[i] = p, p + len(t); restored[i] = True; cur = p + 1
rep = pd.DataFrame({"src": ch.src, "저장좌표일치": ok_stored, "복원": restored, "실패": failed}).groupby("src").mean().round(4)
print("\n[1] 청크 좌표 검증 (비율)\n", rep.to_string())
if failed.mean() > 0.05:
    print("!! 좌표 복원 실패 5% 초과: 아래 결과를 조심해서 읽을 것")

# ---- [2] 원본 benchmark 로드 ----
tests = {}
for s in SOURCES:
    p = f"{BASE}/benchmarks/{s}.json"
    if not os.path.exists(p):
        print(f"!! {p} 없음"); continue
    by_q = {}
    for t in json.load(open(p))["tests"]:
        by_q.setdefault(t["query"].strip(), []).append(t)
    tests[s] = by_q
    print(f"{s}: 원본 tests {sum(len(v) for v in by_q.values())}")


def gold_of(t):
    return [(dkey(sn["file_path"]), tuple(sn["span"])) for sn in t["snippets"]]


def score(ids, gold):
    ids = [int(c) for c in ids if not failed[int(c)]]
    if not ids or not gold: return np.nan, np.nan
    def ov(c, gd):
        return ch.dk.iat[c] == gd[0] and S[c] < gd[1][1] and gd[1][0] < E[c]
    rec = np.mean([any(ov(c, g) for c in ids) for g in gold])
    prec = np.mean([any(ov(c, g) for g in gold) for c in ids])
    return rec, prec


# ---- [3] 출처별, qrels 버전별 원본 대조 ----
out = []
for name, qr in Q.items():
    for s in SOURCES:
        if s not in tests: continue
        keys = [k for k in q.index[q.src == s]]
        rows, nomatch, noqrel, ambiguous = [], 0, 0, 0
        for k in keys:
            ids = list(qr.get(k, {}))
            if not ids:
                noqrel += 1; continue
            docs = {ch.dk.iat[int(c)] for c in ids}
            cands = [t for t in tests[s].get(q.loc[k, "query"].strip(), [])
                     if t["snippets"] and docs & {g[0] for g in gold_of(t)}]
            if not cands:
                nomatch += 1; continue
            res = [score(ids, gold_of(t)) for t in cands]
            if len(cands) > 1: ambiguous += 1
            best = max(res, key=lambda x: (np.nan_to_num(x[0], nan=-1), np.nan_to_num(x[1], nan=-1)))
            rows.append(best)
        r = np.array(rows, float)
        out.append(dict(qrels=name, src=s, 질의=len(keys), 정답없음=noqrel, 원본매칭실패=nomatch,
                        모호=ambiguous, 평가=len(r),
                        rec=np.nanmean(r[:, 0]) if len(r) else np.nan,
                        prec=np.nanmean(r[:, 1]) if len(r) else np.nan,
                        rec_lt1=float((r[:, 0] < 0.999).mean()) if len(r) else np.nan,
                        prec_lt1=float((r[:, 1] < 0.999).mean()) if len(r) else np.nan))
res = pd.DataFrame(out).round(3)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print("\n[3] 원본 span 대조 (rec=정답 구간 포획률, prec=청크 정밀도, *_lt1=1 미만 질의 비율)")
print(res.to_string(index=False))
res.to_csv("verify_qrels_result.csv", index=False)
print("\n저장: verify_qrels_result.csv")
print("판정: rec, prec가 1.0에 가깝고 *_lt1이 작으면 'provenance 검증됨'.")
print("      prec가 1 미만이어도 청크 크기가 span보다 크면 정상일 수 있음(rec를 우선 볼 것).")
