import json, os, sys, numpy as np, pandas as pd
D = "data_lbrag"; H = "out_H"; O = "data"
ch = pd.read_parquet(f"{D}/chunks.parquet")
q  = pd.read_parquet(f"{D}/queries_clean.parquet")
X  = np.load(f"{H}/chunks_948e2085.npy",  mmap_mode="r")
Q  = np.load(f"{H}/queries_948e2085.npy", mmap_mode="r")
print("chunks", len(ch), X.shape, "| queries_clean", len(q), "keep", int(q.keep.sum()), "| Q", Q.shape)
assert len(ch) == X.shape[0], "chunks 행 수 불일치"
qk = q[q.keep].reset_index(drop=True)
assert len(qk) == Q.shape[0], f"keep 질의 {len(qk)} != 임베딩 {Q.shape[0]}: stage_h_lbrag.py 23~46행 확인 필요"

qr = json.load(open(f"{D}/qrels_clean.json"))
print("qrels type:", type(qr).__name__, "| size:", len(qr))
it = list(qr.items())[:3] if isinstance(qr, dict) else qr[:3]
print("qrels sample:", str(it)[:500])

def gold_of(qid):
    v = qr.get(qid)
    if v is None: return []
    if isinstance(v, dict): return sorted(int(k) for k, g in v.items() if float(g) > 0)
    if isinstance(v, list) and all(isinstance(x, (int, np.integer)) for x in v): return sorted(int(x) for x in v)
    raise ValueError(f"미지원 qrels 형식: {str(v)[:200]}")
try:
    gold = [gold_of(x) for x in qk.qid]
except Exception as e:
    sys.exit(f"[중단] {e}\n위 'qrels sample' 줄을 보내주세요.")

N = len(ch)
assert all(0 <= g < N for gs in gold for g in gs), "청크 행번호 범위 밖"
empty = sum(len(g) == 0 for g in gold)
cnt_ok = np.mean([len(g) == n for g, n in zip(gold, qk.n_rel_chunks)])
same_src = np.mean([all(ch.src.iloc[g].eq(s).all() for g in [gs]) for gs, s in zip(gold, qk.src) if gs])
print(f"gold 비어있음 {empty} | len(gold)==n_rel_chunks 비율 {cnt_ok:.3f} | 질의와 같은 src 비율 {same_src:.3f}")
assert same_src > 0.99, "gold 청크의 출처가 질의 출처와 다름 -> qrels 인덱스 해석 오류"

# 추가 정렬 검증: privacy_qa 질의에 따옴표로 들어간 회사명이 gold 문서명에 있는지
m = qk[qk.src == "privacy_qa"]; hit = tot = 0
for i, r in m.iterrows():
    if not gold[i]: continue
    s = r["query"].split('"')
    if len(s) >= 3:
        tot += 1; hit += s[1].lower().replace(" ", "") in ch.doc.iloc[gold[i][0]].lower().replace(" ", "")
print(f"privacy_qa 회사명-문서명 일치: {hit}/{tot}")

os.makedirs(f"{O}/emb/A", exist_ok=True)
pd.DataFrame({"source": ch.src.astype(str), "doc_id": ch.doc.astype(str), "text": ch.text.astype(str)}).to_parquet(f"{O}/chunks.parquet")
pd.DataFrame({"qid": qk.qid, "source": qk.src.astype(str), "text": qk["query"].astype(str), "gold_chunks": gold}).to_parquet(f"{O}/queries.parquet")
for n, s in (("chunks.npy", f"{H}/chunks_948e2085.npy"), ("queries.npy", f"{H}/queries_948e2085.npy")):
    p = f"{O}/emb/A/{n}"
    if os.path.lexists(p): os.remove(p)
    os.symlink(os.path.abspath(s), p)
print("OK ->", O)
