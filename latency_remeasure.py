"""latency_remeasure.py: 공유 서버의 부하 변동에 대비해 라운드로빈 반복으로 구성별 단일 스레드 검색 지연을 재측정한다(사후 분석)."""
import argparse, hashlib, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="cfr_v2.py"); ap.add_argument("--hash_file", default="freeze_hashes.txt")
    ap.add_argument("--data_dir", default="data"); ap.add_argument("--model", default="B")
    ap.add_argument("--pairs", default="out_full/posthoc_exact_B/pq_vs_pcasq8_exact.csv")
    ap.add_argument("--out", default="out_full/posthoc_latency_B"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=7); ap.add_argument("--nq", type=int, default=200)
    ap.add_argument("--pq_train", type=int, default=20000); ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    frozen = {n: h for h, n in (ln.split() for ln in Path(a.hash_file).read_text().splitlines() if len(ln.split()) == 2)}
    name, cur = Path(a.script).name, sha(a.script); assert frozen.get(name) == cur, f"동결 해시와 불일치: {name}"
    sys.path.insert(0, str(Path(a.script).resolve().parent)); C = __import__(Path(a.script).stem); faiss = C.faiss
    ns = argparse.Namespace(data_dir=a.data_dir, model=a.model, max_q_per_source=0, seed=a.seed, rp_draws=1)
    c = C.load_ctx(ns); bld = C.Builder(c, a.seed, a.threads); IP = faiss.METRIC_INNER_PRODUCT
    pr = pd.read_csv(a.pairs); names = ["raw"] + [x for p in zip(pr.pq, pr.pca) for x in p]
    idx, qs = {}, {}
    for n in names:
        if n == "raw": ix = faiss.IndexFlatIP(c.D); ix.add(c.X); idx[n], qs[n] = ix, c.Q; continue
        fam, B = n.rsplit("_B", 1); B = int(B)
        if fam == "PQ":
            ix = faiss.IndexPQ(c.D, B, 8, IP); ix.train(np.ascontiguousarray(c.X[bld.tr[:a.pq_train]])); ix.add(c.X); q = c.Q
        else:
            fn = bld.proj("PCA", B, 0); Xp, q = C._nrm(fn(c.X)), C._nrm(fn(c.Q))
            ix = faiss.IndexScalarQuantizer(B, faiss.ScalarQuantizer.QT_8bit, IP); ix.train(np.ascontiguousarray(Xp[bld.tr])); ix.add(Xp)
        idx[n], qs[n] = ix, q
    print(f"[준비] 구성 {len(names)}개 | nproc={os.cpu_count()}")
    thr = bld.threads; rows = []
    for r in range(a.reps):
        la = os.getloadavg()[0]
        for n in names:
            med, p95 = C.time_index(idx[n], qs[n], thr, n=a.nq)
            rows.append(dict(rep=r, name=n, lat_med=med, lat_p95=p95, load1=la))
        print(f"  rep {r}: load1={la:.1f}", flush=True)
    d = pd.DataFrame(rows); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); d.to_csv(out / "latency_raw.csv", index=False)
    s = d.groupby("name").lat_med.agg(median="median", min="min", max="max").reindex(names).reset_index()
    s["mem_mib"] = [len(faiss.serialize_index(idx[n])) / 2 ** 20 for n in names]; s.to_csv(out / "latency_summary.csv", index=False)
    pd.set_option("display.width", 250); print(s.round(3).to_string(index=False))
    (out / "ADDENDUM.md").write_text(f"POST HOC\nDate (UTC): {pd.Timestamp.now(tz='UTC'):%Y-%m-%dT%H:%M:%SZ}\nFrozen script: {name} sha256 {cur}\n"
        f"Round-robin latency, {a.reps} reps x {a.nq} queries, single thread, shared server (load average logged). PQ trained on {a.pq_train} vectors (latency independent of codebook quality).\n", encoding="utf-8")

if __name__ == "__main__": main()
