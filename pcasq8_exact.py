"""pcasq8_exact.py: 각 PQ와 같은 메모리가 되도록 PCAsq8의 차원을 동적으로 정해 보간 없이 직접 비교한다(사후 분석)."""
import argparse, hashlib, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def cluster_boot(c, diff, nb, min_clusters, seed):
    rng = np.random.default_rng(seed); parts = []
    for s in range(len(c.src_names)):
        m = c.qsrc == s; docs, inv = np.unique(c.qdoc[m], return_inverse=True)
        if len(docs) < min_clusters: continue
        parts.append((np.bincount(inv, weights=diff[m]), np.bincount(inv).astype(float)))
    pt = float(np.mean([sd.sum() / cn.sum() for sd, cn in parts])); bs = np.empty(nb)
    for i in range(nb):
        v = []
        for sd, cn in parts:
            j = rng.integers(0, len(sd), len(sd)); v.append(sd[j].sum() / cn[j].sum())
        bs[i] = np.mean(v)
    return pt, np.percentile(bs, [2.5, 97.5]), len(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="cfr_v2.py"); ap.add_argument("--hash_file", default="freeze_hashes.txt")
    ap.add_argument("--data_dir", default="data"); ap.add_argument("--model", default="B")
    ap.add_argument("--caches", default="out_full/confirm_B/cache,out_full/posthoc_pq_B/cache")
    ap.add_argument("--out", default="out_full/posthoc_exact_B"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--B", type=int, default=4000); ap.add_argument("--min_clusters", type=int, default=20)
    ap.add_argument("--threads", type=int, default=0); ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    frozen = {n: h for h, n in (ln.split() for ln in Path(a.hash_file).read_text().splitlines() if len(ln.split()) == 2)}
    name, cur = Path(a.script).name, sha(a.script)
    assert frozen.get(name) == cur, f"동결 해시와 불일치: {name}"
    print(f"[동결 확인] {name} sha256 일치 {cur[:12]}")
    sys.path.insert(0, str(Path(a.script).resolve().parent)); C = __import__(Path(a.script).stem)
    if a.threads: C.faiss.omp_set_num_threads(a.threads)
    ns = argparse.Namespace(data_dir=a.data_dir, model=a.model, max_q_per_source=0, seed=a.seed, rp_draws=1)
    c = C.load_ctx(ns); bld = C.Builder(c, a.seed, a.threads)
    recs = {}
    for d in a.caches.split(","):
        for f in sorted(Path(d).glob("*.pkl")): recs[f.stem] = pickle.load(open(f, "rb"))
    pc = [(r["B"], r["mem"]) for r in recs.values() if r["fam"] == "PCAsq8"]
    slope, icpt = np.polyfit([p[0] for p in pc], [p[1] for p in pc], 1)
    pqs = sorted([r for r in recs.values() if r["fam"] == "PQ"], key=lambda r: r["mem"])
    match = {r["name"]: min(c.D, int(round((r["mem"] - icpt) / slope))) for r in pqs}
    print(f"[계획] PCAsq8 메모리 ≈ {icpt:.3f} + {slope:.5f}*B | PQ→PCAsq8 매칭 차원: {match}")
    out = Path(a.out); cdir = out / "cache"; cdir.mkdir(parents=True, exist_ok=True)
    for n in sorted(set(match.values())):
        k = f"PCAsq8_B{n}"
        if k not in recs: recs[k] = C.eval_named(c, bld, k, set(), ns, cdir, None)
    boot = C.Boot(c, a.B, a.seed + 7, a.min_clusters); rows = []
    for r in pqs:
        o = recs[f"PCAsq8_B{match[r['name']]}"]
        diff = r["perq"]["global"][:, 0] - o["perq"]["global"][:, 0]
        pt, ci, ns_ = cluster_boot(c, diff, a.boot, a.min_clusters, a.seed + 11)
        rows.append(dict(pq=r["name"], pq_mem=r["mem"], pq_lat=r["lat"], pq_ndcg=boot.point(r["perq"]["global"][:, 0])["macro"],
                         pca=f"PCAsq8_B{match[r['name']]}", pca_mem=o["mem"], pca_lat=o["lat"],
                         pca_ndcg=boot.point(o["perq"]["global"][:, 0])["macro"], diff=pt, ci_lo=ci[0], ci_hi=ci[1],
                         lat_ratio=r["lat"] / o["lat"], n_sources=ns_))
    t = pd.DataFrame(rows); t.to_csv(out / "pq_vs_pcasq8_exact.csv", index=False)
    pd.set_option("display.width", 250); print(t.round(4).to_string(index=False))
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    (out / "ADDENDUM.md").write_text(f"POST HOC (not confirmatory)\nDate (UTC): {stamp}\nFrozen script: {name} sha256 {cur}\n"
        f"PCAsq8 dimension chosen per PQ to match serialized memory (linear fit of memory on B); CI = contract-cluster bootstrap "
        f"({a.boot} resamples, sources with < {a.min_clusters} contracts excluded), independent post hoc implementation.\n", encoding="utf-8")

if __name__ == "__main__": main()
