"""pq_supplement.py: 동결된 cfr_v2.py를 수정하지 않고 import해, 동결 실행에서 빠진 PQ 구성을 사후(post hoc) 분석으로 보충한다."""
import argparse, hashlib, pickle, sys
from pathlib import Path
import numpy as np, pandas as pd

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", default="cfr_v2.py"); ap.add_argument("--hash_file", default="freeze_hashes.txt")
    ap.add_argument("--data_dir", default="data"); ap.add_argument("--model", default="B")
    ap.add_argument("--ref_dir", default="out_full/confirm_B"); ap.add_argument("--out", default="out_full/posthoc_pq_B")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--B", type=int, default=4000)
    ap.add_argument("--min_clusters", type=int, default=20); ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--min_m", type=int, default=16)
    a = ap.parse_args()
    frozen = {n: h for h, n in (ln.split() for ln in Path(a.hash_file).read_text().splitlines() if len(ln.split()) == 2)}
    name, cur = Path(a.script).name, sha(a.script)
    assert frozen.get(name) == cur, f"동결 해시와 불일치: {name}"
    print(f"[동결 확인] {name} sha256 일치 {cur[:12]}")
    sys.path.insert(0, str(Path(a.script).resolve().parent)); C = __import__(Path(a.script).stem)
    if a.threads: C.faiss.omp_set_num_threads(a.threads)
    ns = argparse.Namespace(data_dir=a.data_dir, model=a.model, max_q_per_source=0, seed=a.seed, rp_draws=1)
    c = C.load_ctx(ns); boot = C.Boot(c, a.B, a.seed + 7, a.min_clusters); bld = C.Builder(c, a.seed, a.threads)

    ref = {f.stem: pickle.load(open(f, "rb")) for f in sorted((Path(a.ref_dir) / "cache").glob("*.pkl"))}
    have = {r["B"] for r in ref.values() if r["fam"] == "PQ"}
    ms = [m for m in range(a.min_m, c.D) if c.D % m == 0 and m not in have]
    print(f"[계획] D={c.D} | 보충할 PQ m={ms} | 기존 PQ={sorted(have)}")
    out = Path(a.out); cdir = out / "cache"; cdir.mkdir(parents=True, exist_ok=True)
    new = {f"PQ_B{m}": C.eval_named(c, bld, f"PQ_B{m}", set(), ns, cdir, None) for m in ms}

    rows = []
    for n, r in {**ref, **new}.items():
        if r["fam"] not in ("raw", "PQ", "PCAsq8", "RPsq8", "PCAf32", "RPf32"): continue
        row = dict(name=n, fam=r["fam"], B=r["B"], mem_mib=r["mem"], lat_ms=r["lat"], new=n in new)
        row.update({f"ndcg_{k}": v for k, v in boot.point(r["perq"]["global"][:, 0]).items()})
        rows.append(row)
    t = pd.DataFrame(rows).sort_values("mem_mib").reset_index(drop=True)
    anc = t[t.fam == "PCAsq8"]
    t["pcasq8_same_mem"] = np.interp(np.log(t.mem_mib), np.log(anc.mem_mib), anc.ndcg_macro, left=np.nan, right=np.nan)
    t["diff_vs_pcasq8"] = t.ndcg_macro - t.pcasq8_same_mem
    t.to_csv(out / "pq_vs_pcasq8.csv", index=False)
    print(t[t.fam == "PQ"][["name", "mem_mib", "lat_ms", "ndcg_macro", "pcasq8_same_mem", "diff_vs_pcasq8", "new"]].round(4).to_string(index=False))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for f, g in t.groupby("fam"): ax.plot(g.mem_mib, g.ndcg_macro, "o-" if f in ("PQ", "PCAsq8") else "o:", label=f)
        ax.set_xscale("log"); ax.set_xlabel("index memory (MiB, serialized)"); ax.set_ylabel("macro nDCG@10"); ax.legend()
        fig.tight_layout(); fig.savefig(out / "pq_frontier.png", dpi=200)
    except Exception as e: print("plot skip:", e)
    stamp = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    (out / "ADDENDUM.md").write_text(f"POST HOC (not confirmatory)\nDate (UTC): {stamp}\nFrozen script: {name} sha256 {cur}\n"
        f"Added PQ m={ms} on model {a.model}; reference records from {a.ref_dir}/cache\n"
        f"PQ trained on first 50,000 of the training sample (Builder.make); codebook included in mem_mib.\n", encoding="utf-8")

if __name__ == "__main__": main()
