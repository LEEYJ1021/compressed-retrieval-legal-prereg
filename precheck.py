import os, re, json, glob, shutil, subprocess, importlib
import numpy as np, pandas as pd
P = lambda *a: print(*a, flush=True)
def hdr(t): P("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)

hdr("1. queries.parquet")
qr = pd.read_parquet("data/queries.parquet")
P("shape", qr.shape, "| columns", list(qr.columns))
P(qr.dtypes.to_string())
for c in qr.columns:
    if qr[c].dtype == object and c not in ("text", "gold_chunks"):
        P(f"[{c}] n_unique={qr[c].nunique()} top:", qr[c].value_counts().head(5).to_dict())
L = qr.text.astype(str).str.len(); P("query char length: min/median/max", L.min(), int(L.median()), L.max())
P("\n--- 25 sample queries (5 per source if available) ---")
sc = "source" if "source" in qr.columns else None
parts = [g.sample(min(len(g), 5), random_state=0) for _, g in qr.groupby(sc)] if sc else [qr.sample(25, random_state=0)]
for _, r in pd.concat(parts).iterrows():
    P(f"[{r[sc] if sc else ''}] {str(r.text)[:260]!r}")
P("\n--- separator statistics ---")
for sep in [";", ":", " - ", " | ", "?", "\n", "\""]:
    P(f"contains {sep!r}: {qr.text.astype(str).str.contains(re.escape(sep)).mean():.3f}")
P("first-8-token prefix repetition (top 5):", qr.text.astype(str).str.split().str[:3].str.join(" ").value_counts().head(5).to_dict())

hdr("2. gold_chunks format and doc_id linkage")
ch = pd.read_parquet("data/chunks.parquet")
P("chunks shape", ch.shape, "| columns", list(ch.columns))
g0 = qr.gold_chunks.iloc[0]; P("gold_chunks[0] type", type(g0).__name__, "value", str(g0)[:120])
def lst(g):
    if isinstance(g, str): g = json.loads(g)
    return list(g)
ng = qr.gold_chunks.map(lambda g: len(lst(g))); P("gold per query: mean/min/max", ng.mean().round(2), ng.min(), ng.max())
P("\n--- 12 sample doc_id values (4 per source) ---")
doc = ch.doc_id.values
for s, g in qr.groupby(sc) if sc else [("all", qr)]:
    for i in g.index[:4]:
        gl = lst(qr.gold_chunks[i])
        if gl: P(f"[{s}] doc_id={str(doc[gl[0]])!r}")
P("n distinct doc_id:", ch.doc_id.nunique())
if sc:
    P("\n--- query vs doc_id token overlap (does the query contain the doc name?) ---")
    def toks(d):
        s = os.path.basename(str(d))
        for _ in range(3): s = re.sub(r"\.(txt|pdf|json|md|html)$", "", s, flags=re.I)
        return [t.lower() for t in re.findall(r"[A-Za-z0-9]+", s)]
    res = {}
    for i in qr.sample(min(1500, len(qr)), random_state=0).index:
        gl = lst(qr.gold_chunks[i])
        if not gl: continue
        tk = toks(doc[gl[0]]); tq = set(re.findall(r"[A-Za-z0-9]+", str(qr.text[i]).lower()))
        if tk: res.setdefault(qr[sc][i], []).append(np.mean([t in tq for t in tk]))
    for s, v in res.items(): P(f"{s}: mean fraction of doc_id tokens found in query = {np.mean(v):.3f} (n={len(v)})")

hdr("3. chunk text column (needed by the Matryoshka embedding step)")
cands = [c for c in ["text", "chunk_text", "chunk", "content"] if c in ch.columns]
P("text-like columns found:", cands)
if cands:
    t = ch[cands[0]].astype(str); w = t.str.split().str.len()
    P(f"{cands[0]}: chars median {int(t.str.len().median())}, max {t.str.len().max()}; words median {int(w.median())}, p95 {int(w.quantile(.95))}")
    P("sample chunk:", repr(t.iloc[len(t) // 2][:300]))
else: P("NO text column: MRL embedding step will abort. columns =", list(ch.columns))

hdr("4. embedding files and the encoders that made them")
for m in ["A", "B", "M"]:
    for f in ["chunks.npy", "queries.npy"]:
        p = f"data/emb/{m}/{f}"
        if os.path.exists(p):
            a = np.load(p, mmap_mode="r"); P(f"{p}: shape {a.shape} dtype {a.dtype} norm[0]={np.linalg.norm(a[0]):.4f}")
        else: P(f"{p}: MISSING")
hits = []
for f in glob.glob("*.py") + glob.glob("*.sh") + glob.glob("*.md") + glob.glob("*.json") + glob.glob("*.yaml"):
    try: s = open(f, errors="ignore").read()
    except Exception: continue
    for mm in re.finditer(r"[\w.\-]+/[\w.\-]+", s):
        w = mm.group(0)
        if re.search(r"(bge|e5|gte|minilm|mpnet|nomic|arctic|stella|jina|instructor|embed|legal|sentence)", w, re.I) and not w.endswith((".py", ".npy", ".csv", ".txt")):
            hits.append((f, w))
seen = set()
P("model-name-like strings in project files (look for Model B's encoder):")
for f, w in hits:
    if (f, w) not in seen: seen.add((f, w)); P(f"  {f}: {w}")
for f in glob.glob("*.py"):
    s = open(f, errors="ignore").read()
    for mm in re.finditer(r"(query_prefix|passage_prefix|prefix|instruction)[^\n]{0,120}", s, re.I):
        P(f"  prefix hint {f}: {mm.group(0)[:140]}")
for f in glob.glob("README*") + glob.glob("*.md"):
    s = open(f, errors="ignore").read()
    for mm in re.finditer(r"[^\n]*(Model B|encoder|query prefix|e5|gte)[^\n]*", s):
        P(f"  {f}: {mm.group(0)[:200]}")

hdr("5. machine: GPU, CPU, RAM, disk, libraries")
try: import torch; P("torch", torch.__version__, "cuda available:", torch.cuda.is_available(),
                     torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
except Exception as e: P("torch import failed:", e)
P("cpu count", os.cpu_count())
try: P(open("/proc/meminfo").read().split("\n")[0], "|", open("/proc/meminfo").read().split("\n")[2])
except Exception: pass
du = shutil.disk_usage("."); P(f"disk free here: {du.free / 2**30:.1f} GiB")
P("loadavg", os.getloadavg())
for lib in ["faiss", "sentence_transformers", "pyarrow", "matplotlib"]:
    try: mod = importlib.import_module(lib); P(lib, getattr(mod, "__version__", "ok"))
    except Exception as e: P(lib, "MISSING", e)
try: P(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip() or "nvidia-smi: no output")
except Exception as e: P("nvidia-smi unavailable")

hdr("6. HuggingFace reachability and candidate Matryoshka models (config only, no weights)")
cand = ["Snowflake/snowflake-arctic-embed-m-v1.5", "nomic-ai/nomic-embed-text-v1.5", "mixedbread-ai/mxbai-embed-large-v1",
        "Alibaba-NLP/gte-large-en-v1.5", "jinaai/jina-embeddings-v3"]
try:
    from huggingface_hub import HfApi
    api = HfApi()
    for c in cand:
        try:
            i = api.model_info(c); tags = [t for t in (i.tags or []) if "matryoshka" in t.lower() or "mrl" in t.lower()]
            P(f"OK   {c} | downloads {getattr(i, 'downloads', '?')} | matryoshka tags: {tags}")
        except Exception as e: P(f"FAIL {c}: {type(e).__name__}")
    P("search 'matryoshka' (top 8 by downloads):")
    for mm in api.list_models(search="matryoshka", sort="downloads", limit=8): P("  ", mm.id)
except Exception as e: P("huggingface_hub problem:", e)

hdr("7. previous outputs the new analyses depend on")
for m in ["A", "B"]:
    d = f"out_full/supp2/{m}_std"
    P(m, "pq .npy:", len(glob.glob(f"{d}/pq_*_s*.npy")), "| sqm:", len(glob.glob(f"{d}/sqm_*_s*.npy")),
      "| ladder .npy:", len(glob.glob(f"{d}/lad_*.npy")), "| raw:", os.path.exists(f"{d}/raw.npy"))
for f in ["evalcore.py", "supp_all.py", "supp_extra.py", "embed_mrl.py", "noname.py", "run_extra.sh"]:
    P(f"{f}: {'present' if os.path.exists(f) else 'MISSING'}")
P("\nPRECHECK DONE")
