import os, gc, numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
qr = pd.read_parquet("data/queries.parquet"); T = qr.text.astype(str).tolist()
new = []
for t in T:
    k = t.find(";"); q = t[k + 1:].strip() if k > 0 else t
    new.append(q if len(q) > 10 else t)
print("changed fraction", round(float(np.mean([a != b for a, b in zip(T, new)])), 3))
print("distinct questions:", len(set(new)), "of", len(new))
for i in np.random.default_rng(0).choice(len(new), 8, replace=False): print("---\nBEFORE:", T[i][:160], "\nAFTER :", new[i][:160])
pd.DataFrame({"text": new}).to_parquet("data/queries_noname.parquet")
INSTR = "Represent this sentence for searching relevant passages: "
def try_model(model, name, prefixes):
    ref = np.load(f"data/emb/{model}/queries.npy")[:300]
    try: m = SentenceTransformer(name, device="cuda")
    except Exception as e: print(f"[{model}] load fail {name}: {type(e).__name__}", flush=True); return None
    if m.get_sentence_embedding_dimension() != ref.shape[1]:
        print(f"[{model}] {name}: dim mismatch, skip", flush=True); del m; gc.collect(); torch.cuda.empty_cache(); return None
    for pre in prefixes:
        chk = m.encode([pre + t for t in T[:300]], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        cos = float((chk * ref).sum(1).mean()); print(f"[{model}] {name!r} prefix={pre!r}: cosine {cos:.4f}", flush=True)
        if cos > 0.99:
            E = m.encode([pre + t for t in new], batch_size=64, normalize_embeddings=True, show_progress_bar=False)
            np.save(f"data/emb/{model}/queries_noname.npy", E.astype("float32")); print(f"[{model}] SAVED with {name!r} prefix={pre!r}", flush=True)
            open(f"data/emb/{model}/encoder_detected.txt", "w").write(f"{name}\t{pre}\n"); return name
    del m; gc.collect(); torch.cuda.empty_cache(); return None
try_model("A", "BAAI/bge-base-en-v1.5", [""])
nb = os.environ.get("MODEL_B_NAME")
cands = [nb] if nb else ["BAAI/bge-large-en-v1.5", "intfloat/e5-large-v2", "intfloat/e5-large", "thenlper/gte-large",
        "mixedbread-ai/mxbai-embed-large-v1", "WhereIsAI/UAE-Large-V1", "Snowflake/snowflake-arctic-embed-l", "BAAI/bge-m3", "intfloat/multilingual-e5-large"]
pres = [os.environ["MODEL_B_PREFIX"]] if "MODEL_B_PREFIX" in os.environ else ["", INSTR, "query: "]
for c in cands:
    if try_model("B", c, pres): break
else: print("Model B encoder NOT identified: B no-name analysis skipped")
