import os, sys, numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer
name = os.environ.get("MRL_MODEL", "Snowflake/snowflake-arctic-embed-m-v1.5")
qpre = os.environ.get("QUERY_PREFIX", "Represent this sentence for searching relevant passages: ")
ch = pd.read_parquet("data/chunks.parquet"); qr = pd.read_parquet("data/queries.parquet")
dev = "cuda" if torch.cuda.is_available() else "cpu"; print("model", name, "device", dev, flush=True)
m = SentenceTransformer(name, device=dev); m.max_seq_length = 256
if dev == "cuda": m.half()
out = "data/emb/M"; os.makedirs(out + "/shards", exist_ok=True)
texts = ch["text"].astype(str).tolist(); SH = 20000
for i in range(0, len(texts), SH):
    p = f"{out}/shards/c_{i:07d}.npy"
    if os.path.exists(p): continue
    E = m.encode(texts[i:i + SH], batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    np.save(p, E.astype("float32")); print("shard", i, flush=True)
C = np.concatenate([np.load(f"{out}/shards/c_{i:07d}.npy") for i in range(0, len(texts), SH)])
assert len(C) == len(texts), (len(C), len(texts))
np.save(f"{out}/chunks.npy", C)
Q = m.encode([qpre + t for t in qr.text.astype(str)], batch_size=256, normalize_embeddings=True, show_progress_bar=False)
np.save(f"{out}/queries.npy", Q.astype("float32")); print("done", C.shape, Q.shape, flush=True)
