import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
qr = pd.read_parquet("data/queries.parquet")
m = SentenceTransformer("BAAI/bge-base-en-v1.5")
E = m.encode(["Represent this sentence for searching relevant passages: " + t for t in qr.text],
             batch_size=64, normalize_embeddings=True, show_progress_bar=False)
np.save("data/emb/A/queries_instr.npy", E.astype("float32")); print(E.shape)
