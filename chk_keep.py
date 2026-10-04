import pandas as pd
a = pd.read_parquet("data_lbrag/_deprecated/queries_fixed.parquet").set_index("qid")
b = pd.read_parquet("data_lbrag/queries_clean.parquet").set_index("qid")
print("keep 플래그 불일치:", int((a.keep.reindex(b.index) != b.keep).sum()))
print("fixed의 keep=False:", a.index[~a.keep].tolist())
print("clean의 keep=False:", b.index[~b.keep].tolist())
