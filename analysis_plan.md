# Analysis plan (frozen after exploration on model A, before confirmation on model B)
Date (UTC): 2026-10-04T09:06:36Z
Script: cfr_v2.py (sha256 recorded in prereg.json)
Confirmatory (model B): H1b (PCAsq8 > PCAf32, B>=32, all budgets), H4 (cross-source top-1 confusion rises under RPf32 d=48 vs raw, all non-excluded sources).
Decision rule: every non-excluded row Holm-significant; sources with <20 contracts (privacy_qa) excluded.
Descriptive only (not confirmatory): H1a, H1c (untested, OPQ skipped), H2b, H3a, H3b, H5, PQ results, BM25 comparison, fragility ML.
Run flags: --skip_opq
