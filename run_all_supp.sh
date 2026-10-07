#!/bin/bash
cd ~/Research/IMDS_R2
export OMP_NUM_THREADS=8
S=out_full/supp2; mkdir -p $S
st() { echo "[$(date '+%F %T')] $1"; }
st "START load:$(uptime | awk -F'load average:' '{print $2}')"
for m in A B; do st "centring $m"; python supp_all.py centring --model $m > $S/log_centring_$m.txt 2>&1; done
for m in A B; do st "ladder $m"; python supp_all.py ladder --model $m > $S/log_ladder_$m.txt 2>&1
                 python supp_all.py agg --model $m > $S/agg_ladder_$m.txt 2>&1; done
[ -f data/emb/A/queries_instr.npy ] || { st "embed instr"; python supp_embed_q_instr.py > $S/log_embed.txt 2>&1; }
st "ladder A instr"; python supp_all.py ladder --model A --tag instr --seeds 0,1,2 --q_emb data/emb/A/queries_instr.npy > $S/log_ladder_A_instr.txt 2>&1
python supp_all.py agg --model A --tag instr > $S/agg_ladder_A_instr.txt 2>&1
for m in B A; do st "PQ/SQ8 $m"; python supp_all.py pq --model $m --seeds 0,1,2 >> $S/log_pq_$m.txt 2>&1; done
for m in B A; do st "OPQ $m"; python supp_all.py opq --model $m --seeds 0 --ms 32,64,128 >> $S/log_opq_$m.txt 2>&1; done
for m in B A; do python supp_all.py agg --model $m > $S/agg_final_$m.txt 2>&1; done
st "DONE"
