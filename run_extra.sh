#!/bin/bash
cd ~/Research/IMDS_R2
source ~/venvs/rpbench/bin/activate 2>/dev/null
export OMP_NUM_THREADS=8
S=out_full/supp2; mkdir -p $S
st() { echo "[$(date '+%F %T')] $1"; }
X() { python supp_extra.py "$@"; }
st "START"
st "noname queries + Model B encoder detection"; python noname.py > $S/log_noname.txt 2>&1
st "MRL embed (GPU)"
if python embed_mrl.py > $S/log_embed_mrl.txt 2>&1; then MRL_OK=1; else MRL_OK=0; st "MRL embed FAILED (see log_embed_mrl.txt)"; fi
for m in A B; do st "sq4m $m"; X sq4m --model $m > $S/log_sq4m_$m.txt 2>&1; done
for m in A B; do st "recall $m"; X recall --model $m > $S/log_recall_$m.txt 2>&1; done
for m in A B; do st "binary $m"; X binary --model $m > $S/log_binary_$m.txt 2>&1; done
for m in A B; do
  if [ -f data/emb/$m/queries_noname.npy ]; then
    st "within-doc noname $m"; python withindoc.py --model $m --q_emb data/emb/$m/queries_noname.npy > $S/log_withindoc_$m.txt 2>&1
    st "global ladder noname $m"
    python supp_all.py ladder --model $m --tag noname --seeds 0,1,2 --q_emb data/emb/$m/queries_noname.npy > $S/log_ladder_${m}_noname.txt 2>&1
    python supp_all.py agg --model $m --tag noname > $S/agg_ladder_${m}_noname.txt 2>&1
  fi
done
if [ "$MRL_OK" = "1" ]; then st "MRL analysis"; X mrl --model M --seeds 0,1,2 > $S/log_mrl.txt 2>&1; X recall --model M > $S/log_recall_M.txt 2>&1; fi
for m in A B; do st "rp10 $m"; X rp10 --model $m > $S/log_rp10_$m.txt 2>&1; done
for m in B A; do st "PQ seeds 3,4 $m"; python supp_all.py pq --model $m --seeds 3,4 >> $S/log_pq_${m}_more.txt 2>&1; done
for m in A B; do X sq4m --model $m > $S/log_sq4m_${m}_final.txt 2>&1; python supp_all.py agg --model $m > $S/agg_final5_$m.txt 2>&1; done
{ for f in log_noname.txt log_withindoc_A.txt log_withindoc_B.txt log_sq4m_A_final.txt log_sq4m_B_final.txt log_recall_A.txt log_recall_B.txt \
           log_binary_A.txt log_binary_B.txt log_mrl.txt log_recall_M.txt log_rp10_A.txt log_rp10_B.txt agg_ladder_A_noname.txt agg_ladder_B_noname.txt; do
    echo; echo "################ $f"; [ -f $S/$f ] && tail -n 45 $S/$f || echo "(missing)"; done; } > $S/EXTRA_SUMMARY.txt
st "DONE"
