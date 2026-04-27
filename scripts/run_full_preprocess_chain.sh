#!/usr/bin/env bash
# Sequential preprocessing chain:
#   1. wait for prepare_merged.py (if still running)
#   2. prepare_pdbbind.py    (full, with resume)
#   3. prepare_scpdb.py      (full, with resume)
#
# merge_sb_datasets.py is intentionally NOT run here — invoke it separately
# once pdbbind+scpdb have finished if/when the combined SB train dir is needed.
# Each step writes to its own log under logs/. set -e aborts on first failure.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=/home/bkai/miniconda3/envs/bpp/bin/python
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
LOG_DIR=logs
mkdir -p "$LOG_DIR"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

# Stage 1: PDBbind (full, resume-aware)
# NOTE: prepare_merged.py may still be running on the same GPU. Disk I/O on
# /mnt/d/ is shared, so both will run slower than serial — accepted by user.
echo "[chain] $(stamp) — starting prepare_pdbbind.py"
"$PY" -u scripts/preprocess/prepare_pdbbind.py \
    --raw_dir ../dataset/P-L \
    --index_file ../dataset/INDEX_general_PL.2020R1.lst \
    --dataset_dir ./data/datasets/train_datasets/pdbbind \
    --num_workers 8 \
    > "$LOG_DIR/preprocess_pdbbind_full.log" 2>&1
echo "[chain] $(stamp) — pdbbind done"

# Stage 2: scPDB (full, resume-aware)
echo "[chain] $(stamp) — starting prepare_scpdb.py"
"$PY" -u scripts/preprocess/prepare_scpdb.py \
    --raw_dir ../dataset/scPDB/scPDB \
    --dataset_dir ./data/datasets/train_datasets/scpdb \
    --num_workers 8 \
    > "$LOG_DIR/preprocess_scpdb_full.log" 2>&1
echo "[chain] $(stamp) — scpdb done"

echo "[chain] $(stamp) — all stages complete (pdbbind + scpdb)"
