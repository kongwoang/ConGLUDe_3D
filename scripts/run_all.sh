#!/usr/bin/env bash
# End-to-end pipeline: preprocess all training datasets + train.
#
# Each stage writes its own log under logs/. set -e aborts on first failure
# so a broken stage never feeds garbage into the next.
#
# Tunables (export before running, or `VAR=val bash scripts/run_all.sh`):
#   PY                  python interpreter (default: python from PATH)
#   DATASET_DIR         where raw data lives (default: ../dataset)
#   NUM_WORKERS         parallelism for preprocess scripts (default: 8)
#   CUDA_VISIBLE_DEVICES  GPU id (default: 0)
#   TRAIN_CONFIG        hydra config name for train.py (default: train_full)
#   RESUME_CKPT         optional last.ckpt path; passed to train.py if set
#
# Skip individual stages by setting:
#   SKIP_PDBBIND=1 SKIP_MERGED=1 SKIP_SCPDB=1 SKIP_MERGE_SB=1 SKIP_TRAIN=1

set -euo pipefail
cd "$(dirname "$0")/.."

PY="${PY:-python}"
DATASET_DIR="${DATASET_DIR:-../dataset}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TRAIN_CONFIG="${TRAIN_CONFIG:-train_full}"
RESUME_CKPT="${RESUME_CKPT:-}"

export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

LOG_DIR=logs
mkdir -p "$LOG_DIR"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "[run_all] $(stamp) — $*"; }

# ----- Stage 1: PDBbind (full, resume-aware) -----
if [ "${SKIP_PDBBIND:-0}" = "0" ]; then
    log "starting prepare_pdbbind.py"
    "$PY" -u scripts/preprocess/prepare_pdbbind.py \
        --raw_dir "$DATASET_DIR/P-L" \
        --index_file "$DATASET_DIR/INDEX_general_PL.2020R1.lst" \
        --dataset_dir ./data/datasets/train_datasets/pdbbind \
        --num_workers "$NUM_WORKERS" \
        > "$LOG_DIR/preprocess_pdbbind_full.log" 2>&1
    log "pdbbind done"
else
    log "SKIP_PDBBIND=1 — skipping"
fi

# ----- Stage 2: MERGED (full, resume-aware) -----
if [ "${SKIP_MERGED:-0}" = "0" ]; then
    log "starting prepare_merged.py"
    "$PY" -u scripts/preprocess/prepare_merged.py \
        --raw_dir "$DATASET_DIR/merged_data" \
        --dataset_dir ./data/datasets/train_datasets/merged \
        --pdb_dir ./data/datasets/train_datasets/merged/raw/pdb_files \
        --num_workers "$NUM_WORKERS" \
        > "$LOG_DIR/preprocess_merged_full.log" 2>&1
    log "merged done"
else
    log "SKIP_MERGED=1 — skipping"
fi

# ----- Stage 3: scPDB (full, resume-aware) -----
if [ "${SKIP_SCPDB:-0}" = "0" ]; then
    log "starting prepare_scpdb.py"
    "$PY" -u scripts/preprocess/prepare_scpdb.py \
        --raw_dir "$DATASET_DIR/scPDB/scPDB" \
        --dataset_dir ./data/datasets/train_datasets/scpdb \
        --num_workers "$NUM_WORKERS" \
        > "$LOG_DIR/preprocess_scpdb_full.log" 2>&1
    log "scpdb done"
else
    log "SKIP_SCPDB=1 — skipping"
fi

# ----- Stage 4: merge sb_combined (pdbbind + scpdb) -----
if [ "${SKIP_MERGE_SB:-0}" = "0" ]; then
    log "starting merge_sb_datasets.py"
    "$PY" -u scripts/preprocess/merge_sb_datasets.py \
        --pdbbind_dir ./data/datasets/train_datasets/pdbbind \
        --scpdb_dir   ./data/datasets/train_datasets/scpdb \
        --out_dir     ./data/datasets/train_datasets/sb_combined \
        --neighbors_dir_name 10_neighbors_10.0_cutoff \
        > "$LOG_DIR/merge_sb_full.log" 2>&1
    log "merge_sb done"
else
    log "SKIP_MERGE_SB=1 — skipping"
fi

# ----- Stage 5: train -----
if [ "${SKIP_TRAIN:-0}" = "0" ]; then
    log "starting train.py (config=$TRAIN_CONFIG)"
    if [ -n "$RESUME_CKPT" ]; then
        "$PY" -u train.py --config-name "$TRAIN_CONFIG" \
            resume_from_checkpoint="$RESUME_CKPT" \
            > "$LOG_DIR/train_full.log" 2>&1
    else
        "$PY" -u train.py --config-name "$TRAIN_CONFIG" \
            > "$LOG_DIR/train_full.log" 2>&1
    fi
    log "train done"
else
    log "SKIP_TRAIN=1 — skipping"
fi

log "all stages complete"
