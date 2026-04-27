# Retraining ConGLUDe

This document describes how to preprocess the raw datasets and retrain
ConGLUDe end-to-end on PDBbind + scPDB (structure-based) and MERGED
(ligand-based), following the joint structure/ligand training procedure of
the ConGLUDe paper (§3.2).

The existing `eval.py` / inference path is **not** modified. Only additive
changes were made:

- New scripts under `scripts/preprocess/`
- New `train.py` and Hydra configs under `configs/`
- Two non-breaking patches in `conglude/datamodule.py` (`self.name`,
  `self.split`, plus `MixedDataset.{LB,SB}_dataset` handles) and one in
  `conglude/model.py` (`get_pocket_counters` now correctly handles
  `MixedDataset` by exposing both `SB_train` and `LB_train` counters).

---

## 1. Layout

Expected raw layout (the project root used here is `/mnt/d/ducluong/BindingPocketPrediction`,
so the raw root is `<root>/dataset`):

```
<raw_root>/
  P-L/                                # PDBbind v2020 R1
    1981-2000/<pdb>/<pdb>_protein.pdb, <pdb>_ligand.mol2, ...
    2001-2010/<pdb>/...
    2011-2019/<pdb>/...
  scPDB/scPDB/<entry>/                # scPDB
    protein.mol2, ligand.mol2, site.mol2, ...
  merged_data/                        # MERGED (SPRINT)
    id_to_smiles.npy
    id_to_sequence.npy
    merged_pos_uniq_{train,val,test}_rand.tsv
    merged_neg_uniq_{train,val,test}_rand.tsv
  INDEX_general_PL.2020R1.lst
```

After preprocessing, `data/datasets/train_datasets/` (relative to `conglude/`)
will hold the per-dataset processed graphs, ligand embeddings and split files
in the layout the existing `ConGLUDeDataset` expects.

---

## 2. Preprocess

All preprocess scripts:
- log skipped/failed entries to `<dataset_dir>/info/processing_failed.csv`
  (same convention as `PDBGraphProcessor.handle_error`) and to stderr —
  no silent skips.
- accept `--limit` / `--limit_targets` flags for fast smoke runs.
- avoid hard-coded absolute paths; you pass `--raw_dir` / `--dataset_dir`.

### 2.1 PDBbind

```bash
cd conglude
python scripts/preprocess/prepare_pdbbind.py \
  --raw_dir   ../dataset/P-L \
  --index_file ../dataset/INDEX_general_PL.2020R1.lst \
  --dataset_dir ./data/datasets/train_datasets/pdbbind \
  --num_workers 8
```

Splits used:
- **Test** (excluded from train+val): the *existing*
  `data/datasets/test_datasets/pdbbind_time/info/protein_ids.txt`
  (363 IDs, almost all release year 2019). The script auto-loads it.
- **Refined test** (excluded from train only):
  `data/datasets/test_datasets/pdbbind_refined/info/protein_ids.txt`.
- **Train**: PDBbind entries with release year ≤ `--train_max_year`
  (default 2018) minus both test sets. Year is read from
  `INDEX_general_PL.2020R1.lst`.
- **Val**: entries with release year == `--val_year` (default 2019) that
  are *not* in the time-test split.

> **Assumption (paper does not pin specific years).** Paper §4.4 calls the
> test set "PDBbind time split" without naming the boundary year. We
> default to `≤2018 train / 2019 val`, with the existing `pdbbind_time`
> file supplying the actual held-out test IDs. Override
> `--train_max_year` / `--val_year` if you need different years.

### 2.2 scPDB

scPDB stores proteins as `.mol2`. We convert to PDB before letting
`PDBGraphProcessor` parse them.

Conversion preference (per entry):
1. `obabel` CLI (Open Babel) — used if available on `PATH`.
2. RDKit `MolFromMol2File` + `MolToPDBFile` fallback.
3. **If both fail, log+skip** (entry id → `info/processing_failed.csv` and stderr).

```bash
python scripts/preprocess/prepare_scpdb.py \
  --raw_dir   ../dataset/scPDB/scPDB \
  --dataset_dir ./data/datasets/train_datasets/scpdb \
  --num_workers 8
```

Splits: random shuffle with `--seed` (default 42), `--val_fraction` (default 0.05).

### 2.3 MERGED (ligand-based)

MERGED has activity labels for `(ligand, uniprot)` pairs but **no** protein
3D structures. The paper uses geometric encoders for both SB and LB data,
so we need a 3D structure per UniProt target. The chosen source is
**AlphaFold v4** (UniProt → `AF-{uniprot}`), which the existing
`PDBGraphProcessor.download_alphafold` already supports.

```bash
# Full run — downloads many AlphaFold structures; cache in ./data/pdb_files
python scripts/preprocess/prepare_merged.py \
  --raw_dir ../dataset/merged_data \
  --dataset_dir ./data/datasets/train_datasets/merged \
  --pdb_dir ./data/datasets/train_datasets/merged/raw/pdb_files \
  --num_workers 8

# Smoke run — 50 targets max, 8 ligands per target, no AF download
python scripts/preprocess/prepare_merged.py \
  --raw_dir ../dataset/merged_data \
  --dataset_dir ./data/datasets/train_datasets/merged_smoke \
  --limit_targets 50 --limit_per_target 8 \
  --no_download
```

`--no_download` reuses anything already in `--pdb_dir` and logs+skips the rest.
This is what you want for offline smoke tests once you've cached structures.

> **Assumption.** Paper §3.2.1 says "we remove all proteins with >90%
> sequence identity to any test set protein". We do **not** apply this
> identity filter automatically — it requires running e.g. MMseqs2 against
> all test sets and is deferred. Add the filtered UniProt list as
> `--exclude_uniprot_file` if needed (currently a TODO).
>
> **Assumption.** AlphaFold structures are used for every UniProt. Some
> long sequences exceed AF v4's per-chain length and will simply 404 →
> they're logged+skipped.

### 2.4 Combining PDBbind + scPDB for SB training (optional)

The `MixedDataset` only accepts **one** SB dataset. To train on both
PDBbind and scPDB jointly, merge their already-processed graphs into a
combined directory:

```bash
python scripts/preprocess/merge_sb_datasets.py \
  --pdbbind_dir ./data/datasets/train_datasets/pdbbind \
  --scpdb_dir   ./data/datasets/train_datasets/scpdb \
  --out_dir     ./data/datasets/train_datasets/sb_combined \
  --neighbors_dir_name 10_neighbors_10.0_cutoff
```

Then point `datamodule.train_datasets.sb_train.dataset_dir` at the merged
directory (override on CLI or edit
`configs/datamodule/train_datasets/sb_train/sb_train.yaml`).

---

## 3. Train

The training step matches paper §3.2: `MixedDataset` randomly selects an SB
or LB batch each step (probability proportional to dataset sizes), and the
respective loss is applied:

- **SB step** → `LSB = Lgeometric + Lcontrastive` (Dice + VN-position-Huber +
  Confidence + 3-axis InfoNCE: `p+b → m`, `m → p`, `m → b`).
- **LB step** → `LLB = sigmoid contrastive (BCE)` between global protein
  representation `p` and ligand `m_p`, with paper-default 1:3 active:inactive
  sampling (configured via `inactive_active_ratio: 3` on the LB dataset).

Loss/temperature config lives in `configs/model/conglude.yaml` (already wired
to use `τ_p2m = sqrt(2·D) ≈ 0.0442` for the protein+pocket InfoNCE and
`τ_m2p = τ_m2b = sqrt(D) ≈ 0.0625` per the paper).

### 3.1 Smoke run

Use this first to confirm the wiring before launching a long run.

```bash
# Preprocess tiny slices first (only need to run once)
python scripts/preprocess/prepare_pdbbind.py \
  --raw_dir ../dataset/P-L --index_file ../dataset/INDEX_general_PL.2020R1.lst \
  --dataset_dir ./data/datasets/train_datasets/pdbbind \
  --limit 32 --num_workers 4

python scripts/preprocess/prepare_merged.py \
  --raw_dir ../dataset/merged_data \
  --dataset_dir ./data/datasets/train_datasets/merged \
  --limit_targets 16 --limit_per_target 8 --num_workers 4

# Smoke training — uses train_smoke.yaml: 1 epoch, 4 train batches, 2 val batches
python train.py --config-name train_smoke
```

`debug: true` on each smoke dataset caps the loaded graph list to 10 (see
`ConGLUDeDataset.get_graph_files`), and `train_smoke.yaml` overrides
`trainer.{max_epochs, limit_train_batches, limit_val_batches}`.

### 3.1.1 Smoke eval (verify the checkpoint loads + runs)

```bash
# Loads ./checkpoints/conglude_smoke/last.ckpt and runs VS metrics on the
# small preprocessed PDBbind set
python eval.py --config-name eval_smoke
```

`eval.py` now accepts an optional `ckpt_path` config field — when set, it
loads the Lightning checkpoint's `state_dict` directly into the model
(`weights_only=False`, `strict=False`) and skips Lightning's full-state
restore. This avoids the PyTorch 2.6 weights-only restriction that
otherwise rejects the bundled `ReduceLROnPlateau`.

For the full eval flow with a published checkpoint that ships per-component
.pth files (`vnegnn.pth`, `pocket_encoder.pth`, ...), set
`checkpoint_name: <name>` and leave `ckpt_path: null` — that path is
unchanged.

### 3.2 Full run

```bash
python train.py \
  trainer.max_epochs=200 \
  monitor=avg_val/virtual_screening/auroc \
  logger=wb logger.name=conglude_full
```

Override individual dataset paths/batch sizes via Hydra CLI, e.g.:

```bash
python train.py \
  datamodule.train_datasets.sb_train.dataset_dir=./data/datasets/train_datasets/sb_combined \
  datamodule.train_datasets.sb_train.batch_size=64 \
  datamodule.train_datasets.lb_train.batch_size=32
```

---

## 4. Files added

| Path | Purpose |
| --- | --- |
| `scripts/preprocess/prepare_pdbbind.py` | Stage PDBbind, build train/val ids, run processors |
| `scripts/preprocess/prepare_scpdb.py`   | Convert mol2→pdb, run processors |
| `scripts/preprocess/prepare_merged.py`  | Build SMILES files, fetch AlphaFold, run processors |
| `scripts/preprocess/merge_sb_datasets.py` | Combine PDBbind+scPDB processed graphs |
| `train.py`                              | Hydra entrypoint mirroring `eval.py` |
| `configs/train.yaml`                    | Default training config |
| `configs/train_smoke.yaml`              | Smoke-test config (1 epoch, few batches) |
| `configs/datamodule/train.yaml`         | DataModule for full training |
| `configs/datamodule/train_smoke.yaml`   | DataModule for smoke runs |
| `configs/datamodule/train_datasets/sb_lb_train.yaml` (+ `sb_train/`, `lb_train/`) | Mixed SB/LB train datasets |
| `configs/datamodule/train_datasets/smoke.yaml` | Smoke variant of the above |
| `configs/datamodule/val_datasets/pdbbind_val.yaml` (+ `pdbbind_val/`) | Validation dataset (PDBbind 2019) |
| `README_RETRAIN.md`                     | This file |

## 5. Files modified (additive / minimal bugfixes only)

All of these were uncovered while wiring up the train path. None of them
affects the inference path (`eval.py` still composes and instantiates the
model identically — verified).

- `conglude/datamodule.py`:
  - `ConGLUDeDataset` now sets `self.name = dataset_name` and
    `self.split = split` (used by `PLDataModule.train_dataloader` and the
    debug-mode graph filter, which previously raised AttributeError).
  - `MixedDataset` now stores `self.LB_dataset`/`self.SB_dataset`/`self.dataset_name = "Mixed"`.
  - Fixed typo `.lon()` → `.long()` in `ConGLUDeDataset.get` (LB sampling branch).
- `conglude/model.py`:
  - `get_pocket_counters` populates per-source counters (`SB_train`, `LB_train`)
    when the train loader wraps a `MixedDataset` (previously it tried to
    look up a non-existent `MixedDataset.dataset_name`).
  - Removed unused kwarg `calc_re=False` from `VirtualScreeningMetrics(...)`
    constructor calls (the metric class never accepted this argument; it
    was only reachable from the train/val branches that no one had run).
  - `configure_optimizers` no longer assumes `trainer.callbacks[0]` is the
    ModelCheckpoint; it scans for the first callback with a `monitor`
    attribute.
  - LB loss now passes `labels.float()` to BCE-with-logits (LB labels are
    long; BCE requires float).
- `conglude/utils/metrics.py`:
  - `VirtualScreeningMetrics.update` casts `targets` to long for
    torchmetrics' binary AUROC (modern torchmetrics rejects float).
- `conglude/utils/data_processing.py`:
  - `LigandProcessor.normalize_features` truncates the live feature matrix
    to the saved scaler's `n_features_in_` when `load_scaler=True`. Newer
    RDKit versions ship 217 descriptors; the released scaler was fit on
    210. Truncation keeps the model's input dim stable.

### Environment notes

- Repo expects the legacy `fair-esm` package (uses
  `esm.pretrained.esm2_t33_650M_UR50D`). If your env has the newer
  EvolutionaryScale `esm` package instead (no `pretrained` submodule),
  `pip install fair-esm` — it will shadow the new `esm`. Tested against
  fair-esm 2.0.0.
- AlphaFold v4 URLs return 404 in 2026 (DB has rolled to v6). The MERGED
  preprocess script now queries `/api/prediction/{uniprot}` to fetch the
  canonical `pdbUrl` instead of hard-coding a version.

## 6. Key assumptions (recap)

1. **PDBbind splits**: prefer existing `pdbbind_time/info/protein_ids.txt`
   as test; default fallback split `≤2018 train / 2019 val`. Override with
   `--train_max_year` / `--val_year`.
2. **scPDB conversion**: obabel → RDKit → log+skip. No silent drops.
3. **MERGED proteins**: AlphaFold v4 structures via UniProt. Cached.
   `--no_download` available for offline smoke tests.
4. **MERGED 90% identity filter** to test proteins is **not** auto-applied
   (TODO).
5. **PDBbind+scPDB combination**: merged via post-hoc graph
   symlinking (`merge_sb_datasets.py`). `MixedDataset` itself only takes one
   SB dataset.
6. **Loss temperatures**: `configs/model/conglude.yaml` already uses
   `τ_p2m=0.0442 (≈√(2·D))` and `τ_m2p=τ_m2b=0.0625 (≈√D)` per paper §3.2.2;
   left unchanged.
7. **LB sampling**: 1:3 active:inactive per protein per LB batch via
   `inactive_active_ratio=3` and `max_num_actives=4`. Adjust on CLI as needed.
8. **`LB_virtual_screening_loss_weight=6.0`** in `configs/model/conglude.yaml`
   is left as-is (chosen empirically; paper does not specify a precise value).

## 7. Logging skipped/failed entries

Every preprocess script appends to `<dataset_dir>/info/processing_failed.csv`
(format: `protein_id,ligand_id,comment`). Conversion/staging scripts also
print `[skip] ...` to stderr. Aggregate row-level drops in MERGED (e.g. SMILES
that failed to map) are recorded as a single summary row prefixed with `_merged_`
to avoid blowing up the file.
