"""
Prepare PDBbind v2020 R1 data for ConGLUDe structure-based training.

Pipeline:
1. Read raw structures from --raw_dir (year-bucketed: 1981-2000/2001-2010/2011-2019).
2. Stage protein PDBs and ligand mol2 files into the dataset_dir layout expected
   by `PDBGraphProcessor(extract_ligands="from_file", labeled_smiles="none")`.
3. Build train/val ID lists.
   - Test split: existing `data/datasets/test_datasets/pdbbind_time/info/protein_ids.txt`
     (363 ids, ~2019). Excluded from train/val.
   - Refined test split: existing `pdbbind_refined/info/protein_ids.txt` also excluded
     from train.
   - Train: years <= --train_max_year (default 2018) minus the two test sets.
   - Val: year == --val_year (default 2019) entries not in time-test.
4. Run `PDBGraphProcessor` and `LigandProcessor`.

Skipped entries are logged to <dataset_dir>/info/processing_failed.csv (the
processor already does this for every failed PDB) and to stdout.
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Make `conglude` importable when this script is run directly
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from conglude.utils.common import read_list_from_txt, write_list_to_txt
from conglude.utils.data_processing import PDBGraphProcessor, LigandProcessor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", required=True,
                        help="Path to raw PDBbind directory containing 1981-2000/, 2001-2010/, 2011-2019/")
    parser.add_argument("--index_file", required=True,
                        help="Path to INDEX_general_PL.2020R1.lst")
    parser.add_argument("--dataset_dir", default="./data/datasets/train_datasets/pdbbind",
                        help="Destination dataset directory")
    parser.add_argument("--pdb_dir", default=None,
                        help="Where to stage protein .pdb files (defaults to <dataset_dir>/raw/pdb_files)")
    parser.add_argument("--time_test_ids",
                        default="./data/datasets/test_datasets/pdbbind_time/info/protein_ids.txt",
                        help="Existing pdbbind_time test split (excluded from train/val).")
    parser.add_argument("--refined_test_ids",
                        default="./data/datasets/test_datasets/pdbbind_refined/info/protein_ids.txt",
                        help="Existing pdbbind_refined test split (excluded from train).")
    parser.add_argument("--train_max_year", type=int, default=2018)
    parser.add_argument("--val_year", type=int, default=2019)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap on number of staged PDBs (for smoke runs)")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_processing", action="store_true",
                        help="Only stage files and write splits; don't run PDBGraphProcessor/LigandProcessor.")
    parser.add_argument("--no_resume", action="store_true",
                        help="Disable resume mode: reprocess every staged ID even if its graph .pt already exists.")
    return parser.parse_args()


def parse_index(index_file: str) -> dict:
    """Return {pdb_id_lower: release_year}."""
    years = {}
    with open(index_file) as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                year = int(parts[2])
            except ValueError:
                continue
            years[parts[0].lower()] = year
    return years


def stage_files(raw_dir: str, pdb_dir: str, ligand_dir: str, ids: list,
                failed_log: str) -> list:
    """Copy <pdb>_protein.pdb and <pdb>_ligand.mol2 into the staging dirs.

    Returns the list of pdb_ids that were successfully staged.
    """
    os.makedirs(pdb_dir, exist_ok=True)
    os.makedirs(ligand_dir, exist_ok=True)

    # Build a flat map: lowercase pdb_id -> source folder.
    # Supports both PDBbind layouts:
    #   (A) year-bucketed:  <raw_dir>/{1981-2000,2001-2010,2011-2019}/<entry>/
    #   (B) flat general:   <raw_dir>/<entry>/  (e.g. PDBbind_v2020 general set)
    # An "entry folder" is detected by the presence of <entry>_protein.pdb.
    def looks_like_entry(d):
        try:
            return any(f.endswith("_protein.pdb") for f in os.listdir(d))
        except OSError:
            return False

    src_map = {}
    n_flat = n_nested = 0
    for first in os.listdir(raw_dir):
        first_path = os.path.join(raw_dir, first)
        if not os.path.isdir(first_path):
            continue
        if looks_like_entry(first_path):
            # Layout B: this directory is an entry folder itself
            src_map[first.lower()] = first_path
            n_flat += 1
        else:
            # Layout A: this is a year-bucket; entries are one level deeper
            for entry in os.listdir(first_path):
                entry_path = os.path.join(first_path, entry)
                if os.path.isdir(entry_path):
                    src_map[entry.lower()] = entry_path
                    n_nested += 1

    print(f"[stage] discovered {len(src_map)} candidate entry folders in "
          f"{raw_dir} (flat={n_flat}, year-bucketed={n_nested})")
    if not src_map:
        print(f"[stage] WARNING: no entry-style folders (containing *_protein.pdb) "
              f"found under {raw_dir}", file=sys.stderr)

    staged = []
    with open(failed_log, "a") as flog:
        for pid in ids:
            src = src_map.get(pid.lower())
            if src is None:
                msg = f"{pid},,raw structure folder not found in {raw_dir}"
                print(f"[skip] {msg}", file=sys.stderr)
                flog.write(msg + "\n")
                continue

            prot_src = os.path.join(src, f"{pid.lower()}_protein.pdb")
            lig_src = os.path.join(src, f"{pid.lower()}_ligand.mol2")

            if not os.path.exists(prot_src):
                msg = f"{pid},,missing _protein.pdb at {prot_src}"
                print(f"[skip] {msg}", file=sys.stderr)
                flog.write(msg + "\n")
                continue
            if not os.path.exists(lig_src):
                msg = f"{pid},,missing _ligand.mol2 at {lig_src}"
                print(f"[skip] {msg}", file=sys.stderr)
                flog.write(msg + "\n")
                continue

            dst_prot = os.path.join(pdb_dir, f"{pid.lower()}.pdb")
            dst_lig = os.path.join(ligand_dir, f"{pid.lower()}_ligand.mol2")

            if not os.path.exists(dst_prot):
                shutil.copy(prot_src, dst_prot)
            if not os.path.exists(dst_lig):
                shutil.copy(lig_src, dst_lig)

            staged.append(pid.lower())

    return staged


def main():
    args = parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    pdb_dir = os.path.abspath(args.pdb_dir) if args.pdb_dir else os.path.join(dataset_dir, "raw", "pdb_files")
    ligand_dir = os.path.join(dataset_dir, "raw", "ligand_files")
    info_dir = os.path.join(dataset_dir, "info")
    os.makedirs(info_dir, exist_ok=True)

    failed_log = os.path.join(info_dir, "processing_failed.csv")
    if not os.path.exists(failed_log):
        with open(failed_log, "w") as f:
            f.write("protein_id,ligand_id,comment\n")

    # 1. Parse year index and exclusion sets
    print(f"Reading INDEX file: {args.index_file}")
    pid2year = parse_index(args.index_file)

    time_test = set()
    if os.path.exists(args.time_test_ids):
        time_test = {p.lower() for p in read_list_from_txt(args.time_test_ids)}
        print(f"Excluding {len(time_test)} ids from pdbbind_time test split")
    else:
        print(f"[warn] {args.time_test_ids} not found — no time-test exclusion applied")

    refined_test = set()
    if os.path.exists(args.refined_test_ids):
        refined_test = {p.lower() for p in read_list_from_txt(args.refined_test_ids)}
        print(f"Excluding {len(refined_test)} ids from pdbbind_refined test split (train only)")

    # 2. Build train/val candidate id lists
    all_pids = sorted(pid2year.keys())
    train_ids = [p for p in all_pids
                 if pid2year[p] <= args.train_max_year
                 and p not in time_test
                 and p not in refined_test]
    val_ids = [p for p in all_pids
               if pid2year[p] == args.val_year
               and p not in time_test]

    if args.limit is not None:
        train_ids = train_ids[:args.limit]
        val_ids = val_ids[:max(1, args.limit // 5)]
        print(f"[smoke] capped to {len(train_ids)} train / {len(val_ids)} val")

    print(f"Candidates: {len(train_ids)} train, {len(val_ids)} val")

    # 3. Stage files for the union of train + val
    union_ids = sorted(set(train_ids) | set(val_ids))
    print(f"Staging {len(union_ids)} PDBbind entries to {pdb_dir} and {ligand_dir}")
    staged = stage_files(args.raw_dir, pdb_dir, ligand_dir, union_ids, failed_log)
    staged_set = set(staged)
    print(f"Successfully staged {len(staged)} / {len(union_ids)}")

    if not staged:
        print(f"[error] No raw PDBbind entries were staged from {args.raw_dir}. "
              f"Verify --raw_dir points at the directory containing entry folders "
              f"(each with <pdb_id>_protein.pdb and <pdb_id>_ligand.mol2). "
              f"Aborting before resume logic.", file=sys.stderr)
        sys.exit(1)

    # 4. Filter id lists by what was actually staged, and write split files
    train_ids = [p for p in train_ids if p in staged_set]
    val_ids = [p for p in val_ids if p in staged_set]

    protein_ids_path = os.path.join(info_dir, "protein_ids.txt")
    write_list_to_txt(protein_ids_path, staged)
    write_list_to_txt(os.path.join(info_dir, "train_train_protein_ids.txt"), train_ids)
    write_list_to_txt(os.path.join(info_dir, "train_val_protein_ids.txt"), val_ids)
    print(f"Wrote split files in {info_dir}")

    if args.skip_processing:
        print("--skip_processing set: not running PDBGraphProcessor/LigandProcessor.")
        return

    # 5. Run graph + ligand processors
    print(f"Running PDBGraphProcessor on {dataset_dir} ...")
    graph_processor = PDBGraphProcessor(
        dataset_dir=dataset_dir,
        pdb_dir=pdb_dir,
        overwrite=args.overwrite,
        num_workers=args.num_workers,
        extract_ligands="from_file",
        select_chains="all",
        labeled_smiles="none",
        multi_ligand=False,
        load_pocket=False,
        multi_pdb_targets=False,
        calc_mol_feats=True,
        save_cleaned_pdbs=False,
        save_complex_info=False,
    )

    # 5a. Resume support: skip pids whose graph .pt already exists, and seed
    # the SMILES->index map from the existing index2smiles.json so previously
    # saved graphs keep referencing correct rows in the rebuilt ECFP table.
    cutoff_dir = (
        f"{graph_processor.max_neighbors}_neighbors_"
        f"{graph_processor.neighbor_dist_cutoff}_cutoff"
    )
    graphs_dir = os.path.join(dataset_dir, "processed", "graphs", cutoff_dir)
    done_ids = set()
    if not args.no_resume and os.path.isdir(graphs_dir):
        for fname in os.listdir(graphs_dir):
            if fname.endswith(".pt"):
                # graph filename is "{pid}_{ligand_id}.pt"
                done_ids.add(fname[:-3].rsplit("_", 1)[0].lower())

    remaining = [p for p in staged if p not in done_ids]
    if args.no_resume:
        print(f"[resume] disabled — reprocessing all {len(staged)} staged ids")
    else:
        print(f"[resume] {len(done_ids)} ids already have graph .pt files; "
              f"{len(remaining)} of {len(staged)} still to process")

    # Snapshot state we'll need to merge after the processor runs
    processed_path = os.path.join(info_dir, "processed_protein_ids.txt")
    prev_processed = read_list_from_txt(processed_path) if os.path.exists(processed_path) else []

    index2smiles_path = os.path.join(
        dataset_dir, "processed", "ligand_embeddings", "index2smiles.json"
    )
    if not args.no_resume and os.path.exists(index2smiles_path):
        with open(index2smiles_path) as f:
            prev_index2smiles = json.load(f)
        # JSON gives {idx_str: smiles}; reverse to {smiles: int(idx)} to seed the processor
        graph_processor.smiles2index_dict = {v: int(k) for k, v in prev_index2smiles.items()}
        print(f"[resume] seeding processor with {len(graph_processor.smiles2index_dict)} existing SMILES indices")

    # Keep the full staged list aside; feed only the remaining ones to the processor
    write_list_to_txt(os.path.join(info_dir, "all_protein_ids.txt"), staged)

    if not remaining:
        print("[resume] nothing to process — all staged IDs already have graphs. "
              "Skipping graph + ligand processors.")
        # protein_ids.txt already holds the full staged list (written above).
        return

    write_list_to_txt(protein_ids_path, remaining)

    try:
        graph_processor.process()
    finally:
        # Always restore protein_ids.txt to the full staged list so downstream
        # consumers (training, eval) see every ID, not just this run's batch.
        write_list_to_txt(protein_ids_path, staged)

    # Merge the processor's freshly written processed_protein_ids.txt with
    # previously processed ids (the processor overwrites this file).
    new_processed = read_list_from_txt(processed_path) if os.path.exists(processed_path) else []
    merged = sorted(set(prev_processed) | set(new_processed))
    write_list_to_txt(processed_path, merged)
    print(f"[resume] processed_protein_ids.txt now lists {len(merged)} ids "
          f"(prev={len(prev_processed)}, new_run={len(new_processed)})")

    print("Running LigandProcessor ...")
    ligand_processor = LigandProcessor(
        dataset_dir=dataset_dir,
        num_workers=args.num_workers,
        load_scaler=True,   # use shared scaler; set False to fit per-dataset
        save_scaler=False,
    )
    ligand_processor.process()
    print("Done.")


if __name__ == "__main__":
    main()
