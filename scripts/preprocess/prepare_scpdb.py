"""
Prepare scPDB data for ConGLUDe structure-based training.

Pipeline:
1. For each <raw_dir>/<entry_id>/ (e.g. 10mh_1), convert protein.mol2 -> .pdb.
   Conversion preference: `obabel` CLI -> RDKit fallback -> log+skip (never silent).
2. Copy ligand.mol2 to <dataset_dir>/raw/ligand_files/<entry_id>_ligand.mol2.
3. Build random train/val split (default 95/5).
4. Run PDBGraphProcessor(extract_ligands="from_file", labeled_smiles="none")
   and LigandProcessor.

All conversion failures are logged to <dataset_dir>/info/processing_failed.csv
and to stderr.
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from conglude.utils.common import read_list_from_txt, write_list_to_txt
from conglude.utils.data_processing import PDBGraphProcessor, LigandProcessor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", required=True,
                        help="Path to scPDB folder containing <entry>/protein.mol2,ligand.mol2,...")
    parser.add_argument("--dataset_dir", default="./data/datasets/train_datasets/scpdb")
    parser.add_argument("--pdb_dir", default=None,
                        help="Where to stage converted protein .pdb files "
                             "(defaults to <dataset_dir>/raw/pdb_files)")
    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap on number of staged entries (for smoke runs)")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_processing", action="store_true")
    parser.add_argument("--no_resume", action="store_true",
                        help="Disable resume mode: reprocess every staged ID even if its graph .pt already exists.")
    return parser.parse_args()


def have_obabel() -> bool:
    return shutil.which("obabel") is not None


def convert_mol2_to_pdb_obabel(mol2_path: str, pdb_path: str) -> bool:
    """Use Open Babel CLI. Returns True on success."""
    try:
        result = subprocess.run(
            ["obabel", mol2_path, "-O", pdb_path],
            capture_output=True, text=True, timeout=120,
        )
        return result.returncode == 0 and os.path.exists(pdb_path) and os.path.getsize(pdb_path) > 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def convert_mol2_to_pdb_rdkit(mol2_path: str, pdb_path: str) -> bool:
    """RDKit fallback. Returns True on success."""
    try:
        from rdkit import Chem
    except ImportError:
        return False
    mol = Chem.MolFromMol2File(mol2_path, sanitize=False, removeHs=False)
    if mol is None:
        return False
    try:
        Chem.MolToPDBFile(mol, pdb_path)
    except Exception:
        return False
    return os.path.exists(pdb_path) and os.path.getsize(pdb_path) > 0


def convert_protein(mol2_path: str, pdb_path: str, prefer_obabel: bool) -> tuple:
    """Try obabel then RDKit. Return (ok, method)."""
    if prefer_obabel and convert_mol2_to_pdb_obabel(mol2_path, pdb_path):
        return True, "obabel"
    if convert_mol2_to_pdb_rdkit(mol2_path, pdb_path):
        return True, "rdkit"
    return False, "none"


def main():
    args = parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    pdb_dir = os.path.abspath(args.pdb_dir) if args.pdb_dir else os.path.join(dataset_dir, "raw", "pdb_files")
    ligand_dir = os.path.join(dataset_dir, "raw", "ligand_files")
    info_dir = os.path.join(dataset_dir, "info")
    for d in [pdb_dir, ligand_dir, info_dir]:
        os.makedirs(d, exist_ok=True)

    failed_log = os.path.join(info_dir, "processing_failed.csv")
    if not os.path.exists(failed_log):
        with open(failed_log, "w") as f:
            f.write("protein_id,ligand_id,comment\n")

    if not os.path.isdir(args.raw_dir):
        raise FileNotFoundError(args.raw_dir)

    entries = sorted(e for e in os.listdir(args.raw_dir)
                     if os.path.isdir(os.path.join(args.raw_dir, e)))
    if args.limit is not None:
        entries = entries[:args.limit]
        print(f"[smoke] capped to {len(entries)} entries")

    prefer_obabel = have_obabel()
    if not prefer_obabel:
        print("[warn] obabel CLI not found on PATH; using RDKit fallback only.", file=sys.stderr)

    staged = []
    n_obabel = n_rdkit = n_failed = 0
    with open(failed_log, "a") as flog:
        for i, entry in enumerate(entries):
            if i % 200 == 0:
                print(f"[{i}/{len(entries)}] obabel={n_obabel} rdkit={n_rdkit} failed={n_failed}")

            src = os.path.join(args.raw_dir, entry)
            mol2_prot = os.path.join(src, "protein.mol2")
            mol2_lig = os.path.join(src, "ligand.mol2")
            if not os.path.exists(mol2_prot):
                msg = f"{entry},,protein.mol2 not found"
                print(f"[skip] {msg}", file=sys.stderr)
                flog.write(msg + "\n")
                n_failed += 1
                continue
            if not os.path.exists(mol2_lig):
                msg = f"{entry},,ligand.mol2 not found"
                print(f"[skip] {msg}", file=sys.stderr)
                flog.write(msg + "\n")
                n_failed += 1
                continue

            dst_prot = os.path.join(pdb_dir, f"{entry}.pdb")
            if not os.path.exists(dst_prot):
                ok, method = convert_protein(mol2_prot, dst_prot, prefer_obabel)
                if not ok:
                    msg = f"{entry},,mol2->pdb conversion failed (obabel+RDKit)"
                    print(f"[skip] {msg}", file=sys.stderr)
                    flog.write(msg + "\n")
                    if os.path.exists(dst_prot):
                        try:
                            os.remove(dst_prot)
                        except OSError:
                            pass
                    n_failed += 1
                    continue
                if method == "obabel":
                    n_obabel += 1
                else:
                    n_rdkit += 1
            else:
                # Already converted by a previous run.
                n_obabel += 1

            dst_lig = os.path.join(ligand_dir, f"{entry}_ligand.mol2")
            if not os.path.exists(dst_lig):
                shutil.copy(mol2_lig, dst_lig)

            staged.append(entry)

    print(f"Staging summary: {len(staged)} ok / {n_failed} failed "
          f"(obabel={n_obabel}, rdkit fallback={n_rdkit})")

    if not staged:
        print(f"[error] No scPDB entries staged successfully from {args.raw_dir}. "
              f"Verify --raw_dir contains <entry>/protein.mol2,ligand.mol2 directories. "
              f"Aborting before resume logic.", file=sys.stderr)
        sys.exit(1)

    # Random train/val split
    rng = random.Random(args.seed)
    shuffled = staged[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * args.val_fraction)))
    val_ids = sorted(shuffled[:n_val])
    train_ids = sorted(shuffled[n_val:])

    staged_ids = sorted(staged)
    protein_ids_path = os.path.join(info_dir, "protein_ids.txt")
    write_list_to_txt(protein_ids_path, staged_ids)
    write_list_to_txt(os.path.join(info_dir, "train_train_protein_ids.txt"), train_ids)
    write_list_to_txt(os.path.join(info_dir, "train_val_protein_ids.txt"), val_ids)
    print(f"Wrote splits: {len(train_ids)} train / {len(val_ids)} val")

    if args.skip_processing:
        print("--skip_processing set: not running PDBGraphProcessor/LigandProcessor.")
        return

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

    # Resume support: skip pids whose graph .pt already exists, and seed
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
                # graph filename is "{entry}_{ligand_id}.pt"
                done_ids.add(fname[:-3].rsplit("_", 1)[0])

    remaining = [p for p in staged_ids if p not in done_ids]
    if args.no_resume:
        print(f"[resume] disabled — reprocessing all {len(staged_ids)} staged ids")
    else:
        print(f"[resume] {len(done_ids)} ids already have graph .pt files; "
              f"{len(remaining)} of {len(staged_ids)} still to process")

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
    write_list_to_txt(os.path.join(info_dir, "all_protein_ids.txt"), staged_ids)

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
        write_list_to_txt(protein_ids_path, staged_ids)

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
        load_scaler=True,
        save_scaler=False,
    )
    ligand_processor.process()
    print("Done.")


if __name__ == "__main__":
    main()
