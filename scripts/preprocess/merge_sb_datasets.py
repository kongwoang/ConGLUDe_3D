"""
Merge processed PDBbind and scPDB graph datasets into a single SB train dir.

Why: ConGLUDe paper trains on PDBbind+scPDB jointly, but `PLDataModule` only
accepts ONE structure-based train dataset. This script writes remapped `.pt`
graph files and ligand metadata into a combined dataset_dir and uses that as
the SB_train dataset_dir.

This script does NOT re-run feature extraction — it expects both source dirs
to already be processed by `prepare_pdbbind.py` and `prepare_scpdb.py`. It
rewrites graph files with union ligand indices, unions ligand metadata, and
writes joint split files.

Note: ligand fingerprints/descriptors are tied to per-dataset SMILES indexing.
Combining ligands across datasets requires re-running LigandProcessor on the
union. We do that automatically here.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from conglude.utils.common import read_list_from_txt, write_list_to_txt, read_json, write_json
from conglude.utils.data_processing import LigandProcessor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdbbind_dir", required=True)
    parser.add_argument("--scpdb_dir", required=True)
    parser.add_argument("--out_dir", required=True,
                        help="Combined SB dataset_dir (will be created)")
    parser.add_argument("--neighbors_dir_name", default="10_neighbors_10.0_cutoff",
                        help="Subdir name under processed/graphs (must match the cutoff/neighbors used)")
    parser.add_argument("--copy", action="store_true",
                        help="Deprecated; merged graph files are always materialized so ligand indices can be remapped.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--skip_ligand_processor", action="store_true")
    return parser.parse_args()


def merge_id_lists(out_info, src_infos, file_name):
    union = set()
    for info in src_infos:
        path = os.path.join(info, file_name)
        if os.path.exists(path):
            union.update(read_list_from_txt(path))
    union = sorted(union)
    write_list_to_txt(os.path.join(out_info, file_name), union)
    return len(union)


def remap_ligand_indices(indices, old_index2smiles, new_smiles):
    return [new_smiles[old_index2smiles[str(int(i))]] for i in indices]


def merge_index2smiles(src_dirs, out_dir):
    """Merge per-dataset index2smiles.json into one (re-indexed)."""
    new_smiles = {}
    new_index2smiles = {}
    for src in src_dirs:
        path = os.path.join(src, "processed", "ligand_embeddings", "index2smiles.json")
        if not os.path.exists(path):
            print(f"[warn] no index2smiles.json in {src}", file=sys.stderr)
            continue
        d = read_json(path)
        for k, smi in d.items():
            if smi not in new_smiles:
                new_smiles[smi] = len(new_smiles)
    for smi, i in new_smiles.items():
        new_index2smiles[str(i)] = smi
    out_path = os.path.join(out_dir, "processed", "ligand_embeddings", "index2smiles.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_json(out_path, new_index2smiles)
    print(f"Merged SMILES vocabulary: {len(new_smiles)} unique entries")
    return new_smiles


def merge_graphs(out_graph_dir, src_roots, neighbors_dir_name, new_smiles):
    n = 0
    for src in src_roots:
        graph_dir = os.path.join(src, "processed", "graphs", neighbors_dir_name)
        if not os.path.isdir(graph_dir):
            print(f"[warn] graph dir not found: {graph_dir}", file=sys.stderr)
            continue

        index2smiles_path = os.path.join(src, "processed", "ligand_embeddings", "index2smiles.json")
        if not os.path.exists(index2smiles_path):
            print(f"[warn] no index2smiles.json in {src}; skipping graphs", file=sys.stderr)
            continue
        old_index2smiles = read_json(index2smiles_path)

        for fname in os.listdir(graph_dir):
            if not fname.endswith(".pt"):
                continue

            dst = os.path.join(out_graph_dir, fname)
            if os.path.lexists(dst):
                os.remove(dst)

            graph = torch.load(os.path.join(graph_dir, fname), weights_only=False)
            graph.actives = remap_ligand_indices(graph.actives, old_index2smiles, new_smiles)
            graph.inactives = remap_ligand_indices(graph.inactives, old_index2smiles, new_smiles)
            graph.labeled_ligands = remap_ligand_indices(graph.labeled_ligands, old_index2smiles, new_smiles)
            torch.save(graph, dst)
            n += 1
    return n


def main():
    args = parse_args()

    out_dir = os.path.abspath(args.out_dir)
    out_info = os.path.join(out_dir, "info")
    out_graph_dir = os.path.join(out_dir, "processed", "graphs", args.neighbors_dir_name)
    out_ligand_dir = os.path.join(out_dir, "processed", "ligand_embeddings")
    for d in [out_info, out_graph_dir, out_ligand_dir]:
        os.makedirs(d, exist_ok=True)

    src_infos = [os.path.join(args.pdbbind_dir, "info"),
                 os.path.join(args.scpdb_dir, "info")]
    src_roots = [args.pdbbind_dir, args.scpdb_dir]

    print("Merging id splits ...")
    n_proc = merge_id_lists(out_info, src_infos, "processed_protein_ids.txt")
    n_all  = merge_id_lists(out_info, src_infos, "protein_ids.txt")
    n_train = merge_id_lists(out_info, src_infos, "train_train_protein_ids.txt")
    n_val  = merge_id_lists(out_info, src_infos, "train_val_protein_ids.txt")
    print(f"  processed={n_proc} all={n_all} train={n_train} val={n_val}")

    # Merge SMILES vocabulary then re-run LigandProcessor on the union
    new_smiles = merge_index2smiles(src_roots, out_dir)

    n = merge_graphs(out_graph_dir, src_roots, args.neighbors_dir_name, new_smiles)
    print(f"Merged {n} graph .pt files into {out_graph_dir}")

    if args.skip_ligand_processor:
        print("--skip_ligand_processor set: not running LigandProcessor.")
        return

    print("Running LigandProcessor on the merged SMILES vocabulary ...")
    lp = LigandProcessor(
        dataset_dir=out_dir,
        num_workers=args.num_workers,
        load_scaler=True,
        save_scaler=False,
    )
    lp.process()
    print("Done.")


if __name__ == "__main__":
    main()
