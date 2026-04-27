"""
Prepare the MERGED ligand-based dataset for ConGLUDe.

Pipeline:
1. Read merged_pos/neg_uniq_{train,val,test}_rand.tsv (cols: ligand, aa_seq) and
   id_to_smiles.npy (ligand_id -> SMILES). aa_seq is a UniProt accession.
2. For each unique UniProt target seen in the chosen splits, write
   raw/smiles_files/AF-{uniprot}/actives.txt and inactives.txt.
3. Write info/protein_ids.txt = ["AF-{uniprot}", ...].
4. Optionally download AlphaFold v4 structures into <pdb_dir> (cached). Use
   --no_download to skip (then PDBGraphProcessor will pick up whatever PDBs
   you have already cached, and log+skip the rest).
5. Write per-task split files
       train_train_protein_ids.txt
       train_val_protein_ids.txt
   (train task because the model is in train/val mode for these dataloaders)
6. Run PDBGraphProcessor(extract_ligands="none", labeled_smiles="binary")
   and LigandProcessor.

Skipped entries (missing structures, missing SMILES) are appended to
<dataset_dir>/info/processing_failed.csv.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from conglude.utils.common import read_list_from_txt, write_list_to_txt
from conglude.utils.data_processing import PDBGraphProcessor, LigandProcessor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_dir", required=True,
                        help="Path to merged_data/ folder containing the .tsv files "
                             "and id_to_smiles.npy")
    parser.add_argument("--dataset_dir", default="./data/datasets/train_datasets/merged")
    parser.add_argument("--pdb_dir", default=None,
                        help="Where AlphaFold structures are cached / will be downloaded "
                             "(defaults to <dataset_dir>/raw/pdb_files)")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        help="Which TSV splits to process. 'test' is allowed but rarely useful here.")
    parser.add_argument("--limit_targets", type=int, default=None,
                        help="Cap number of UniProt targets staged (for smoke runs)")
    parser.add_argument("--limit_per_target", type=int, default=None,
                        help="Cap actives & inactives per target (for smoke runs)")
    parser.add_argument("--no_download", action="store_true",
                        help="Skip AlphaFold downloads — use only cached PDBs in --pdb_dir.")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_processing", action="store_true")
    parser.add_argument("--no_resume", action="store_true",
                        help="Disable resume mode: reprocess every staged ID even if its graph .pt already exists.")
    return parser.parse_args()


def af_id(uniprot: str) -> str:
    return f"AF-{uniprot}"


def load_smiles_map(raw_dir: str) -> dict:
    path = os.path.join(raw_dir, "id_to_smiles.npy")
    print(f"Loading {path} (this can take a moment)")
    arr = np.load(path, allow_pickle=True).item()
    return arr


def build_target_ligands(raw_dir: str, splits, smiles_map: dict,
                        limit_targets: int, limit_per_target: int,
                        failed_log_fh):
    """Return {split -> {uniprot -> (actives_smiles, inactives_smiles)}}."""

    out = {}
    target_universe = None  # global cap, applied across splits

    for split in splits:
        pos_path = os.path.join(raw_dir, f"merged_pos_uniq_{split}_rand.tsv")
        neg_path = os.path.join(raw_dir, f"merged_neg_uniq_{split}_rand.tsv")
        if not (os.path.exists(pos_path) and os.path.exists(neg_path)):
            print(f"[warn] missing TSV for split={split}: {pos_path}, {neg_path}",
                  file=sys.stderr)
            continue

        # Read pos/neg lazily; expect cols 'ligand' (int id) and 'aa_seq' (uniprot)
        print(f"Reading {pos_path}")
        pos_df = pd.read_csv(pos_path, sep="\t")
        print(f"Reading {neg_path}")
        neg_df = pd.read_csv(neg_path, sep="\t")

        # Map ligand id -> smiles, drop unknown ones
        def map_smiles(df, label):
            df = df.copy()
            df["smiles"] = df["ligand"].map(smiles_map)
            n_before = len(df)
            df = df[df["smiles"].notna() & (df["smiles"].astype(str).str.len() > 0)]
            n_dropped = n_before - len(df)
            if n_dropped:
                print(f"  {label} split={split}: dropped {n_dropped} rows with missing SMILES")
                # Don't dump millions of rows; record an aggregate failure marker.
                failed_log_fh.write(
                    f"_merged_{split}_{label},,dropped_{n_dropped}_rows_missing_smiles\n"
                )
            return df

        pos_df = map_smiles(pos_df, "pos")
        neg_df = map_smiles(neg_df, "neg")

        # Group by uniprot
        per_target = {}
        for uniprot, group in pos_df.groupby("aa_seq"):
            actives = group["smiles"].drop_duplicates().tolist()
            per_target.setdefault(uniprot, [[], []])[0] = actives
        for uniprot, group in neg_df.groupby("aa_seq"):
            inactives = group["smiles"].drop_duplicates().tolist()
            per_target.setdefault(uniprot, [[], []])[1] = inactives

        # Drop targets with no actives at all
        per_target = {u: v for u, v in per_target.items() if len(v[0]) > 0}

        # Apply per-target ligand cap
        if limit_per_target is not None:
            for u in per_target:
                per_target[u][0] = per_target[u][0][:limit_per_target]
                per_target[u][1] = per_target[u][1][:limit_per_target * 3]

        out[split] = per_target

    # Apply global target cap consistently across splits.
    # Take train targets first (if present), then add val targets not yet seen.
    if limit_targets is not None:
        ordered_targets = []
        seen = set()
        for split in ["train"] + [s for s in splits if s != "train"]:
            if split not in out:
                continue
            for u in sorted(out[split].keys()):
                if u not in seen:
                    seen.add(u)
                    ordered_targets.append(u)
                if len(ordered_targets) >= limit_targets:
                    break
            if len(ordered_targets) >= limit_targets:
                break
        keep = set(ordered_targets[:limit_targets])
        for split in out:
            out[split] = {u: v for u, v in out[split].items() if u in keep}
        print(f"[smoke] limited to {len(keep)} targets total across splits")

    return out


def write_smiles_files(dataset_dir: str, per_target_per_split: dict) -> set:
    """Write raw/smiles_files/AF-{uniprot}/{actives,inactives}.txt.

    Files are written using the union of all splits. Returns the set of
    uniprots actually written.
    """
    smiles_root = os.path.join(dataset_dir, "raw", "smiles_files")
    union = {}
    for split, mapping in per_target_per_split.items():
        for u, (actives, inactives) in mapping.items():
            agg = union.setdefault(u, [set(), set()])
            agg[0].update(actives)
            agg[1].update(inactives)

    written = set()
    for u, (actives, inactives) in union.items():
        target_dir = os.path.join(smiles_root, af_id(u))
        os.makedirs(target_dir, exist_ok=True)
        with open(os.path.join(target_dir, "actives.txt"), "w") as f:
            f.write("\n".join(sorted(actives)) + ("\n" if actives else ""))
        with open(os.path.join(target_dir, "inactives.txt"), "w") as f:
            f.write("\n".join(sorted(inactives)) + ("\n" if inactives else ""))
        written.add(u)
    return written


def fetch_alphafold(uniprot: str, pdb_dir: str, session: requests.Session,
                    failed_log_fh) -> bool:
    """Download the canonical AlphaFold PDB for `uniprot` if missing.

    Uses the AF DB prediction API (`/api/prediction/{uid}`) to discover the
    current `pdbUrl`; AF DB has rolled past v4 (now v6 as of 2026), so
    hard-coded version URLs become 404s over time.
    """
    out_path = os.path.join(pdb_dir, f"{af_id(uniprot)}.pdb")
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return True

    api_url = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot}"
    try:
        api = session.get(api_url, timeout=30)
    except requests.RequestException as e:
        msg = f"{af_id(uniprot)},,alphafold_api_error:{type(e).__name__}"
        print(f"[skip] {msg}", file=sys.stderr)
        failed_log_fh.write(msg + "\n")
        return False
    if api.status_code != 200:
        msg = f"{af_id(uniprot)},,alphafold_api_http_{api.status_code}"
        print(f"[skip] {msg}", file=sys.stderr)
        failed_log_fh.write(msg + "\n")
        return False
    try:
        entries = api.json()
    except ValueError:
        msg = f"{af_id(uniprot)},,alphafold_api_bad_json"
        print(f"[skip] {msg}", file=sys.stderr)
        failed_log_fh.write(msg + "\n")
        return False
    if not entries:
        msg = f"{af_id(uniprot)},,alphafold_no_entries"
        print(f"[skip] {msg}", file=sys.stderr)
        failed_log_fh.write(msg + "\n")
        return False
    pdb_url = entries[0].get("pdbUrl")
    if not pdb_url:
        msg = f"{af_id(uniprot)},,alphafold_no_pdbUrl"
        print(f"[skip] {msg}", file=sys.stderr)
        failed_log_fh.write(msg + "\n")
        return False

    try:
        r = session.get(pdb_url, timeout=60, stream=True)
    except requests.RequestException as e:
        msg = f"{af_id(uniprot)},,alphafold_download_error:{type(e).__name__}"
        print(f"[skip] {msg}", file=sys.stderr)
        failed_log_fh.write(msg + "\n")
        return False
    if r.status_code != 200:
        msg = f"{af_id(uniprot)},,alphafold_http_{r.status_code}"
        print(f"[skip] {msg}", file=sys.stderr)
        failed_log_fh.write(msg + "\n")
        return False
    with open(out_path, "wb") as fh:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                fh.write(chunk)
    return True


def main():
    args = parse_args()

    dataset_dir = os.path.abspath(args.dataset_dir)
    pdb_dir = os.path.abspath(args.pdb_dir) if args.pdb_dir else os.path.join(dataset_dir, "raw", "pdb_files")
    info_dir = os.path.join(dataset_dir, "info")
    for d in [pdb_dir, info_dir]:
        os.makedirs(d, exist_ok=True)

    failed_log = os.path.join(info_dir, "processing_failed.csv")
    if not os.path.exists(failed_log):
        with open(failed_log, "w") as f:
            f.write("protein_id,ligand_id,comment\n")

    smiles_map = load_smiles_map(args.raw_dir)

    with open(failed_log, "a") as flog:
        per_split = build_target_ligands(
            args.raw_dir, args.splits, smiles_map,
            args.limit_targets, args.limit_per_target, flog,
        )

    if not per_split:
        print("[error] no usable splits — aborting.", file=sys.stderr)
        sys.exit(1)

    written_uniprots = write_smiles_files(dataset_dir, per_split)
    print(f"Wrote SMILES files for {len(written_uniprots)} targets")

    # Stage AlphaFold structures
    available = set()
    with open(failed_log, "a") as flog:
        if args.no_download:
            for u in sorted(written_uniprots):
                if os.path.exists(os.path.join(pdb_dir, f"{af_id(u)}.pdb")):
                    available.add(u)
                else:
                    msg = f"{af_id(u)},,no_download_and_no_cached_pdb"
                    print(f"[skip] {msg}", file=sys.stderr)
                    flog.write(msg + "\n")
        else:
            session = requests.Session()
            for i, u in enumerate(sorted(written_uniprots)):
                if i % 50 == 0:
                    print(f"[AF download] {i}/{len(written_uniprots)} (ok={len(available)})")
                if fetch_alphafold(u, pdb_dir, session, flog):
                    available.add(u)

    print(f"AlphaFold structures available for {len(available)} / {len(written_uniprots)} targets")
    if not available:
        print("[error] no AlphaFold structures available — aborting.", file=sys.stderr)
        sys.exit(1)

    # Per-split protein_ids files (only those with actives in that split AND a structure available)
    staged_ids = sorted(af_id(u) for u in available)
    protein_ids_path = os.path.join(info_dir, "protein_ids.txt")
    write_list_to_txt(protein_ids_path, staged_ids)
    for split in per_split:
        ids = [af_id(u) for u in per_split[split].keys() if u in available]
        ids.sort()
        if split == "test":
            split_label = "test"
        elif split == "val":
            split_label = "train_val"
        else:
            split_label = "train_train"
        write_list_to_txt(os.path.join(info_dir, f"{split_label}_protein_ids.txt"), ids)
        print(f"  wrote {len(ids)} ids for split={split} ({split_label})")

    if args.skip_processing:
        print("--skip_processing set: not running PDBGraphProcessor/LigandProcessor.")
        return

    print(f"Running PDBGraphProcessor on {dataset_dir} ...")
    graph_processor = PDBGraphProcessor(
        dataset_dir=dataset_dir,
        pdb_dir=pdb_dir,
        overwrite=args.overwrite,
        num_workers=args.num_workers,
        extract_ligands="none",
        select_chains="all",
        labeled_smiles="binary",
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
                # graph filename is "{AF-uniprot}.pt" (extract_ligands="none")
                done_ids.add(fname[:-3])

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
