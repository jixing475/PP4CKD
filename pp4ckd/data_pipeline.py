"""
ChEMBL Data Extraction & PPB3 Cleaning Pipeline for PP4CKD.

Extracts bioactivity data from ChEMBL SQLite database and constructs:
  - Cleaned, non-isomeric desalted SMILES catalog (heavy atoms < 80)
  - Top 95 activity types thresholded at <= 10 uM or >= 50% inhibition
  - Targets filtered to >= 5 active molecules (7,676 targets in ChEMBL 36)
  - CKDdb renal disease target annotations (623 targets)
  - Sparse CSR molecule-target interaction matrix
"""

import os
import sys
import time
import json
import sqlite3
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import multiprocessing as mp
import numpy as np
import pandas as pd
from scipy import sparse

from .fingerprints import standardize_smiles


def init_worker():
    """Initializer for multiprocessing pool workers."""
    pass


def _clean_worker(task: Tuple[int, str, str]) -> Tuple[int, str, Optional[str]]:
    molregno, chembl_id, raw_smi = task
    clean_smi = standardize_smiles(raw_smi, max_heavy_atoms=80)
    return molregno, chembl_id, clean_smi


def run_pipeline(
    db_path: str,
    out_dir: str,
    ckd_csv_path: Optional[str] = None,
    n_cpus: int = 16,
) -> Dict[str, str]:
    """
    Execute full data extraction and cleaning pipeline.

    Parameters
    ----------
    db_path : str
        Path to ChEMBL SQLite database file (e.g., chembl_36.db).
    out_dir : str
        Directory to save cleaned artifacts.
    ckd_csv_path : str, optional
        Path to reference ckd_623_targets.csv if available.
    n_cpus : int, default=16
        Number of CPU processes for parallel standardization.

    Returns
    -------
    dict
        Paths to created artifacts.
    """
    start_time = time.time()
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PP4CKD ChEMBL Data Extraction & Standardization Pipeline")
    print(f"Database: {db_path}")
    print(f"Output:   {out_dir}")
    print("=" * 70)

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    cur = conn.cursor()

    # Step 1: Top 95 activity types
    print("\n[Step 1/6] Extracting Top 95 Activity Types...")
    cur.execute("""
        SELECT standard_type 
        FROM activities 
        WHERE standard_type IS NOT NULL 
        GROUP BY standard_type 
        ORDER BY count(*) DESC 
        LIMIT 95
    """)
    top95_types = [r[0] for r in cur.fetchall()]
    print(f"  Extracted {len(top95_types)} types (Top 5: {top95_types[:5]})")
    with open(out_path / "top95_activity_types.json", "w") as f:
        json.dump(top95_types, f, indent=2)

    # Step 2: Query active pairs
    print("\n[Step 2/6] Querying active interactions from ChEMBL SQLite...")
    placeholders = ",".join(["?"] * len(top95_types))
    query_sql = f"""
    SELECT DISTINCT act.molregno, md.chembl_id as mol_chembl_id, cs.canonical_smiles, a.tid
    FROM activities act
    JOIN assays a ON act.assay_id = a.assay_id
    JOIN compound_properties cp ON act.molregno = cp.molregno
    JOIN compound_structures cs ON act.molregno = cs.molregno
    JOIN molecule_dictionary md ON act.molregno = md.molregno
    WHERE act.standard_type IN ({placeholders})
      AND cp.heavy_atoms < 80
      AND cp.heavy_atoms > 0
      AND (
        (act.pchembl_value >= 5.0 AND (act.standard_relation IN ('=', '<', '<=', '~') OR act.standard_relation IS NULL))
        OR (act.standard_units = 'nM' AND act.standard_value <= 10000 AND act.standard_value > 0 AND act.standard_relation IN ('=', '<', '<=', '~'))
        OR (act.standard_units = 'uM' AND act.standard_value <= 10 AND act.standard_value > 0 AND act.standard_relation IN ('=', '<', '<=', '~'))
        OR (act.standard_units = '%' AND act.standard_value >= 50 AND (act.standard_relation IN ('=', '>', '>=', '~') OR act.standard_relation IS NULL))
      )
      AND a.tid IS NOT NULL
      AND cs.canonical_smiles IS NOT NULL
    """
    t0 = time.time()
    cur.execute(query_sql, top95_types)
    rows = cur.fetchall()
    print(f"  Extracted {len(rows):,} raw active pairs in {time.time()-t0:.2f}s")

    # Step 3: Parallel SMILES standardization
    print("\n[Step 3/6] Parallel SMILES standardization...")
    unique_mols = {}
    for molregno, mol_chembl_id, raw_smi, tid in rows:
        if molregno not in unique_mols:
            unique_mols[molregno] = (mol_chembl_id, raw_smi)

    tasks = [(molregno, info[0], info[1]) for molregno, info in unique_mols.items()]
    t0 = time.time()
    cpus = min(n_cpus, mp.cpu_count())
    with mp.Pool(processes=cpus, initializer=init_worker) as pool:
        cleaned_results = pool.map(_clean_worker, tasks, chunksize=2000)

    cleaned_smi_dict = {}
    valid_count = 0
    for molregno, chembl_id, clean_smi in cleaned_results:
        if clean_smi:
            cleaned_smi_dict[molregno] = clean_smi
            valid_count += 1
    print(f"  Standardized {valid_count:,}/{len(unique_mols):,} valid molecules in {time.time()-t0:.2f}s")

    # Deduplicate standardized structures
    smi_to_clean_id = {}
    clean_id_to_smi = []
    molregno_to_clean_id = {}
    for molregno, clean_smi in cleaned_smi_dict.items():
        if clean_smi not in smi_to_clean_id:
            cid = len(clean_id_to_smi)
            smi_to_clean_id[clean_smi] = cid
            clean_id_to_smi.append(clean_smi)
        molregno_to_clean_id[molregno] = smi_to_clean_id[clean_smi]

    # Step 4: Target filtering (>= 5 active compounds)
    print("\n[Step 4/6] Target filtering (>= 5 active molecules)...")
    clean_pairs = set()
    for molregno, mol_chembl_id, raw_smi, tid in rows:
        if molregno in molregno_to_clean_id:
            cid = molregno_to_clean_id[molregno]
            clean_pairs.add((cid, tid))

    target_mol_counts = {}
    for cid, tid in clean_pairs:
        target_mol_counts[tid] = target_mol_counts.get(tid, 0) + 1

    filtered_tids = {tid for tid, cnt in target_mol_counts.items() if cnt >= 5}
    final_pairs = [(cid, tid) for cid, tid in clean_pairs if tid in filtered_tids]

    active_mols_set = {cid for cid, tid in final_pairs}
    old_cid_to_new_mid = {}
    final_smiles_list = []
    for old_cid in sorted(active_mols_set):
        new_mid = len(final_smiles_list)
        old_cid_to_new_mid[old_cid] = new_mid
        final_smiles_list.append(clean_id_to_smi[old_cid])

    # Step 5: Target metadata and CKD subset identification
    print("\n[Step 5/6] Fetching target metadata & identifying CKDdb subset...")
    tids_list = sorted(list(filtered_tids))
    tid_placeholders = ",".join(["?"] * len(tids_list))
    cur.execute(f"""
        SELECT tid, chembl_id, pref_name, target_type, organism
        FROM target_dictionary
        WHERE tid IN ({tid_placeholders})
    """, tids_list)
    target_meta_rows = cur.fetchall()
    target_meta_dict = {
        r[0]: {"chembl_id": r[1], "pref_name": r[2], "target_type": r[3], "organism": r[4]}
        for r in target_meta_rows
    }

    # Identify CKD subset
    ckd_chembl_set = set()
    if ckd_csv_path and os.path.exists(ckd_csv_path):
        ckd_ref = pd.read_csv(ckd_csv_path)
        ckd_chembl_set = set(ckd_ref["chembl_id"].dropna().unique())
        print(f"  Loaded {len(ckd_chembl_set)} reference CKD ChEMBL IDs from {ckd_csv_path}")

    # Build target mapping
    target_mapping = []
    ckd_target_indices = []
    tid_to_target_idx = {}

    for tid in tids_list:
        idx = len(target_mapping)
        tid_to_target_idx[tid] = idx
        meta = target_meta_dict.get(tid, {})
        ch_id = meta.get("chembl_id", f"CHEMBL_TID_{tid}")
        is_ckd = ch_id in ckd_chembl_set
        if is_ckd:
            ckd_target_indices.append(idx)

        target_mapping.append({
            "target_idx": idx,
            "tid": tid,
            "chembl_id": ch_id,
            "pref_name": meta.get("pref_name", ""),
            "target_type": meta.get("target_type", ""),
            "organism": meta.get("organism", ""),
            "active_molecules": target_mol_counts[tid],
            "is_ckd": is_ckd,
        })

    # Step 6: Construct CSR matrix
    print("\n[Step 6/6] Generating CSR interaction matrix...")
    row_indices = [old_cid_to_new_mid[cid] for cid, tid in final_pairs]
    col_indices = [tid_to_target_idx[tid] for cid, tid in final_pairs]
    data_arr = np.ones(len(row_indices), dtype=np.uint8)

    n_mols = len(final_smiles_list)
    n_targets = len(target_mapping)
    csr = sparse.csr_matrix(
        (data_arr, (row_indices, col_indices)),
        shape=(n_mols, n_targets),
        dtype=np.uint8,
    )

    # Persist artifacts
    csr_file = out_path / "mol_target_csr.npz"
    sparse.save_npz(csr_file, csr)

    target_df = pd.DataFrame(target_mapping)
    tsv_file = out_path / "target_labels.tsv"
    target_df.to_csv(tsv_file, sep="\t", index=False)

    np.save(out_path / "ckd_target_indices.npy", np.array(ckd_target_indices, dtype=np.int32))

    smi_file = out_path / "cleaned_smiles.txt"
    with open(smi_file, "w") as f:
        for s in final_smiles_list:
            f.write(s + "\n")

    conn.close()
    elapsed = time.time() - start_time
    print(f"\nPipeline finished in {elapsed:.1f}s ({elapsed/60:.2f} min).")
    print(f"Retained: {n_mols:,} molecules, {n_targets:,} targets, {csr.nnz:,} interactions.")

    return {
        "csr_file": str(csr_file),
        "target_labels": str(tsv_file),
        "smiles_file": str(smi_file),
    }


def main():
    parser = argparse.ArgumentParser(description="ChEMBL Data Extraction Pipeline for PP4CKD")
    parser.add_argument("--db-path", type=str, required=True, help="Path to chembl_36.db SQLite database")
    parser.add_argument("--out-dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--ckd-csv", type=str, default=None, help="Reference ckd_623_targets.csv")
    parser.add_argument("--cpus", type=int, default=16, help="CPU cores for parallel processing")
    args = parser.parse_args()

    run_pipeline(
        db_path=args.db_path,
        out_dir=args.out_dir,
        ckd_csv_path=args.ckd_csv,
        n_cpus=args.cpus,
    )


if __name__ == "__main__":
    main()
