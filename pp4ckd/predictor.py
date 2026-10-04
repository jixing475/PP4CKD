"""
Inference & Prediction Interface for PP4CKD.

Provides unified prediction API for chemical SMILES:
  - Global Top-N target ranking across 7,676 ChEMBL 36 targets
  - Disease-specific CKDdb Top-N ranking across 623 renal disease targets
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np
import pandas as pd
import torch

from .model import PPB3Net
from .fingerprints import (
    compute_fingerprint,
    standardize_smiles,
    AVAILABLE_FINGERPRINTS,
    FINGERPRINT_DIMS,
)

# Default data paths relative to package root
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_DIR = _PACKAGE_ROOT / "data"
_DEFAULT_MODELS_DIR = _PACKAGE_ROOT / "models"


class PP4CKDPredictor:
    """
    PP4CKD Multi-Target Predictor.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to PyTorch .pt model file. If None, looks for fused_full_model.pt
        or ecfp4_full_model.pt in package models directory.
    target_labels_path : str or Path, optional
        Path to target_labels.tsv. If None, uses bundled data.
    ckd_targets_path : str or Path, optional
        Path to ckd_623_targets.csv. If None, uses bundled data.
    fp_type : str, default="Fused"
        Fingerprint representation ("Fused", "ECFP4", "MHFP6", etc.).
    device : str or torch.device, optional
        Computation device ("cpu", "cuda", "mps"). Auto-detects by default.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        target_labels_path: Optional[Union[str, Path]] = None,
        ckd_targets_path: Optional[Union[str, Path]] = None,
        fp_type: str = "Fused",
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.fp_type = fp_type

        # 1. Determine device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # 2. Resolve target metadata
        target_labels_file = (
            Path(target_labels_path)
            if target_labels_path
            else _DEFAULT_DATA_DIR / "target_labels.tsv"
        )
        if not target_labels_file.exists():
            raise FileNotFoundError(
                f"Target metadata not found at: {target_labels_file}. "
                "Please verify data directory."
            )

        self.targets_df = pd.read_csv(target_labels_file, sep="\t")
        self.n_targets = len(self.targets_df)

        # 3. Resolve CKD target annotations
        ckd_file = (
            Path(ckd_targets_path)
            if ckd_targets_path
            else _DEFAULT_DATA_DIR / "ckd_623_targets.csv"
        )
        if ckd_file.exists():
            self.ckd_meta_df = pd.read_csv(ckd_file)
            ckd_chembl_ids = set(self.ckd_meta_df["chembl_id"].dropna())
            # Ensure is_ckd column in targets_df matches
            if "is_ckd" not in self.targets_df.columns:
                self.targets_df["is_ckd"] = self.targets_df["chembl_id"].isin(ckd_chembl_ids)
            else:
                # Fill or combine
                self.targets_df["is_ckd"] = self.targets_df["is_ckd"].astype(bool) | self.targets_df["chembl_id"].isin(ckd_chembl_ids)
        else:
            self.ckd_meta_df = None
            if "is_ckd" not in self.targets_df.columns:
                self.targets_df["is_ckd"] = False

        self.ckd_indices = np.where(self.targets_df["is_ckd"].values)[0]

        # 4. Resolve Model Checkpoint
        resolved_model_path = None
        if model_path is not None:
            candidate = Path(model_path)
            if candidate.exists():
                resolved_model_path = candidate
        else:
            preferred = _DEFAULT_MODELS_DIR / f"{self.fp_type.lower()}_full_model.pt"
            fallback_fused = _DEFAULT_MODELS_DIR / "fused_full_model.pt"
            fallback_ecfp4 = _DEFAULT_MODELS_DIR / "ecfp4_full_model.pt"

            if preferred.exists():
                resolved_model_path = preferred
            elif fallback_fused.exists():
                resolved_model_path = fallback_fused
                self.fp_type = "Fused"
            elif fallback_ecfp4.exists():
                resolved_model_path = fallback_ecfp4
                self.fp_type = "ECFP4"

        self.model_path = resolved_model_path
        self.input_dim = FINGERPRINT_DIMS.get(self.fp_type, 4096)

        if resolved_model_path is not None and resolved_model_path.exists():
            self.model = PPB3Net.load_from_checkpoint(
                resolved_model_path,
                input_dim=self.input_dim,
                output_dim=self.n_targets,
                device=self.device,
            )
        else:
            # Model weight not present yet: instantiate uninitialized model for structural operations
            self.model = PPB3Net(input_dim=self.input_dim, output_dim=self.n_targets)
            self.model.to(self.device)
            self.model.eval()

    def predict_probabilities(self, smiles: str) -> np.ndarray:
        """
        Calculate raw target probabilities for a single SMILES.

        Parameters
        ----------
        smiles : str
            Input SMILES.

        Returns
        -------
        np.ndarray
            Array of shape (n_targets,) with sigmoid probabilities.
        """
        clean_smi = standardize_smiles(smiles)
        if clean_smi is None:
            raise ValueError(f"Failed to parse or standardize SMILES: {smiles}")

        fp_vec = compute_fingerprint(clean_smi, fp_type=self.fp_type, pack_bits=False)
        x = torch.from_numpy(fp_vec).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = self.model.predict_proba(x).squeeze(0).cpu().numpy()

        return probs

    def predict(
        self,
        smiles: str,
        top_k: int = 10,
        ckd_only: bool = False,
    ) -> pd.DataFrame:
        """
        Predict top-K targets for a query molecule.

        Parameters
        ----------
        smiles : str
            Input SMILES string.
        top_k : int, default=10
            Number of top predicted targets to return.
        ckd_only : bool, default=False
            If True, restricts ranking strictly within the 623 CKDdb renal disease targets.

        Returns
        -------
        pd.DataFrame
            Ranked target predictions with columns:
            ['Rank', 'Target Name', 'ChEMBL ID', 'Probability', 'Target Type', 'Organism', 'CKDdb Target']
        """
        probs = self.predict_probabilities(smiles)

        if ckd_only:
            candidate_indices = self.ckd_indices
            sub_probs = probs[candidate_indices]
            top_sub_order = np.argsort(-sub_probs)[:top_k]
            top_indices = candidate_indices[top_sub_order]
        else:
            top_indices = np.argsort(-probs)[:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            row = self.targets_df.iloc[idx]
            results.append({
                "Rank": rank,
                "Target Name": row.get("pref_name", ""),
                "ChEMBL ID": row.get("chembl_id", ""),
                "Probability": float(probs[idx]),
                "Target Type": row.get("target_type", ""),
                "Organism": row.get("organism", ""),
                "CKDdb Target": bool(row.get("is_ckd", False)),
            })

        return pd.DataFrame(results)

    def predict_dual(self, smiles: str, top_k: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Run simultaneous Global Top-K and CKDdb Specific Top-K prediction.

        Parameters
        ----------
        smiles : str
            Input chemical SMILES.
        top_k : int, default=10
            Number of ranks to return for each view.

        Returns
        -------
        dict
            {"global": pd.DataFrame, "ckd": pd.DataFrame}
        """
        global_df = self.predict(smiles, top_k=top_k, ckd_only=False)
        ckd_df = self.predict(smiles, top_k=top_k, ckd_only=True)
        return {
            "global": global_df,
            "ckd": ckd_df,
        }


def main():
    parser = argparse.ArgumentParser(
        description="PP4CKD: Polypharmacology Target Predictor for Chronic Kidney Disease"
    )
    parser.add_argument(
        "--smiles",
        "-s",
        type=str,
        default="CC(=O)Oc1ccccc1C(=O)O",
        help="Input query SMILES (default: Aspirin)",
    )
    parser.add_argument(
        "--fp",
        type=str,
        default="Fused",
        choices=AVAILABLE_FINGERPRINTS,
        help="Fingerprint representation (default: Fused)",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=10,
        help="Number of top targets to return (default: 10)",
    )
    parser.add_argument(
        "--ckd-only",
        action="store_true",
        help="Filter predictions exclusively to CKDdb 623 renal disease targets",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to custom model checkpoint (.pt)",
    )
    args = parser.parse_args()

    predictor = PP4CKDPredictor(model_path=args.model, fp_type=args.fp)
    print("=" * 78)
    print(f"PP4CKD Target Prediction [Fingerprint: {args.fp} | Device: {predictor.device}]")
    print(f"Query SMILES: {args.smiles}")
    print("=" * 78)

    if args.ckd_only:
        print(f"\n[CKDdb Renal Targets (623 Targets) - Top {args.top_k}]")
        df = predictor.predict(args.smiles, top_k=args.top_k, ckd_only=True)
        print(df.to_string(index=False))
    else:
        dual = predictor.predict_dual(args.smiles, top_k=args.top_k)
        print(f"\n[Global Prediction - Top {args.top_k} of {predictor.n_targets} ChEMBL Targets]")
        print(dual["global"].to_string(index=False))
        print(f"\n[CKDdb Specific Prediction - Top {args.top_k} Renal Disease Targets]")
        print(dual["ckd"].to_string(index=False))

    print("\n" + "=" * 78)


if __name__ == "__main__":
    main()
