"""
PP4CKD Quickstart Example: Minimal Reproducible Inference.

Demonstrates multi-target prediction for two representative drugs:
  1. Aspirin (Acetylsalicylic acid): Anti-inflammatory / antiplatelet
  2. Dapagliflozin (Farxiga): SGLT2 inhibitor for Chronic Kidney Disease & T2D
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pp4ckd import PP4CKDPredictor


def run_quickstart():
    print("=" * 82)
    print("PP4CKD: Polypharmacology Predictor for Chronic Kidney Disease")
    print("Minimal Reproducible Quickstart Demo")
    print("=" * 82)

    # Instantiate predictor
    # Automatically resolves model weights and bundled CKDdb annotations
    predictor = PP4CKDPredictor()
    print(f"\n[Environment] Device: {predictor.device} | Fingerprint: {predictor.fp_type}")
    print(f"[Metadata] Loaded {predictor.n_targets:,} total ChEMBL targets ({len(predictor.ckd_indices)} CKDdb renal targets)")
    if predictor.model_path:
        print(f"[Model] Checkpoint: {predictor.model_path.name}")
    else:
        print("[Model] Using default neural architecture")

    # Test cases: Aspirin and Dapagliflozin
    test_compounds = [
        {
            "name": "Aspirin",
            "smiles": "CC(=O)Oc1ccccc1C(=O)O",
            "clinical_role": "COX inhibitor; anti-inflammatory and cardio-renal protective agent",
        },
        {
            "name": "Dapagliflozin",
            "smiles": "CCc1ccc(Cc2cc(Cl)c(C3OC(CO)C(O)C(O)C3O)cc2)cc1",
            "clinical_role": "SGLT2 inhibitor; landmark guideline-directed therapy for CKD",
        },
    ]

    for compound in test_compounds:
        name = compound["name"]
        smiles = compound["smiles"]
        role = compound["clinical_role"]

        print("\n" + "#" * 82)
        print(f"Compound: {name}")
        print(f"SMILES:   {smiles}")
        print(f"Role:     {role}")
        print("#" * 82)

        predictions = predictor.predict_dual(smiles, top_k=5)

        print("\n--> [Global Top-5 Predicted Targets (across 7,676 ChEMBL targets)]:")
        print(predictions["global"].to_string(index=False))

        print("\n--> [CKDdb Specific Top-5 Predicted Targets (across 623 Renal Disease targets)]:")
        print(predictions["ckd"].to_string(index=False))

    print("\n" + "=" * 82)
    print("Quickstart execution finished successfully!")
    print("=" * 82)


if __name__ == "__main__":
    run_quickstart()
