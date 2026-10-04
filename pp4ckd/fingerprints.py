"""
Molecular Fingerprint Generation & Standardization Engine for PP4CKD.

Supports 7 standardized molecular fingerprint representations:
  1. ECFP4   - Extended Connectivity Fingerprint (Morgan r=2, 4096 bits)
  2. ECFP6   - Extended Connectivity Fingerprint (Morgan r=3, 4096 bits)
  3. AtomPair- Hashed Atom-Pair Fingerprint (4096 bits)
  4. Layered - Layered Topological Path Fingerprint (4096 bits)
  5. RDKit   - RDKit Daylight-like Path Fingerprint (4096 bits)
  6. MHFP6   - MinHash Fingerprint (folded secfp r=3, 4096 bits)
  7. Fused   - Concatenation of ECFP4 (4096) + MHFP6 (4096) -> 8192 bits
"""

import sys
from typing import List, Optional, Tuple, Union
import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.MolStandardize import rdMolStandardize
except ImportError:
    Chem = None
    AllChem = None
    rdMolStandardize = None

try:
    from mhfp.encoder import MHFPEncoder
    _MHFP_AVAILABLE = True
except ImportError:
    MHFPEncoder = None
    _MHFP_AVAILABLE = False


AVAILABLE_FINGERPRINTS = [
    "ECFP4",
    "ECFP6",
    "AtomPair",
    "Layered",
    "RDKit",
    "MHFP6",
    "Fused",
]

FINGERPRINT_DIMS = {
    "ECFP4": 4096,
    "ECFP6": 4096,
    "AtomPair": 4096,
    "Layered": 4096,
    "RDKit": 4096,
    "MHFP6": 4096,
    "Fused": 8192,
}

_global_lfc = None
_global_uncharger = None
_global_mhfp_encoder = None


def _get_standardizers():
    global _global_lfc, _global_uncharger
    if _global_lfc is None and rdMolStandardize is not None:
        _global_lfc = rdMolStandardize.LargestFragmentChooser()
        _global_uncharger = rdMolStandardize.Uncharger()
    return _global_lfc, _global_uncharger


def _get_mhfp_encoder():
    global _global_mhfp_encoder
    if _global_mhfp_encoder is None:
        if not _MHFP_AVAILABLE:
            raise ImportError(
                "mhfp package is not installed. Please install it via `pip install mhfp` "
                "to use MHFP6 or Fused fingerprints."
            )
        _global_mhfp_encoder = MHFPEncoder()
    return _global_mhfp_encoder


def standardize_smiles(smiles: str, max_heavy_atoms: int = 80) -> Optional[str]:
    """
    Standardize a SMILES string according to the PPB3 protocol:
      1. Parse molecule with RDKit.
      2. Choose largest covalent fragment (remove salts/solvents).
      3. Neutralize common charges (Uncharger).
      4. Ensure heavy atom count is between 1 and max_heavy_atoms.
      5. Output canonical non-isomeric SMILES.

    Parameters
    ----------
    smiles : str
        Input SMILES string.
    max_heavy_atoms : int, default=80
        Upper bound on heavy atoms.

    Returns
    -------
    str or None
        Standardized SMILES or None if parsing/standardization fails.
    """
    if Chem is None:
        raise ImportError("rdkit is required for SMILES standardization. Install via conda or pip.")

    if not smiles or not isinstance(smiles, str):
        return None

    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None

        lfc, uncharger = _get_standardizers()
        mol = lfc.choose(mol)
        if mol is None:
            return None

        mol = uncharger.uncharge(mol)
        if mol is None:
            return None

        n_heavy = mol.GetNumHeavyAtoms()
        if n_heavy == 0 or n_heavy >= max_heavy_atoms:
            return None

        clean_smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
        return clean_smi
    except Exception:
        return None


def smiles_to_mol(smiles: str) -> Optional["Chem.Mol"]:
    """Standardize SMILES and return an RDKit Mol object."""
    clean_smi = standardize_smiles(smiles)
    if clean_smi is None:
        return None
    return Chem.MolFromSmiles(clean_smi)


def compute_fingerprint(
    mol_or_smiles: Union[str, "Chem.Mol"],
    fp_type: str = "Fused",
    pack_bits: bool = False,
) -> np.ndarray:
    """
    Compute molecular fingerprint bit vector.

    Parameters
    ----------
    mol_or_smiles : str or rdkit.Chem.Mol
        Input SMILES string or RDKit Mol object.
    fp_type : str, default="Fused"
        Fingerprint type in {"ECFP4", "ECFP6", "AtomPair", "Layered", "RDKit", "MHFP6", "Fused"}.
    pack_bits : bool, default=False
        If True, pack boolean bits into uint8 array (512 bytes for 4096-bit, 1024 bytes for 8192-bit).
        If False, returns float32 array of shape (input_dim,).

    Returns
    -------
    np.ndarray
        Computed fingerprint vector.
    """
    if Chem is None:
        raise ImportError("rdkit is required. Install via conda or pip.")

    fp_key = fp_type.upper()
    valid_keys = {k.upper(): k for k in AVAILABLE_FINGERPRINTS}
    if fp_key not in valid_keys:
        raise ValueError(f"Unsupported fp_type '{fp_type}'. Choose from {AVAILABLE_FINGERPRINTS}")
    canonical_fp = valid_keys[fp_key]

    # Resolve Mol and canonical SMILES
    if isinstance(mol_or_smiles, str):
        clean_smi = standardize_smiles(mol_or_smiles)
        if clean_smi is None:
            dim = FINGERPRINT_DIMS[canonical_fp]
            return np.zeros(dim // 8 if pack_bits else dim, dtype=np.uint8 if pack_bits else np.float32)
        mol = Chem.MolFromSmiles(clean_smi)
    else:
        mol = mol_or_smiles
        clean_smi = Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)

    if canonical_fp == "Fused":
        arr1 = compute_fingerprint(mol, "ECFP4", pack_bits=pack_bits)
        arr2 = compute_fingerprint(clean_smi, "MHFP6", pack_bits=pack_bits)
        return np.concatenate([arr1, arr2])

    target_dim = 4096

    if canonical_fp == "ECFP4":
        if hasattr(AllChem, "GetMorganFingerprintAsBitVect"):
            # Suppress warning or use modern generator if present
            try:
                from rdkit.Chem import rdFingerprintGenerator
                gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=target_dim)
                bv = gen.GetFingerprint(mol)
            except (ImportError, AttributeError):
                bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=target_dim)
        else:
            bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=target_dim)
    elif canonical_fp == "ECFP6":
        try:
            from rdkit.Chem import rdFingerprintGenerator
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=target_dim)
            bv = gen.GetFingerprint(mol)
        except (ImportError, AttributeError):
            bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius=3, nBits=target_dim)
    elif canonical_fp == "AtomPair":
        try:
            from rdkit.Chem import rdFingerprintGenerator
            gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=target_dim)
            bv = gen.GetFingerprint(mol)
        except (ImportError, AttributeError):
            bv = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=target_dim)
    elif canonical_fp == "Layered":
        bv = Chem.RDKFingerprint(mol, minPath=1, maxPath=7, fpSize=target_dim, nBitsPerHash=2, useHs=True)
    elif canonical_fp == "RDKit":
        try:
            from rdkit.Chem import rdFingerprintGenerator
            gen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=target_dim)
            bv = gen.GetFingerprint(mol)
        except (ImportError, AttributeError):
            bv = Chem.RDKFingerprint(mol, fpSize=target_dim)
    elif canonical_fp == "MHFP6":
        encoder = _get_mhfp_encoder()
        secfp = encoder.secfp_from_smiles(clean_smi, length=target_dim, radius=3, rings=True, kekulize=True, sanitize=False)
        if pack_bits:
            # pack boolean list of length 4096 into 512 uint8
            return np.packbits(np.array(secfp, dtype=np.uint8))
        else:
            return np.array(secfp, dtype=np.float32)
    else:
        raise ValueError(f"Unknown fingerprint: {canonical_fp}")

    if pack_bits:
        return np.packbits(bv)
    else:
        arr = np.zeros((target_dim,), dtype=np.float32)
        for idx in bv.GetOnBits():
            arr[idx] = 1.0
        return arr


def batch_compute_fingerprints(
    smiles_list: List[str],
    fp_type: str = "Fused",
    pack_bits: bool = True,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Compute fingerprints for a batch of SMILES strings.

    Parameters
    ----------
    smiles_list : list of str
        List of chemical SMILES strings.
    fp_type : str, default="Fused"
        Fingerprint type.
    pack_bits : bool, default=True
        Whether to pack bits into uint8 bytes.
    n_jobs : int, default=1
        Number of CPU workers (if > 1, uses multiprocessing).

    Returns
    -------
    np.ndarray
        Array of shape (N, dim) or (N, dim // 8).
    """
    if n_jobs == 1:
        results = [compute_fingerprint(s, fp_type=fp_type, pack_bits=pack_bits) for s in smiles_list]
        return np.array(results)
    else:
        import multiprocessing as mp
        with mp.Pool(processes=n_jobs) as pool:
            results = pool.starmap(
                compute_fingerprint,
                [(s, fp_type, pack_bits) for s in smiles_list],
                chunksize=1000,
            )
        return np.array(results)
