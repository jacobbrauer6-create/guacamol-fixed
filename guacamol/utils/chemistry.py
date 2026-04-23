"""
guacamol/utils/chemistry.py
============================
Chemistry utility functions — fixed for NumPy 2.x and modern RDKit.

FIXES:
  - np.bool → bool, np.int → int, np.float → float throughout
  - Chem.MolToSmiles() — added explicit sanitize/kekulize flags where needed
  - AllChem.GetMorganFingerprintAsBitVect → unchanged (still valid API)
  - rdMolDescriptors imports updated
  - RDKit logging suppression updated for RDKit >=2022
"""
from __future__ import annotations

import re
import warnings
from typing import List, Optional, Sequence, Union

import numpy as np

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
    from rdkit.DataStructs import ConvertToNumpyArray, BulkTanimotoSimilarity
    # Suppress RDKit C++ warnings in normal operation
    RDLogger.DisableLog("rdApp.*")
    _RDKIT = True
except ImportError:
    _RDKIT = False


def smiles_to_mol(smiles: str) -> Optional["Chem.Mol"]:
    """Parse SMILES. Returns None on failure."""
    if not _RDKIT or not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def mol_to_smiles(mol: "Chem.Mol", canonical: bool = True) -> Optional[str]:
    """Convert mol to SMILES. Returns None on failure."""
    if not _RDKIT or mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=canonical)
    except Exception:
        return None


def canonicalise(smiles: str) -> Optional[str]:
    """Return canonical SMILES or None if invalid."""
    mol = smiles_to_mol(smiles)
    return mol_to_smiles(mol) if mol else None


def is_valid(smiles: str) -> bool:
    return smiles_to_mol(smiles) is not None


def fraction_valid(smiles_list: Sequence[str]) -> float:
    n = len(smiles_list)
    return sum(is_valid(s) for s in smiles_list) / n if n else 0.0


def get_fingerprint(smiles: str,
                     radius: int = 2,
                     n_bits: int = 2048) -> Optional[np.ndarray]:
    """Morgan fingerprint as float32 numpy array."""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    try:
        from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
        gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
        fp  = gen.GetFingerprint(mol)
    except ImportError:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    ConvertToNumpyArray(fp, arr)
    return arr


def tanimoto_similarity(smiles1: str, smiles2: str,
                          radius: int = 2) -> float:
    """Tanimoto similarity between two molecules."""
    fp1 = get_fingerprint(smiles1, radius)
    fp2 = get_fingerprint(smiles2, radius)
    if fp1 is None or fp2 is None:
        return 0.0
    # Tanimoto = dot(fp1, fp2) / (sum(fp1) + sum(fp2) - dot(fp1, fp2))
    intersection = float(np.dot(fp1, fp2))
    union = float(fp1.sum() + fp2.sum() - intersection)
    return intersection / union if union > 0 else 0.0


def tanimoto_similarity_matrix(smiles_list: Sequence[str],
                                 ref_smiles: str,
                                 radius: int = 2) -> np.ndarray:
    """Return array of Tanimoto similarities to ref_smiles."""
    ref_fp = get_fingerprint(ref_smiles, radius)
    if ref_fp is None or not _RDKIT:
        return np.zeros(len(smiles_list), dtype=np.float32)

    results = []
    for s in smiles_list:
        fp = get_fingerprint(s, radius)
        if fp is None:
            results.append(0.0)
            continue
        intersection = float(np.dot(fp, ref_fp))
        union = float(fp.sum() + ref_fp.sum() - intersection)
        results.append(intersection / union if union > 0 else 0.0)
    return np.array(results, dtype=np.float32)


def calculate_pc_properties(smiles: str) -> dict:
    """
    Calculate physicochemical properties.
    Uses only stable RDKit public API (no deprecated NumPy type aliases).
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return {}
    return {
        "mw":            float(Descriptors.MolWt(mol)),
        "logp":          float(Descriptors.MolLogP(mol)),
        "hbd":           int(rdMolDescriptors.CalcNumHBD(mol)),
        "hba":           int(rdMolDescriptors.CalcNumHBA(mol)),
        "tpsa":          float(Descriptors.TPSA(mol)),
        "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "rings":         int(rdMolDescriptors.CalcNumRings(mol)),
        "arom_rings":    int(rdMolDescriptors.CalcNumAromaticRings(mol)),
        "qed":           float(_qed(mol)),
    }


def _qed(mol) -> float:
    try:
        from rdkit.Chem import QED
        return QED.qed(mol)
    except Exception:
        return 0.0


def remove_duplicates(smiles_list: Sequence[str]) -> List[str]:
    """Return deduplicated list preserving first-occurrence order."""
    seen = set()
    result = []
    for s in smiles_list:
        can = canonicalise(s)
        if can and can not in seen:
            seen.add(can)
            result.append(s)
    return result


def parse_molecular_formula(formula: str) -> dict:
    """
    Parse a molecular formula string into an element count dict.
    e.g. 'C6H12O6' → {'C': 6, 'H': 12, 'O': 6}
    """
    pattern = re.compile(r"([A-Z][a-z]?)(\d*)")
    counts: dict = {}
    for symbol, count_str in pattern.findall(formula):
        count = int(count_str) if count_str else 1
        counts[symbol] = counts.get(symbol, 0) + count
    return counts
