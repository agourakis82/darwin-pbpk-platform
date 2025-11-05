"""
KEC Topological Features for Molecular Graphs
==============================================

Computes 10 KEC-inspired topological features:
- Entropy: Molecular complexity, flexibility
- Curvature: Rigidity, geometry
- Coherence: Connectivity, symmetry

Author: Dr. Agourakis
Date: October 26, 2025
"""

import numpy as np
from typing import List, Union
from rdkit import Chem
from rdkit.Chem import Descriptors, GraphDescriptors


def compute_kec_features(mol: Chem.Mol) -> np.ndarray:
    """
    Compute 10 KEC-inspired topological features for a molecule.
    
    Features:
        1. Bertz Complexity (Entropy - molecular complexity)
        2. Num Rotatable Bonds (Entropy - flexibility)
        3. Num Aromatic Rings (Curvature - rigidity)
        4. Fraction Csp3 (Curvature - saturation)
        5. Balaban J (Coherence - connectivity)
        6. Chi0 (Coherence - symmetry)
        7. MolLogP (PK-relevant - lipophilicity)
        8. TPSA (PK-relevant - polar surface area)
        9. Num H Donors (PK-relevant)
        10. Num H Acceptors (PK-relevant)
    
    Args:
        mol: RDKit molecule object
    
    Returns:
        Array of shape [10] with features
    """
    features = []
    
    try:
        # Entropy-related (molecular complexity)
        features.append(Descriptors.BertzCT(mol))
        features.append(Descriptors.NumRotatableBonds(mol))
        
        # Curvature-related (geometry, strain)
        features.append(Descriptors.NumAromaticRings(mol))
        features.append(Descriptors.FractionCSP3(mol))
        
        # Coherence-related (connectivity, symmetry)
        features.append(GraphDescriptors.BalabanJ(mol))
        features.append(GraphDescriptors.Chi0(mol))
        
        # PK-relevant properties
        features.append(Descriptors.MolLogP(mol))
        features.append(Descriptors.TPSA(mol))
        features.append(Descriptors.NumHDonors(mol))
        features.append(Descriptors.NumHAcceptors(mol))
        
    except Exception as e:
        # If any descriptor fails, return zeros
        print(f"Warning: Failed to compute descriptors for molecule: {e}")
        features = [0.0] * 10
    
    return np.array(features, dtype=np.float32)


def compute_kec_features_batch(mol_list: List[Chem.Mol]) -> np.ndarray:
    """
    Compute KEC features for a batch of molecules.
    
    Args:
        mol_list: List of RDKit molecule objects
    
    Returns:
        Array of shape [N, 10] with features for all molecules
    """
    features_list = []
    
    for mol in mol_list:
        features = compute_kec_features(mol)
        features_list.append(features)
    
    return np.vstack(features_list)


def compute_kec_features_from_smiles(smiles: Union[str, List[str]]) -> np.ndarray:
    """
    Compute KEC features directly from SMILES string(s).
    
    Args:
        smiles: SMILES string or list of SMILES strings
    
    Returns:
        Array of shape [10] (single SMILES) or [N, 10] (list of SMILES)
    """
    if isinstance(smiles, str):
        # Single SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(10, dtype=np.float32)
        return compute_kec_features(mol)
    
    else:
        # List of SMILES
        mol_list = [Chem.MolFromSmiles(s) for s in smiles]
        # Replace None with empty mol (will return zeros)
        mol_list = [mol if mol is not None else Chem.MolFromSmiles('C') for mol in mol_list]
        return compute_kec_features_batch(mol_list)


# Feature names for interpretability
KEC_FEATURE_NAMES = [
    "BertzCT",           # Complexity (Entropy)
    "NumRotatableBonds", # Flexibility (Entropy)
    "NumAromaticRings",  # Rigidity (Curvature)
    "FractionCsp3",      # Saturation (Curvature)
    "BalabanJ",          # Connectivity (Coherence)
    "Chi0",              # Symmetry (Coherence)
    "MolLogP",           # Lipophilicity
    "TPSA",              # Polar Surface Area
    "NumHDonors",        # H-bond donors
    "NumHAcceptors"      # H-bond acceptors
]

