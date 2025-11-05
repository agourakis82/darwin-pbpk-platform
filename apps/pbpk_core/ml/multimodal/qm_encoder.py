"""
Quantum Mechanics (QM) Descriptor Encoder - REAL IMPLEMENTATION
=================================================================

Gera descritores quânticos usando RDKit (cargas parciais, orbitais).

Pipeline Atual (Week 1-7):
- Cargas parciais Gasteiger
- Propriedades eletrônicas RDKit
- Aproximações HOMO/LUMO

Pipeline Futuro (Week 8+):
- Cálculos DFT reais com Psi4
- HOMO/LUMO exatos
- Polarizabilidade, dipolo, etc.

Autor: Dr. Demetrios Chiuratto Agourakis
Data: Outubro 2025
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors, Crippen
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class QMDescriptorEncoder:
    """
    Encoder QM para moléculas.
    
    FASE 1 (Atual): RDKit-based approximations
    FASE 2 (Week 8): Psi4 DFT calculations
    
    Descritores (15):
    - Cargas parciais (4): max, min, mean, std (Gasteiger)
    - Propriedades eletrônicas (5): 
      * TPSA (topological polar surface area)
      * Número de elétrons de valência
      * Número de elétrons radicalares
      * Aromatic fraction
      * Unsaturation degree
    - Orbitais aproximados (3):
      * HOMO proxy (via ionization energy approx)
      * LUMO proxy (via electron affinity approx)
      * HOMO-LUMO gap proxy
    - Reatividade (3):
      * Eletronegatividade (pauling)
      * Nucleofilicidade (proxy)
      * Eletrofilicidade (proxy)
    """
    
    def __init__(self, embedding_dim: int = 15):
        self.embedding_dim = embedding_dim
        self.has_psi4 = False  # Will be True em Week 8
        
        logger.info("✅ QMDescriptorEncoder inicializado")
        logger.info(f"   Dimensão: {embedding_dim}")
        logger.info("   Método: RDKit approximations (Psi4 DFT em Week 8)")
    
    def calculate_partial_charges(self, mol: Chem.Mol) -> dict:
        """
        Calcula cargas parciais usando método Gasteiger.
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            Dict com estatísticas das cargas
        """
        try:
            # Calcular cargas Gasteiger
            AllChem.ComputeGasteigerCharges(mol)
            
            charges = []
            for atom in mol.GetAtoms():
                try:
                    charge = atom.GetDoubleProp('_GasteigerCharge')
                    if not np.isnan(charge) and not np.isinf(charge):
                        charges.append(charge)
                except:
                    pass
            
            if len(charges) == 0:
                return {'max': 0.0, 'min': 0.0, 'mean': 0.0, 'std': 0.0}
            
            return {
                'max': float(np.max(charges)),
                'min': float(np.min(charges)),
                'mean': float(np.mean(charges)),
                'std': float(np.std(charges))
            }
            
        except Exception as e:
            logger.debug(f"Erro ao calcular cargas: {e}")
            return {'max': 0.0, 'min': 0.0, 'mean': 0.0, 'std': 0.0}
    
    def calculate_electronic_properties(self, mol: Chem.Mol) -> dict:
        """
        Calcula propriedades eletrônicas básicas.
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            Dict com propriedades eletrônicas
        """
        try:
            props = {}
            
            # TPSA (Topological Polar Surface Area)
            props['tpsa'] = Descriptors.TPSA(mol)
            
            # Elétrons de valência
            props['n_valence_electrons'] = float(Descriptors.NumValenceElectrons(mol))
            
            # Elétrons radicalares
            props['n_radical_electrons'] = float(Descriptors.NumRadicalElectrons(mol))
            
            # Fração aromática
            n_aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
            props['aromatic_fraction'] = n_aromatic_atoms / max(1, mol.GetNumAtoms())
            
            # Grau de insaturação (proxy para conjugação)
            # Formula: (2C + 2 + N - H - X) / 2
            c_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
            n_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)
            h_count = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
            x_count = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() in [9, 17, 35, 53])
            
            unsaturation = (2*c_count + 2 + n_count - h_count - x_count) / 2.0
            props['unsaturation_degree'] = max(0.0, float(unsaturation))
            
            return props
            
        except Exception as e:
            logger.debug(f"Erro ao calcular propriedades eletrônicas: {e}")
            return {
                'tpsa': 0.0,
                'n_valence_electrons': 0.0,
                'n_radical_electrons': 0.0,
                'aromatic_fraction': 0.0,
                'unsaturation_degree': 0.0
            }
    
    def approximate_frontier_orbitals(self, mol: Chem.Mol) -> dict:
        """
        Aproxima energias de orbitais de fronteira (HOMO/LUMO).
        
        NOTA: Estas são APROXIMAÇÕES usando descritores RDKit.
        Para valores reais, usar Psi4 DFT (Week 8).
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            Dict com HOMO/LUMO aproximados
        """
        try:
            # Aproximações baseadas em correlações empíricas
            
            # HOMO proxy: relacionado com ionization potential
            # Correlaciona com: -logP, eletronegatividade, TPSA
            logp = Crippen.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            n_donors = Descriptors.NumHDonors(mol)
            
            # Aproximação empírica (valores em eV, scaled)
            homo_proxy = -5.0 - 0.5 * logp - 0.01 * tpsa - 0.2 * n_donors
            
            # LUMO proxy: relacionado com electron affinity
            # Correlaciona com: logP, n_acceptors, aromatic fraction
            n_acceptors = Descriptors.NumHAcceptors(mol)
            n_aromatic = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
            aromatic_frac = n_aromatic / max(1, mol.GetNumAtoms())
            
            lumo_proxy = -2.0 + 0.5 * logp + 0.3 * n_acceptors + 2.0 * aromatic_frac
            
            # Gap
            gap_proxy = lumo_proxy - homo_proxy
            
            return {
                'homo_proxy': float(homo_proxy),
                'lumo_proxy': float(lumo_proxy),
                'gap_proxy': float(gap_proxy)
            }
            
        except Exception as e:
            logger.debug(f"Erro ao aproximar orbitais: {e}")
            return {
                'homo_proxy': -5.0,
                'lumo_proxy': -2.0,
                'gap_proxy': 3.0
            }
    
    def calculate_reactivity_indices(self, mol: Chem.Mol) -> dict:
        """
        Calcula índices de reatividade (aproximados).
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            Dict com índices de reatividade
        """
        try:
            # Eletronegatividade proxy (baseado em composição atômica)
            atom_electroneg = {
                1: 2.20,  # H
                6: 2.55,  # C
                7: 3.04,  # N
                8: 3.44,  # O
                9: 3.98,  # F
                15: 2.19, # P
                16: 2.58, # S
                17: 3.16, # Cl
                35: 2.96, # Br
                53: 2.66  # I
            }
            
            electroneg_sum = 0.0
            n_atoms = 0
            for atom in mol.GetAtoms():
                atomic_num = atom.GetAtomicNum()
                if atomic_num in atom_electroneg:
                    electroneg_sum += atom_electroneg[atomic_num]
                    n_atoms += 1
            
            avg_electroneg = electroneg_sum / max(1, n_atoms)
            
            # Nucleofilicidade proxy (HOMO energy proxy + n_donors)
            homo = self.approximate_frontier_orbitals(mol)['homo_proxy']
            n_donors = Descriptors.NumHDonors(mol)
            nucleophilicity = -homo + 0.5 * n_donors  # Maior HOMO = mais nucleofílico
            
            # Eletrofilicidade proxy (LUMO energy proxy + n_acceptors)
            lumo = self.approximate_frontier_orbitals(mol)['lumo_proxy']
            n_acceptors = Descriptors.NumHAcceptors(mol)
            electrophilicity = -lumo + 0.3 * n_acceptors  # Menor LUMO = mais eletrofílico
            
            return {
                'electronegativity': float(avg_electroneg),
                'nucleophilicity_proxy': float(nucleophilicity),
                'electrophilicity_proxy': float(electrophilicity)
            }
            
        except Exception as e:
            logger.debug(f"Erro ao calcular reatividade: {e}")
            return {
                'electronegativity': 2.5,
                'nucleophilicity_proxy': 0.0,
                'electrophilicity_proxy': 0.0
            }
    
    def encode(self, mol: Chem.Mol) -> np.ndarray:
        """
        Codifica molécula em descritores QM.
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            numpy array (embedding_dim,)
        """
        if mol is None:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        
        try:
            descriptors = []
            
            # Cargas parciais (4)
            charges = self.calculate_partial_charges(mol)
            descriptors.extend([
                charges['max'],
                charges['min'],
                charges['mean'],
                charges['std']
            ])
            
            # Propriedades eletrônicas (5)
            electronic = self.calculate_electronic_properties(mol)
            descriptors.extend([
                electronic['tpsa'],
                electronic['n_valence_electrons'],
                electronic['n_radical_electrons'],
                electronic['aromatic_fraction'],
                electronic['unsaturation_degree']
            ])
            
            # Orbitais aproximados (3)
            orbitals = self.approximate_frontier_orbitals(mol)
            descriptors.extend([
                orbitals['homo_proxy'],
                orbitals['lumo_proxy'],
                orbitals['gap_proxy']
            ])
            
            # Reatividade (3)
            reactivity = self.calculate_reactivity_indices(mol)
            descriptors.extend([
                reactivity['electronegativity'],
                reactivity['nucleophilicity_proxy'],
                reactivity['electrophilicity_proxy']
            ])
            
            # Total: 4+5+3+3 = 15 descritores
            descriptors_array = np.array(descriptors, dtype=np.float32)
            
            # Garantir dimensão correta
            if len(descriptors_array) < self.embedding_dim:
                descriptors_array = np.pad(
                    descriptors_array,
                    (0, self.embedding_dim - len(descriptors_array))
                )
            else:
                descriptors_array = descriptors_array[:self.embedding_dim]
            
            # Tratar NaN/Inf
            descriptors_array = np.nan_to_num(
                descriptors_array,
                nan=0.0,
                posinf=1e6,
                neginf=-1e6
            )
            
            return descriptors_array
            
        except Exception as e:
            logger.error(f"Erro ao codificar molécula: {e}")
            return np.zeros(self.embedding_dim, dtype=np.float32)
    
    def get_embedding_dim(self) -> int:
        return self.embedding_dim
    
    def __repr__(self):
        method = "Psi4 DFT" if self.has_psi4 else "RDKit Approximations"
        return f"QMDescriptorEncoder(dim={self.embedding_dim}, method={method})"


if __name__ == "__main__":
    # Teste com moléculas reais
    logging.basicConfig(level=logging.INFO)
    
    encoder = QMDescriptorEncoder()
    
    test_smiles = [
        "CCO",  # Ethanol
        "c1ccccc1",  # Benzene (aromático)
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine (heterocíclico)
    ]
    
    names = ["Ethanol", "Benzene", "Ibuprofen", "Caffeine"]
    
    print("\n" + "="*80)
    print("TESTE QM DESCRIPTOR ENCODER - RDKit Approximations")
    print("="*80 + "\n")
    
    for smiles, name in zip(test_smiles, names):
        mol = Chem.MolFromSmiles(smiles)
        embedding = encoder.encode(mol)
        
        print(f"🧪 {name} ({smiles})")
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   Charge max: {embedding[0]:.3f}")
        print(f"   TPSA: {embedding[4]:.3f}")
        print(f"   HOMO proxy: {embedding[9]:.3f} eV")
        print(f"   LUMO proxy: {embedding[10]:.3f} eV")
        print(f"   Gap proxy: {embedding[11]:.3f} eV")
        print(f"   Norm: {np.linalg.norm(embedding):.2f}")
        print()
    
    print("="*80)
    print(f"✅ QM Encoder funcionando!")
    print(f"   {encoder}")
    print("   🔬 Week 8: Upgrade para Psi4 DFT real!")
    print("="*80)

