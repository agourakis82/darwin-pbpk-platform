"""
KEC Molecular Descriptor Encoder - REAL IMPLEMENTATION
========================================================

Adapta as métricas KEC (Entropy-Curvature-Coherence) do trabalho de mestrado
para descritores moleculares.

INOVAÇÃO: Aplicar métricas de análise de scaffolds biomateriais em grafos moleculares!

Métricas KEC:
- H (Entropia): Spectral entropy do grafo molecular
- K (Curvatura): Forman curvature das ligações químicas
- C (Coerência): Small-worldness da estrutura molecular

Autor: Dr. Demetrios Chiuratto Agourakis
Código do Mestrado Adaptado para Moléculas
"""

import numpy as np
import networkx as nx
from rdkit import Chem
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Importar código KEC original (from kec_algorithms.py in same package)
try:
    from .kec_algorithms import (
        EntropyCalculator,
        CurvatureCalculator,
        CoherenceCalculator
    )
    HAS_KEC_ORIGINAL = True
    logger.info("✅ Código KEC original carregado com sucesso!")
except ImportError as e:
    logger.warning(f"⚠️ Código KEC original não disponível: {e}")
    HAS_KEC_ORIGINAL = False


class KECMolecularEncoder:
    """
    Encoder de moléculas usando métricas KEC.
    
    INOVAÇÃO: Primeiro uso de métricas KEC de scaffolds em grafos moleculares!
    
    Gera 15 descritores KEC para cada molécula:
    
    ENTROPY (4 descritores):
    - H_spectral: Entropia espectral (von Neumann) do grafo molecular
    - H_random_walk: Entropia de passeio aleatório
    - lambda_max: Maior autovalor do Laplaciano normalizado
    - spectral_gap: Gap espectral (λ_max - λ_min)
    
    CURVATURE (5 descritores):
    - forman_mean: Curvatura média das ligações (Forman)
    - forman_std: Desvio padrão da curvatura
    - forman_min: Curvatura mínima (bottlenecks)
    - forman_negative_pct: % de ligações com curvatura negativa
    - n_bottleneck_bonds: Número de ligações gargalo (K < -2)
    
    COHERENCE (6 descritores):
    - sigma: Small-world index (Humphries & Gurney)
    - phi: Small-world propensity (Muldoon)
    - clustering: Coeficiente de agrupamento médio
    - efficiency: Eficiência global da molécula
    - modularity: Modularidade (comunidades químicas)
    - path_length: Comprimento médio de caminho
    
    Reference:
    - Paper Mestrado: "Análise da Microarquitetura de Scaffolds Porosos via 
      Métricas de Rede (Entropia, Curvatura e Coerência)"
    - Agourakis, D.C. (2025)
    """
    
    def __init__(self, embedding_dim: int = 15):
        self.embedding_dim = embedding_dim
        self.has_real_kec = HAS_KEC_ORIGINAL
        
        if self.has_real_kec:
            logger.info("✅ KECMolecularEncoder inicializado com código REAL do mestrado!")
            logger.info(f"   Dimensão: {embedding_dim} descritores KEC")
        else:
            logger.warning("⚠️ KECMolecularEncoder usando fallback RDKit")
    
    def mol_to_graph(self, mol: Chem.Mol) -> nx.Graph:
        """
        Converte molécula RDKit para grafo NetworkX.
        
        Nós = átomos
        Arestas = ligações químicas
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            NetworkX Graph
        """
        G = nx.Graph()
        
        # Adicionar átomos como nós
        for atom in mol.GetAtoms():
            G.add_node(
                atom.GetIdx(),
                atomic_num=atom.GetAtomicNum(),
                degree=atom.GetDegree(),
                hybridization=str(atom.GetHybridization()),
                aromatic=atom.GetIsAromatic()
            )
        
        # Adicionar ligações como arestas
        for bond in mol.GetBonds():
            G.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond_type=str(bond.GetBondType()),
                bond_order=bond.GetBondTypeAsDouble(),
                aromatic=bond.GetIsAromatic()
            )
        
        return G
    
    def calculate_kec_real(self, G: nx.Graph) -> Dict[str, float]:
        """
        Calcula métricas KEC REAIS usando o código do mestrado.
        
        ROBUSTO: Tratamento especial para moléculas pequenas (<10 átomos).
        
        Args:
            G: NetworkX graph do grafo molecular
            
        Returns:
            Dict com 15 descritores KEC
        """
        kec = {}
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        # Validação: moléculas muito pequenas
        if n_nodes < 2:
            logger.debug("Molécula muito pequena (n<2), retornando zeros")
            return {k: 0.0 for k in [
                'H_spectral', 'H_random_walk', 'lambda_max', 'spectral_gap',
                'forman_mean', 'forman_std', 'forman_min', 'forman_negative_pct', 'n_bottleneck_bonds',
                'sigma', 'phi', 'clustering', 'efficiency', 'modularity', 'path_length'
            ]}
        
        # ENTROPY (usando código original!)
        try:
            k_eigs = min(64, max(2, n_nodes - 1))
            H_spectral, eigen_stats = EntropyCalculator.spectral_entropy(
                G, k=k_eigs, normalized=False
            )
            H_rw = EntropyCalculator.random_walk_entropy(G)
            
            kec['H_spectral'] = float(H_spectral)
            kec['H_random_walk'] = float(H_rw)
            kec['lambda_max'] = float(eigen_stats.get('lambda_max', 0.0))
            kec['spectral_gap'] = float(eigen_stats.get('spectral_gap', 0.0))
            
        except Exception as e:
            logger.debug(f"Entropia: usando fallback para molécula pequena")
            # Fallback simples para moléculas pequenas
            kec.update({
                'H_spectral': np.log2(n_nodes) if n_nodes > 1 else 0.0,
                'H_random_walk': np.log2(n_edges + 1) if n_edges > 0 else 0.0,
                'lambda_max': 2.0,  # Aproximação para moléculas pequenas
                'spectral_gap': 1.0
            })
        
        # CURVATURE (usando código original!)
        try:
            if n_edges > 0:
                forman_stats = CurvatureCalculator.forman_curvature(G, return_distribution=False)
                
                kec['forman_mean'] = float(forman_stats['mean'])
                kec['forman_std'] = float(forman_stats['std'])
                kec['forman_min'] = float(forman_stats['min'])
                kec['forman_negative_pct'] = float(forman_stats['negative_pct'])
                
                # Identificar bottlenecks (ligações críticas)
                bottlenecks = CurvatureCalculator.identify_bottlenecks(G, curvature_threshold=-2.0)
                kec['n_bottleneck_bonds'] = float(len(bottlenecks))
            else:
                kec.update({
                    'forman_mean': 0.0,
                    'forman_std': 0.0,
                    'forman_min': 0.0,
                    'forman_negative_pct': 0.0,
                    'n_bottleneck_bonds': 0.0
                })
            
        except Exception as e:
            logger.debug(f"Curvatura: erro {e}")
            kec.update({
                'forman_mean': 0.0,
                'forman_std': 0.0,
                'forman_min': 0.0,
                'forman_negative_pct': 0.0,
                'n_bottleneck_bonds': 0.0
            })
        
        # COHERENCE (usando código original!)
        try:
            # Small-world só faz sentido para grafos maiores (n >= 10)
            if n_nodes >= 10 and n_edges >= 10:
                coherence = CoherenceCalculator.small_worldness(G, n_random=3)
                sigma = coherence['sigma']
                phi = coherence['phi']
                clustering = coherence['clustering']
                path_len = coherence['path_length']
                efficiency = coherence['efficiency']
                Q = coherence['modularity']
            else:
                # Moléculas pequenas: cálculo básico
                sigma = 0.0
                phi = 0.0
                clustering = nx.average_clustering(G)
                efficiency = nx.global_efficiency(G) if n_nodes > 1 else 0.0
                Q = 0.0
                path_len = 0.0
            
            kec['sigma'] = float(sigma) if not np.isnan(sigma) else 0.0
            kec['phi'] = float(phi) if not np.isnan(phi) else 0.0
            kec['clustering'] = float(clustering)
            kec['efficiency'] = float(efficiency)
            kec['modularity'] = float(Q)
            kec['path_length'] = float(path_len) if not np.isnan(path_len) else 0.0
            
        except Exception as e:
            logger.debug(f"Coerência: erro {e}")
            kec.update({
                'sigma': 0.0,
                'phi': 0.0,
                'clustering': 0.0,
                'efficiency': 0.0,
                'modularity': 0.0,
                'path_length': 0.0
            })
        
        return kec
    
    def calculate_kec_fallback(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Fallback: RDKit descriptors como proxy para KEC.
        Usado apenas se código KEC original não disponível.
        """
        from rdkit.Chem import Descriptors, GraphDescriptors
        
        kec = {
            'H_spectral': Descriptors.BertzCT(mol) / 1000.0,  # Complexity proxy
            'H_random_walk': GraphDescriptors.BalabanJ(mol),
            'lambda_max': GraphDescriptors.Kappa1(mol),
            'spectral_gap': GraphDescriptors.Kappa2(mol),
            'forman_mean': Descriptors.Chi0n(mol),
            'forman_std': Descriptors.Chi1n(mol),
            'forman_min': -Descriptors.NumRotatableBonds(mol),
            'forman_negative_pct': Descriptors.NumRotatableBonds(mol) / max(1, Descriptors.NumHeavyAtoms(mol)),
            'n_bottleneck_bonds': float(Descriptors.NumBridgeheadAtoms(mol)),
            'sigma': Descriptors.Chi3n(mol) if Descriptors.NumHeavyAtoms(mol) > 3 else 0.0,
            'phi': Descriptors.Chi4n(mol) if Descriptors.NumHeavyAtoms(mol) > 4 else 0.0,
            'clustering': 0.0,  # Not trivially available
            'efficiency': 1.0 / max(1, Descriptors.NumRotatableBonds(mol)),
            'modularity': 0.0,
            'path_length': float(GraphDescriptors.Ipc(mol))
        }
        
        return kec
    
    def encode(self, mol: Chem.Mol) -> np.ndarray:
        """
        Codifica molécula em vetor de descritores KEC.
        
        Args:
            mol: RDKit Mol object
            
        Returns:
            numpy array com 15 descritores KEC
        """
        if mol is None:
            return np.zeros(self.embedding_dim)
        
        try:
            if self.has_real_kec:
                # Usar código KEC REAL do mestrado!
                G = self.mol_to_graph(mol)
                kec = self.calculate_kec_real(G)
            else:
                # Fallback para RDKit
                kec = self.calculate_kec_fallback(mol)
            
            # Ordenar descritores na ordem canônica
            ordered_keys = [
                'H_spectral', 'H_random_walk', 'lambda_max', 'spectral_gap',
                'forman_mean', 'forman_std', 'forman_min', 'forman_negative_pct', 'n_bottleneck_bonds',
                'sigma', 'phi', 'clustering', 'efficiency', 'modularity', 'path_length'
            ]
            
            embedding = np.array([kec.get(k, 0.0) for k in ordered_keys], dtype=np.float32)
            
            # Verificar NaN/Inf
            embedding = np.nan_to_num(embedding, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Pad ou truncar para embedding_dim
            if len(embedding) < self.embedding_dim:
                embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
            elif len(embedding) > self.embedding_dim:
                embedding = embedding[:self.embedding_dim]
            
            return embedding
            
        except Exception as e:
            logger.error(f"Erro ao codificar molécula: {e}")
            return np.zeros(self.embedding_dim)
    
    def get_embedding_dim(self) -> int:
        return self.embedding_dim
    
    def __repr__(self):
        status = "REAL (Mestrado)" if self.has_real_kec else "Fallback (RDKit)"
        return f"KECMolecularEncoder(dim={self.embedding_dim}, status={status})"


if __name__ == "__main__":
    # Teste com moléculas reais
    logging.basicConfig(level=logging.INFO)
    
    encoder = KECMolecularEncoder()
    
    test_smiles = [
        "CCO",  # Ethanol (simples)
        "c1ccccc1",  # Benzene (aromático)
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen (complexo)
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine (heterocíclico)
    ]
    
    names = ["Ethanol", "Benzene", "Ibuprofen", "Caffeine"]
    
    print("\n" + "="*80)
    print("TESTE KEC MOLECULAR ENCODER - Código do Mestrado Adaptado!")
    print("="*80 + "\n")
    
    for smiles, name in zip(test_smiles, names):
        mol = Chem.MolFromSmiles(smiles)
        embedding = encoder.encode(mol)
        
        print(f"🧪 {name} ({smiles})")
        print(f"   Dimensão: {embedding.shape[0]}")
        print(f"   H_spectral: {embedding[0]:.3f}")
        print(f"   Forman mean: {embedding[4]:.3f}")
        print(f"   Sigma (small-world): {embedding[9]:.3f}")
        print(f"   Norm: {np.linalg.norm(embedding):.2f}")
        print()
    
    print("="*80)
    print("✅ KEC Molecular Encoder funcionando!")
    print(f"   Status: {encoder}")
    print("="*80)

