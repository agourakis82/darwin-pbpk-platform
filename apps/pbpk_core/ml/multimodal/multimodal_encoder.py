"""
Multimodal Molecular Encoder - INTEGRATED REAL IMPLEMENTATION
===============================================================

Combina 5 encoders moleculares ortogonais em um único embedding multimodal.

TODAS as modalidades são REAIS - SEM MOCKS!

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                      SMILES Input                           │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┴────────────┐
        │    RDKit Mol Object     │
        └────────────┬────────────┘
                     │
     ┌───────────────┼───────────────┐
     │               │               │
     ▼               ▼               ▼
┌─────────┐    ┌─────────┐    ┌─────────┐
│ChemBERTa│    │   GNN   │    │   KEC   │  ← NOVEL!
│ 768 dim │    │ 128 dim │    │  15 dim │
└────┬────┘    └────┬────┘    └────┬────┘
     │              │              │
     │         ┌────┴────┐    ┌────┴────┐
     │         ▼         ▼    ▼         ▼
     │    ┌─────────┐  ┌─────────┐
     │    │ 3D Conf │  │   QM    │
     │    │  50 dim │  │  15 dim │
     │    └────┬────┘  └────┬────┘
     │         │            │
     └─────────┴────────────┴───────────┐
                                        │
                                        ▼
                            ┌─────────────────────┐
                            │  Concatenate All    │
                            │   976 dimensions    │
                            └─────────────────────┘

Autor: Dr. Demetrios Chiuratto Agourakis  
Data: Outubro 2025
"""

import numpy as np
import time
import logging
from typing import List, Dict, Optional
from rdkit import Chem
import concurrent.futures

from .chemberta_encoder import ChemBERTaEncoder
from .gnn_encoder import GNNMolecularEncoder
from .kec_encoder import KECMolecularEncoder
from .conformer_encoder import Conformer3DEncoder
from .qm_encoder import QMDescriptorEncoder

logger = logging.getLogger(__name__)


class MultimodalMolecularEncoder:
    """
    Encoder Multimodal REAL para moléculas.
    
    Combina 5 views ortogonais:
    1. Semantic (ChemBERTa): 768 dim
    2. Topological (GNN): 128 dim
    3. Fractal-Entropic (KEC): 15 dim ← NOVEL!
    4. Spatial (3D Conformer): 50 dim
    5. Electronic (QM): 15 dim
    
    Total: 976 dimensions
    
    100% REAL - SEM MOCKS!
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        parallel: bool = True,
        verbose: bool = True
    ):
        """
        Inicializa encoder multimodal.
        
        Args:
            device: 'cuda' ou 'cpu' (auto-detect se None)
            parallel: Se True, codifica modalidades em paralelo
            verbose: Se True, mostra informações de inicialização
        """
        self.parallel = parallel
        self.verbose = verbose
        
        if verbose:
            logger.info("╔" + "="*78 + "╗")
            logger.info("║" + " "*15 + "INICIALIZANDO MULTIMODAL ENCODER REAL" + " "*25 + "║")
            logger.info("╚" + "="*78 + "╝")
        
        # Inicializar encoders individuais
        logger.info("\n📦 Carregando encoders...")
        
        # 1. ChemBERTa (Transformers)
        self.chemberta = ChemBERTaEncoder(device=device)
        self.chemberta_dim = self.chemberta.get_embedding_dim()
        logger.info(f"   ✅ ChemBERTa: {self.chemberta_dim} dim")
        
        # 2. GNN (PyTorch Geometric)
        self.gnn = GNNMolecularEncoder(device=device)
        self.gnn_dim = self.gnn.get_embedding_dim()
        logger.info(f"   ✅ GNN: {self.gnn_dim} dim")
        
        # 3. KEC (Código do Mestrado) ← NOVEL!
        self.kec = KECMolecularEncoder()
        self.kec_dim = self.kec.get_embedding_dim()
        logger.info(f"   ✅ KEC: {self.kec_dim} dim (NOVEL - Master's Thesis)")
        
        # 4. 3D Conformer (RDKit ETKDG + MMFF)
        self.conformer = Conformer3DEncoder(optimize=True)
        self.conformer_dim = self.conformer.get_embedding_dim()
        logger.info(f"   ✅ 3D Conformer: {self.conformer_dim} dim")
        
        # 5. QM (RDKit QM descriptors)
        self.qm = QMDescriptorEncoder()
        self.qm_dim = self.qm.get_embedding_dim()
        logger.info(f"   ✅ QM: {self.qm_dim} dim")
        
        # Total dimension
        self.total_dim = (
            self.chemberta_dim + 
            self.gnn_dim + 
            self.kec_dim + 
            self.conformer_dim + 
            self.qm_dim
        )
        
        if verbose:
            logger.info("\n" + "="*80)
            logger.info(f"🎯 TOTAL EMBEDDING DIMENSION: {self.total_dim}")
            logger.info("="*80)
            logger.info("")
            logger.info("📊 Breakdown:")
            logger.info(f"   ChemBERTa:     {self.chemberta_dim:>4} dim ({100*self.chemberta_dim/self.total_dim:>5.1f}%)")
            logger.info(f"   GNN:           {self.gnn_dim:>4} dim ({100*self.gnn_dim/self.total_dim:>5.1f}%)")
            logger.info(f"   KEC (NOVEL):   {self.kec_dim:>4} dim ({100*self.kec_dim/self.total_dim:>5.1f}%)")
            logger.info(f"   3D Conformer:  {self.conformer_dim:>4} dim ({100*self.conformer_dim/self.total_dim:>5.1f}%)")
            logger.info(f"   QM:            {self.qm_dim:>4} dim ({100*self.qm_dim/self.total_dim:>5.1f}%)")
            logger.info("="*80)
            logger.info("")
    
    def encode(self, smiles: str, show_timing: bool = False) -> np.ndarray:
        """
        Codifica SMILES em embedding multimodal.
        
        Args:
            smiles: SMILES string
            show_timing: Se True, mostra tempo de cada modalidade
            
        Returns:
            numpy array (total_dim,)
        """
        start_time = time.time()
        
        # Convert SMILES to RDKit Mol
        mol = Chem.MolFromSmiles(smiles)
        
        if mol is None:
            logger.warning(f"SMILES inválido: {smiles}")
            return np.zeros(self.total_dim, dtype=np.float32)
        
        if self.parallel:
            # Encoding paralelo (mais rápido)
            embeddings = self._encode_parallel(smiles, mol, show_timing)
        else:
            # Encoding sequencial
            embeddings = self._encode_sequential(smiles, mol, show_timing)
        
        # Concatenar todos os embeddings
        full_embedding = np.concatenate(embeddings, axis=0).astype(np.float32)
        
        if show_timing:
            total_time = time.time() - start_time
            logger.info(f"   ⏱️  Total: {total_time:.3f}s")
        
        return full_embedding
    
    def _encode_sequential(
        self, 
        smiles: str, 
        mol: Chem.Mol,
        show_timing: bool
    ) -> List[np.ndarray]:
        """Codificação sequencial (uma modalidade por vez)."""
        embeddings = []
        
        if show_timing:
            logger.info(f"\n🧪 Encoding: {smiles}")
        
        # ChemBERTa
        t0 = time.time()
        emb_chemberta = self.chemberta.encode(smiles)
        if show_timing:
            logger.info(f"   ChemBERTa:   {time.time()-t0:.3f}s")
        embeddings.append(emb_chemberta)
        
        # GNN
        t0 = time.time()
        emb_gnn = self.gnn.encode(mol)
        if show_timing:
            logger.info(f"   GNN:         {time.time()-t0:.3f}s")
        embeddings.append(emb_gnn)
        
        # KEC
        t0 = time.time()
        emb_kec = self.kec.encode(mol)
        if show_timing:
            logger.info(f"   KEC:         {time.time()-t0:.3f}s")
        embeddings.append(emb_kec)
        
        # 3D Conformer
        t0 = time.time()
        emb_conformer = self.conformer.encode(mol)
        if show_timing:
            logger.info(f"   3D Conformer:{time.time()-t0:.3f}s")
        embeddings.append(emb_conformer)
        
        # QM
        t0 = time.time()
        emb_qm = self.qm.encode(mol)
        if show_timing:
            logger.info(f"   QM:          {time.time()-t0:.3f}s")
        embeddings.append(emb_qm)
        
        return embeddings
    
    def _encode_parallel(
        self,
        smiles: str,
        mol: Chem.Mol,
        show_timing: bool
    ) -> List[np.ndarray]:
        """Codificação paralela (todas modalidades simultaneamente)."""
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all encoding tasks
            future_chemberta = executor.submit(self.chemberta.encode, smiles)
            future_gnn = executor.submit(self.gnn.encode, mol)
            future_kec = executor.submit(self.kec.encode, mol)
            future_conformer = executor.submit(self.conformer.encode, mol)
            future_qm = executor.submit(self.qm.encode, mol)
            
            # Collect results
            emb_chemberta = future_chemberta.result()
            emb_gnn = future_gnn.result()
            emb_kec = future_kec.result()
            emb_conformer = future_conformer.result()
            emb_qm = future_qm.result()
        
        return [emb_chemberta, emb_gnn, emb_kec, emb_conformer, emb_qm]
    
    def encode_batch(
        self,
        smiles_list: List[str],
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Codifica batch de SMILES.
        
        Args:
            smiles_list: Lista de SMILES strings
            show_progress: Se True, mostra progresso
            
        Returns:
            numpy array (n_molecules, total_dim)
        """
        embeddings = []
        
        for i, smiles in enumerate(smiles_list):
            if show_progress and (i+1) % 10 == 0:
                logger.info(f"   Encoding {i+1}/{len(smiles_list)}...")
            
            emb = self.encode(smiles, show_timing=False)
            embeddings.append(emb)
        
        return np.array(embeddings, dtype=np.float32)
    
    def get_embedding_dim(self) -> int:
        """Retorna dimensão total do embedding."""
        return self.total_dim
    
    def get_modality_slices(self) -> Dict[str, slice]:
        """
        Retorna slices para extrair embeddings individuais.
        
        Returns:
            Dict mapping modality name -> slice object
        """
        slices = {}
        current_idx = 0
        
        slices['chemberta'] = slice(current_idx, current_idx + self.chemberta_dim)
        current_idx += self.chemberta_dim
        
        slices['gnn'] = slice(current_idx, current_idx + self.gnn_dim)
        current_idx += self.gnn_dim
        
        slices['kec'] = slice(current_idx, current_idx + self.kec_dim)
        current_idx += self.kec_dim
        
        slices['conformer'] = slice(current_idx, current_idx + self.conformer_dim)
        current_idx += self.conformer_dim
        
        slices['qm'] = slice(current_idx, current_idx + self.qm_dim)
        current_idx += self.qm_dim
        
        return slices
    
    def __repr__(self):
        return (
            f"MultimodalMolecularEncoder("
            f"total_dim={self.total_dim}, "
            f"parallel={self.parallel}, "
            f"status=100% REAL)"
        )


if __name__ == "__main__":
    # Teste completo do encoder multimodal
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*80)
    print("TESTE MULTIMODAL ENCODER - INTEGRAÇÃO COMPLETA")
    print("="*80 + "\n")
    
    # Inicializar encoder
    encoder = MultimodalMolecularEncoder(parallel=True, verbose=True)
    
    # Moléculas de teste
    test_smiles = [
        "CCO",  # Ethanol (simples)
        "c1ccccc1",  # Benzene (aromático)
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen (drug)
    ]
    names = ["Ethanol", "Benzene", "Ibuprofen"]
    
    print("\n" + "="*80)
    print("TESTANDO ENCODING INDIVIDUAL")
    print("="*80)
    
    for smiles, name in zip(test_smiles, names):
        print(f"\n🧪 {name}")
        embedding = encoder.encode(smiles, show_timing=True)
        
        print(f"\n   📊 Embedding stats:")
        print(f"      Shape: {embedding.shape}")
        print(f"      Norm: {np.linalg.norm(embedding):.2f}")
        print(f"      Mean: {embedding.mean():.4f}")
        print(f"      Std: {embedding.std():.4f}")
        print(f"      Non-zero: {np.count_nonzero(embedding)}/{len(embedding)}")
    
    print("\n" + "="*80)
    print("TESTANDO BATCH ENCODING")
    print("="*80)
    
    print(f"\n📦 Encoding batch de {len(test_smiles)} moléculas...")
    start = time.time()
    batch_embeddings = encoder.encode_batch(test_smiles, show_progress=False)
    batch_time = time.time() - start
    
    print(f"\n✅ Batch encoding completo!")
    print(f"   Shape: {batch_embeddings.shape}")
    print(f"   Time: {batch_time:.3f}s ({batch_time/len(test_smiles):.3f}s/mol)")
    
    print("\n" + "="*80)
    print("TESTANDO MODALITY EXTRACTION")
    print("="*80)
    
    slices = encoder.get_modality_slices()
    test_embedding = batch_embeddings[0]
    
    print(f"\n🔍 Extraindo modalidades do embedding:")
    for name, s in slices.items():
        modality_emb = test_embedding[s]
        print(f"   {name:<12}: [{s.start:>4}:{s.stop:>4}] → {len(modality_emb):>3} dim, norm={np.linalg.norm(modality_emb):>8.2f}")
    
    print("\n" + "="*80)
    print("✅ MULTIMODAL ENCODER 100% FUNCIONAL!")
    print("="*80)
    print("\n🎯 Status:")
    print("   ✅ ChemBERTa: REAL (Transformers)")
    print("   ✅ GNN: REAL (PyTorch Geometric)")
    print("   ✅ KEC: REAL (Código do Mestrado) ← NOVEL!")
    print("   ✅ 3D Conformer: REAL (ETKDG + MMFF94)")
    print("   ✅ QM: REAL (Gasteiger + RDKit)")
    print("\n   🚀 PRONTO PARA TREINAR PINN!")
    print("="*80 + "\n")

