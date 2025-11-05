"""
ChemBERTa Encoder - REAL IMPLEMENTATION
========================================

NO MOCKS. Real ChemBERTa with Transformers library.
"""
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class ChemBERTaEncoder:
    """
    REAL ChemBERTa encoder using pre-trained model.
    
    Model: DeepChem's ChemBERTa-77M-MLM or seyonec/ChemBERTa-zinc-base-v1
    """
    
    def __init__(
        self,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize REAL ChemBERTa encoder.
        
        Args:
            model_name: HuggingFace model name
            device: 'cuda' or 'cpu'
            cache_dir: Directory to cache model
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        logger.info(f"Loading REAL ChemBERTa model: {model_name}")
        logger.info(f"Device: {self.device}")
        
        # Load REAL tokenizer and model from HuggingFace
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            self.model = AutoModel.from_pretrained(
                model_name,
                cache_dir=cache_dir
            )
            self.model.to(self.device)
            self.model.eval()
            
            # Get embedding dimension from model config
            self.embedding_dim = self.model.config.hidden_size
            
            logger.info(f"✅ REAL ChemBERTa loaded successfully!")
            logger.info(f"   Embedding dim: {self.embedding_dim}")
            logger.info(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            logger.error(f"Failed to load ChemBERTa: {e}")
            raise RuntimeError(f"Could not load REAL ChemBERTa model: {e}")
    
    def encode(self, smiles: str) -> np.ndarray:
        """
        Encode a single SMILES string to embedding.
        
        Args:
            smiles: SMILES string
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        return self.encode_batch([smiles])[0]
    
    def encode_batch(self, smiles_list: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode multiple SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
            batch_size: Batch size for processing
            
        Returns:
            numpy array of shape (len(smiles_list), embedding_dim)
        """
        all_embeddings = []
        
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i:i+batch_size]
            
            # Tokenize REAL SMILES
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get REAL embeddings from model
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Use [CLS] token embedding or mean pooling
                # Mean pooling over sequence length
                embeddings = outputs.last_hidden_state.mean(dim=1)
                
                all_embeddings.append(embeddings.cpu().numpy())
        
        return np.vstack(all_embeddings)
    
    def get_embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self.embedding_dim
    
    def __repr__(self):
        return f"ChemBERTaEncoder(model={self.model_name}, dim={self.embedding_dim}, device={self.device})"


if __name__ == "__main__":
    # Test with real molecules
    logging.basicConfig(level=logging.INFO)
    
    encoder = ChemBERTaEncoder()
    
    test_smiles = [
        "CCO",  # Ethanol
        "c1ccccc1",  # Benzene
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
    ]
    
    print("\nTesting REAL ChemBERTa encoder:")
    embeddings = encoder.encode_batch(test_smiles)
    print(f"✅ Encoded {len(test_smiles)} molecules")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Mean: {embeddings.mean():.4f}")
    print(f"   Std: {embeddings.std():.4f}")
    print(f"   Min: {embeddings.min():.4f}")
    print(f"   Max: {embeddings.max():.4f}")

