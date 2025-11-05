#!/usr/bin/env python3
"""
🚀 TRAIN PBPK MODEL WITH EXPANDED DATASET
==========================================

Re-treina o modelo Trial 84 (best performing) com dataset expandido:
- Original: 478 compostos
- Expandido: 5032 compostos (10.5x mais!)

Expected improvement: R² 0.209 → 0.30-0.40

Author: Dr. Demetrios Chiuratto Agourakis  
Date: October 28, 2025
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import pickle
import json
from typing import Dict, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️  Device: {device}")

# Paths
COMBINED_DATA = BASE_DIR / 'data' / 'processed' / 'combined_kec_tdc_split.pkl'
EMBEDDINGS = BASE_DIR / 'data' / 'embeddings' / 'rich_embeddings_expanded_788d.npz'
OUTPUT_DIR = BASE_DIR / 'results' / 'pbpk_expanded_training'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class PKDataset(Dataset):
    """PyTorch dataset for PK parameters"""
    
    def __init__(self, embeddings, targets, masks):
        self.embeddings = torch.FloatTensor(embeddings)
        self.targets = {
            'fu': torch.FloatTensor(targets['fu']),
            'vd': torch.FloatTensor(targets['vd']),
            'clearance': torch.FloatTensor(targets['clearance'])
        }
        self.masks = {
            'fu': torch.BoolTensor(masks['fu']),
            'vd': torch.BoolTensor(masks['vd']),
            'clearance': torch.BoolTensor(masks['clearance'])
        }
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return {
            'embedding': self.embeddings[idx],
            'targets': {k: v[idx] for k, v in self.targets.items()},
            'masks': {k: v[idx] for k, v in self.masks.items()}
        }


class FlexiblePKModel(nn.Module):
    """
    Trial 84 architecture (best performing)
    
    Architecture:
    - Input: 788d (768d ChemBERTa + 20d RDKit)
    - Hidden: [384, 1024, 640]
    - Dropout: 0.492
    - Activation: GELU
    - Output: 3 (Fu, Vd, Clearance)
    """
    
    def __init__(
        self,
        input_dim=788,
        hidden_dims=[384, 1024, 640],
        output_dim=3,
        dropout=0.492,
        use_batch_norm=False,
        activation='gelu'
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'gelu':
                layers.append(nn.GELU())
            elif activation == 'elu':
                layers.append(nn.ELU())
            
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def apply_transforms_simple(df):
    """Apply simple logit/log1p transforms"""
    
    # Fu: logit transform
    fu_transformed = np.log(
        np.clip(df['fu'].values, 1e-6, 1-1e-6) / 
        (1 - np.clip(df['fu'].values, 1e-6, 1-1e-6))
    )
    fu_transformed = np.clip(fu_transformed, -10, 10)
    
    # Vd: log1p transform
    vd_transformed = np.log1p(df['vd'].values)
    vd_transformed = np.clip(vd_transformed, -10, 10)
    
    # Clearance: log1p transform
    cl_transformed = np.log1p(df['clearance'].values)
    cl_transformed = np.clip(cl_transformed, -10, 10)
    
    # Replace NaN with 0 (will be masked anyway)
    fu_transformed = np.nan_to_num(fu_transformed, nan=0.0)
    vd_transformed = np.nan_to_num(vd_transformed, nan=0.0)
    cl_transformed = np.nan_to_num(cl_transformed, nan=0.0)
    
    return {
        'targets': {
            'fu': fu_transformed,
            'vd': vd_transformed,
            'clearance': cl_transformed
        },
        'masks': {
            'fu': ~df['fu'].isna().values,
            'vd': ~df['vd'].isna().values,
            'clearance': ~df['clearance'].isna().values
        }
    }


def inverse_transform_simple(pred, param):
    """Inverse transform predictions"""
    
    if param == 'fu':
        # Inverse logit
        pred = np.clip(pred, -10, 10)
        return 1 / (1 + np.exp(-pred))
    
    elif param in ['vd', 'clearance']:
        # Inverse log1p
        pred = np.clip(pred, -10, 10)
        return np.expm1(pred)
    
    return pred


def load_data():
    """Load and prepare expanded dataset"""
    print("\n📁 Loading expanded dataset...")
    
    # Load splits
    with open(COMBINED_DATA, 'rb') as f:
        data = pickle.load(f)
    
    # Load embeddings
    emb_data = np.load(EMBEDDINGS)
    embeddings_all = emb_data['embeddings']
    smiles_all = emb_data['smiles'].tolist()
    
    print(f"  ✓ Embeddings: {embeddings_all.shape}")
    print(f"  ✓ Total SMILES: {len(smiles_all)}")
    
    # Create SMILES to embedding mapping
    smiles_to_idx = {smi: i for i, smi in enumerate(smiles_all)}
    
    # Prepare data for each split
    datasets = {}
    
    for split in ['train', 'val', 'test']:
        df = data[split]['dataframe']
        split_smiles = data[split]['smiles']
        
        # Filter to only SMILES that have embeddings
        valid_smiles = [smi for smi in split_smiles if smi in smiles_to_idx]
        
        # Get embeddings in the same order as valid_smiles
        indices = [smiles_to_idx[smi] for smi in valid_smiles]
        split_embeddings = embeddings_all[indices]
        
        # Get matching dataframe rows in the same order
        df_matched = df[df['smiles'].isin(valid_smiles)].copy()
        df_matched = df_matched.set_index('smiles').loc[valid_smiles].reset_index()
        
        # Apply transforms
        transformed = apply_transforms_simple(df_matched)
        
        datasets[split] = {
            'embeddings': split_embeddings,
            'targets': transformed['targets'],
            'masks': transformed['masks'],
            'dataframe': df_matched
        }
        
        print(f"\n  {split.upper()}:")
        print(f"    Compounds: {len(split_embeddings)}")
        print(f"    Fu:  {transformed['masks']['fu'].sum()} ({transformed['masks']['fu'].sum()/len(split_embeddings)*100:.1f}%)")
        print(f"    Vd:  {transformed['masks']['vd'].sum()} ({transformed['masks']['vd'].sum()/len(split_embeddings)*100:.1f}%)")
        print(f"    CL:  {transformed['masks']['clearance'].sum()} ({transformed['masks']['clearance'].sum()/len(split_embeddings)*100:.1f}%)")
    
    return datasets


def evaluate(model, dataloader, split_name='val'):
    """Evaluate model"""
    model.eval()
    
    all_preds = {'fu': [], 'vd': [], 'clearance': []}
    all_targets = {'fu': [], 'vd': [], 'clearance': []}
    all_masks = {'fu': [], 'vd': [], 'clearance': []}
    
    with torch.no_grad():
        for batch in dataloader:
            embeddings = batch['embedding'].to(device)
            outputs = model(embeddings)
            
            for i, param in enumerate(['fu', 'vd', 'clearance']):
                preds = outputs[:, i].cpu().numpy()
                targets = batch['targets'][param].numpy()
                masks = batch['masks'][param].numpy()
                
                all_preds[param].extend(preds[masks])
                all_targets[param].extend(targets[masks])
                all_masks[param].extend(masks)
    
    # Calculate metrics
    metrics = {}
    
    for param in ['fu', 'vd', 'clearance']:
        if len(all_preds[param]) == 0:
            continue
        
        # Inverse transform
        preds_orig = inverse_transform_simple(np.array(all_preds[param]), param)
        targets_orig = inverse_transform_simple(np.array(all_targets[param]), param)
        
        # R²
        ss_res = np.sum((targets_orig - preds_orig) ** 2)
        ss_tot = np.sum((targets_orig - targets_orig.mean()) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else -np.inf
        
        # MAE
        mae = np.mean(np.abs(targets_orig - preds_orig))
        
        # RMSE
        rmse = np.sqrt(np.mean((targets_orig - preds_orig) ** 2))
        
        # 2-fold accuracy
        ratio = preds_orig / (targets_orig + 1e-10)
        fold2_acc = np.mean((ratio >= 0.5) & (ratio <= 2.0)) * 100
        
        metrics[param] = {
            'r2': float(r2),
            'mae': float(mae),
            'rmse': float(rmse),
            'fold2_acc': float(fold2_acc),
            'n_samples': len(preds_orig)
        }
    
    # Overall R²
    all_r2 = [metrics[p]['r2'] for p in ['fu', 'vd', 'clearance'] if p in metrics]
    metrics['overall'] = {'r2': float(np.mean(all_r2)) if all_r2 else -np.inf}
    
    return metrics


def train_model(datasets, num_epochs=200, batch_size=64, learning_rate=0.0003):
    """Train model with expanded dataset"""
    
    print("\n" + "="*80)
    print("🚀 TRAINING PBPK MODEL WITH EXPANDED DATASET")
    print("="*80)
    
    # Create dataloaders
    train_dataset = PKDataset(
        datasets['train']['embeddings'],
        datasets['train']['targets'],
        datasets['train']['masks']
    )
    
    val_dataset = PKDataset(
        datasets['val']['embeddings'],
        datasets['val']['targets'],
        datasets['val']['masks']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"\n📊 Dataset sizes:")
    print(f"  Train: {len(train_dataset)}")
    print(f"  Val:   {len(val_dataset)}")
    
    # Initialize model (Trial 84 architecture)
    model = FlexiblePKModel(
        input_dim=788,
        hidden_dims=[384, 1024, 640],
        output_dim=3,
        dropout=0.492,
        use_batch_norm=False,
        activation='gelu'
    ).to(device)
    
    print(f"\n🤖 Model: Trial 84 architecture")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # Loss function
    criterion = nn.MSELoss(reduction='none')
    
    # Training loop
    best_val_r2 = -np.inf
    best_epoch = 0
    patience = 30
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_r2': [],
        'val_metrics': []
    }
    
    print(f"\n🎯 Starting training...")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Early stopping patience: {patience}")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch in train_loader:
            embeddings = batch['embedding'].to(device)
            outputs = model(embeddings)
            
            # Compute loss for each parameter
            loss = 0
            for i, param in enumerate(['fu', 'vd', 'clearance']):
                targets = batch['targets'][param].to(device)
                masks = batch['masks'][param].to(device)
                
                param_loss = criterion(outputs[:, i], targets)
                param_loss = (param_loss * masks).sum() / (masks.sum() + 1e-10)
                
                loss += param_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        
        # Validation
        val_metrics = evaluate(model, val_loader, 'val')
        val_r2 = val_metrics['overall']['r2']
        
        scheduler.step(val_r2)
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_r2'].append(val_r2)
        history['val_metrics'].append(val_metrics)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\nEpoch {epoch+1}/{num_epochs}:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val R²: {val_r2:.4f}")
            for param in ['fu', 'vd', 'clearance']:
                if param in val_metrics:
                    print(f"    {param.upper()}: R²={val_metrics[param]['r2']:.4f}, MAE={val_metrics[param]['mae']:.4f}")
        
        # Save best model
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_r2': val_r2,
                'val_metrics': val_metrics
            }, OUTPUT_DIR / 'best_model_expanded.pt')
            
            print(f"  🎉 New best model saved! R² = {val_r2:.4f}")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏸️  Early stopping at epoch {epoch+1}")
            break
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'history': history
    }, OUTPUT_DIR / 'final_model_expanded.pt')
    
    # Save training history
    with open(OUTPUT_DIR / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✅ Training complete!")
    print(f"  Best epoch: {best_epoch+1}")
    print(f"  Best Val R²: {best_val_r2:.4f}")
    
    return model, history


def main():
    """Main execution"""
    
    print("="*80)
    print("🚀 TRAIN PBPK MODEL WITH EXPANDED DATASET")
    print("="*80)
    
    # Load data
    datasets = load_data()
    
    # Train model
    model, history = train_model(
        datasets,
        num_epochs=200,
        batch_size=64,
        learning_rate=0.0003
    )
    
    # Final evaluation on test set
    print("\n" + "="*80)
    print("📊 FINAL TEST SET EVALUATION")
    print("="*80)
    
    # Load best model
    checkpoint = torch.load(OUTPUT_DIR / 'best_model_expanded.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    test_dataset = PKDataset(
        datasets['test']['embeddings'],
        datasets['test']['targets'],
        datasets['test']['masks']
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    test_metrics = evaluate(model, test_loader, 'test')
    
    print(f"\n✅ TEST SET RESULTS:")
    print(f"  Overall R²: {test_metrics['overall']['r2']:.4f}")
    
    for param in ['fu', 'vd', 'clearance']:
        if param in test_metrics:
            m = test_metrics[param]
            print(f"\n  {param.upper()}:")
            print(f"    R²:        {m['r2']:.4f}")
            print(f"    MAE:       {m['mae']:.4f}")
            print(f"    RMSE:      {m['rmse']:.4f}")
            print(f"    2-fold:    {m['fold2_acc']:.1f}%")
            print(f"    Samples:   {m['n_samples']}")
    
    # Save test results
    with open(OUTPUT_DIR / 'test_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    # Compare with baseline
    print("\n" + "="*80)
    print("📈 COMPARISON WITH BASELINE")
    print("="*80)
    
    baseline_r2 = 0.209
    improvement = ((test_metrics['overall']['r2'] - baseline_r2) / baseline_r2) * 100
    
    print(f"\n  Baseline (Trial 84):   R² = {baseline_r2:.4f}")
    print(f"  Expanded Dataset:      R² = {test_metrics['overall']['r2']:.4f}")
    print(f"  Improvement:           {improvement:+.1f}%")
    
    if test_metrics['overall']['r2'] >= 0.30:
        print(f"\n  🎉 TARGET ACHIEVED! R² ≥ 0.30!")
    else:
        print(f"\n  ⚠️  Target not reached (R² < 0.30)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

