#!/usr/bin/env python3
"""
🚀 PHASE 2.1: Baseline MLP with ChemBERTa Embeddings
=====================================================

Simple Multi-Layer Perceptron using pre-computed ChemBERTa embeddings.
Multi-task learning for Fu, Vd, and Clearance.

Target: R² > 0.30

Author: Dr. Demetrios Chiuratto Agourakis
Date: October 28, 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 BASELINE MLP - CHEMBERTA EMBEDDINGS")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 256
LEARNING_RATE = 3e-4  # Reduced for stability
NUM_EPOCHS = 150  # More epochs
PATIENCE = 25  # More patience
HIDDEN_DIMS = [512, 256, 128]
DROPOUT = 0.3

print(f"\n⚙️  Configuration:")
print(f"   Device: {DEVICE}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Hidden dims: {HIDDEN_DIMS}")

if DEVICE.type == 'cuda':
    print(f"\n🔥 GPU Info:")
    print(f"   Name: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"   Memory: {props.total_memory / (1024**3):.1f} GB")

# Paths
EMBEDDINGS_FILE = Path('data/processed/embeddings/chemberta_768d/chemberta_embeddings_consolidated.npz')
TRAIN_FILE = Path('data/processed/splits/train.parquet')
VAL_FILE = Path('data/processed/splits/val.parquet')
TEST_FILE = Path('data/processed/splits/test.parquet')
OUTPUT_DIR = Path('models/baseline_mlp')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================
print(f"\n📁 Loading data...")

# Load embeddings
embeddings_data = np.load(EMBEDDINGS_FILE)
embeddings = embeddings_data['embeddings']
print(f"  ✓ Loaded embeddings: {embeddings.shape}")

# Load splits
df_train = pd.read_parquet(TRAIN_FILE)
df_val = pd.read_parquet(VAL_FILE)
df_test = pd.read_parquet(TEST_FILE)

print(f"  ✓ Train: {len(df_train):,} samples")
print(f"  ✓ Val:   {len(df_val):,} samples")
print(f"  ✓ Test:  {len(df_test):,} samples")

# ============================================================================
# TRANSFORMS (Simple Log)
# ============================================================================
def logit_transform(x):
    """Logit transform for fu (0-1 range)"""
    x_clipped = np.clip(x, 1e-6, 1 - 1e-6)
    return np.log(x_clipped / (1 - x_clipped))

def inverse_logit_transform(y):
    """Inverse logit transform"""
    return 1 / (1 + np.exp(-y))

def log1p_transform(x):
    """Log1p transform for Vd and Clearance"""
    return np.log1p(x)

def inverse_log1p_transform(y):
    """Inverse log1p transform"""
    return np.expm1(y)

# ============================================================================
# DATASET
# ============================================================================
class PBPKDataset(Dataset):
    def __init__(self, df, embeddings):
        self.df = df.reset_index(drop=True)
        self.embeddings = embeddings
        
        # Extract and transform targets
        self.fu = df['fu'].values
        self.vd = df['vd'].values
        self.clearance = df['clearance'].values
        
        # Transform
        self.fu_transformed = np.array([
            logit_transform(x) if not np.isnan(x) else 0.0 
            for x in self.fu
        ], dtype=np.float32)
        
        self.vd_transformed = np.array([
            log1p_transform(x) if not np.isnan(x) else 0.0 
            for x in self.vd
        ], dtype=np.float32)
        
        self.clearance_transformed = np.array([
            log1p_transform(x) if not np.isnan(x) else 0.0 
            for x in self.clearance
        ], dtype=np.float32)
        
        # Masks
        self.fu_mask = ~np.isnan(self.fu)
        self.vd_mask = ~np.isnan(self.vd)
        self.clearance_mask = ~np.isnan(self.clearance)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        embedding = torch.FloatTensor(self.embeddings[idx])
        
        targets = torch.FloatTensor([
            self.fu_transformed[idx],
            self.vd_transformed[idx],
            self.clearance_transformed[idx]
        ])
        
        masks = torch.BoolTensor([
            self.fu_mask[idx],
            self.vd_mask[idx],
            self.clearance_mask[idx]
        ])
        
        return embedding, targets, masks

# Create datasets
print(f"\n📦 Creating datasets...")
train_dataset = PBPKDataset(df_train, embeddings[:len(df_train)])
val_dataset = PBPKDataset(df_val, embeddings[len(df_train):len(df_train)+len(df_val)])
test_dataset = PBPKDataset(df_test, embeddings[len(df_train)+len(df_val):])

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"  ✓ Train batches: {len(train_loader)}")
print(f"  ✓ Val batches:   {len(val_loader)}")
print(f"  ✓ Test batches:  {len(test_loader)}")

# ============================================================================
# MODEL
# ============================================================================
class BaselineMLP(nn.Module):
    def __init__(self, input_dim=768, hidden_dims=[512, 256, 128], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Task-specific heads
        self.fu_head = nn.Linear(prev_dim, 1)
        self.vd_head = nn.Linear(prev_dim, 1)
        self.clearance_head = nn.Linear(prev_dim, 1)
        
    def forward(self, x):
        features = self.encoder(x)
        
        fu_pred = self.fu_head(features).squeeze(-1)
        vd_pred = self.vd_head(features).squeeze(-1)
        clearance_pred = self.clearance_head(features).squeeze(-1)
        
        return torch.stack([fu_pred, vd_pred, clearance_pred], dim=1)

model = BaselineMLP(
    input_dim=768,
    hidden_dims=HIDDEN_DIMS,
    dropout=DROPOUT
).to(DEVICE)

print(f"\n🤖 Model architecture:")
print(model)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n  Total parameters: {total_params:,}")
print(f"  Trainable parameters: {trainable_params:,}")

# ============================================================================
# TRAINING
# ============================================================================
criterion = nn.MSELoss(reduction='none')
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

def compute_r2(y_true, y_pred):
    """Compute R² score"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    return r2

def evaluate(model, dataloader):
    """Evaluate model"""
    model.eval()
    
    all_preds = [[], [], []]  # fu, vd, clearance
    all_targets = [[], [], []]
    total_loss = 0
    
    with torch.no_grad():
        for embeddings, targets, masks in dataloader:
            embeddings = embeddings.to(DEVICE)
            targets = targets.to(DEVICE)
            masks = masks.to(DEVICE)
            
            predictions = model(embeddings)
            
            # Compute loss
            losses = criterion(predictions, targets)
            masked_losses = losses * masks.float()
            loss = masked_losses.sum() / masks.sum()
            total_loss += loss.item()
            
            # Collect predictions and targets
            predictions_cpu = predictions.cpu().numpy()
            targets_cpu = targets.cpu().numpy()
            masks_cpu = masks.cpu().numpy()
            
            for i in range(3):
                valid_mask = masks_cpu[:, i]
                all_preds[i].extend(predictions_cpu[valid_mask, i])
                all_targets[i].extend(targets_cpu[valid_mask, i])
    
    avg_loss = total_loss / len(dataloader)
    
    # Compute R² for each task
    r2_scores = {}
    for i, task in enumerate(['fu', 'vd', 'clearance']):
        if len(all_targets[i]) > 0:
            r2_scores[task] = compute_r2(all_targets[i], all_preds[i])
        else:
            r2_scores[task] = 0.0
    
    avg_r2 = np.mean(list(r2_scores.values()))
    
    return avg_loss, r2_scores, avg_r2

print(f"\n🏋️  Training...")
print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val R² (Fu)':>12} | {'Val R² (Vd)':>12} | {'Val R² (CL)':>12} | {'Avg R²':>8}")
print("-" * 100)

best_val_r2 = -float('inf')
patience_counter = 0
history = []

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    train_loss = 0
    
    for embeddings, targets, masks in train_loader:
        embeddings = embeddings.to(DEVICE)
        targets = targets.to(DEVICE)
        masks = masks.to(DEVICE)
        
        optimizer.zero_grad()
        predictions = model(embeddings)
        
        losses = criterion(predictions, targets)
        
        # Weighted loss: give more weight to tasks with more data
        # Clearance: 90%, Vd: 19%, Fu: 18%
        task_weights = torch.tensor([1.0, 1.5, 3.0], device=predictions.device)  # Fu, Vd, CL weights
        weighted_losses = losses * masks.float() * task_weights
        
        loss = weighted_losses.sum() / (masks.float() * task_weights).sum()
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    val_loss, val_r2_scores, avg_val_r2 = evaluate(model, val_loader)
    
    # Learning rate scheduling
    scheduler.step(avg_val_r2)
    
    # Print progress
    print(f"{epoch+1:5d} | {avg_train_loss:10.6f} | {val_loss:10.6f} | "
          f"{val_r2_scores['fu']:12.6f} | {val_r2_scores['vd']:12.6f} | "
          f"{val_r2_scores['clearance']:12.6f} | {avg_val_r2:8.6f}")
    
    # Save history
    history.append({
        'epoch': epoch + 1,
        'train_loss': avg_train_loss,
        'val_loss': val_loss,
        'val_r2_fu': val_r2_scores['fu'],
        'val_r2_vd': val_r2_scores['vd'],
        'val_r2_clearance': val_r2_scores['clearance'],
        'avg_val_r2': avg_val_r2
    })
    
    # Early stopping
    if avg_val_r2 > best_val_r2:
        best_val_r2 = avg_val_r2
        patience_counter = 0
        
        # Save best model
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_r2': avg_val_r2,
            'val_r2_scores': val_r2_scores
        }, OUTPUT_DIR / 'best_model.pt')
    else:
        patience_counter += 1
        
    if patience_counter >= PATIENCE:
        print(f"\n⏸️  Early stopping at epoch {epoch + 1}")
        break

print(f"\n✅ Training complete!")
print(f"   Best validation R²: {best_val_r2:.6f}")

# ============================================================================
# TEST EVALUATION
# ============================================================================
print(f"\n📊 Evaluating on test set...")

# Load best model
checkpoint = torch.load(OUTPUT_DIR / 'best_model.pt', weights_only=False)
model.load_state_dict(checkpoint['model_state_dict'])

test_loss, test_r2_scores, avg_test_r2 = evaluate(model, test_loader)

print(f"\n🎯 Test Results:")
print(f"   Loss: {test_loss:.6f}")
print(f"   R² Fu:        {test_r2_scores['fu']:.6f}")
print(f"   R² Vd:        {test_r2_scores['vd']:.6f}")
print(f"   R² Clearance: {test_r2_scores['clearance']:.6f}")
print(f"   Avg R²:       {avg_test_r2:.6f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
results = {
    'model': 'BaselineMLP',
    'input_features': 'ChemBERTa-768d',
    'hidden_dims': HIDDEN_DIMS,
    'dropout': DROPOUT,
    'batch_size': BATCH_SIZE,
    'learning_rate': LEARNING_RATE,
    'total_epochs': epoch + 1,
    'best_epoch': checkpoint['epoch'],
    'best_val_r2': float(best_val_r2),
    'test_loss': float(test_loss),
    'test_r2_fu': float(test_r2_scores['fu']),
    'test_r2_vd': float(test_r2_scores['vd']),
    'test_r2_clearance': float(test_r2_scores['clearance']),
    'avg_test_r2': float(avg_test_r2),
    'total_parameters': total_params,
    'trainable_parameters': trainable_params,
    'timestamp': datetime.now().isoformat()
}

with open(OUTPUT_DIR / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

with open(OUTPUT_DIR / 'training_history.json', 'w') as f:
    json.dump(history, f, indent=2)

print(f"\n💾 Saved:")
print(f"   Model: {OUTPUT_DIR / 'best_model.pt'}")
print(f"   Results: {OUTPUT_DIR / 'results.json'}")
print(f"   History: {OUTPUT_DIR / 'training_history.json'}")

print("\n" + "="*80)
print("🎉 BASELINE MLP COMPLETE!")
print("="*80)
print(f"\n✅ Final Test R²: {avg_test_r2:.6f}")
print(f"   {'Target':>15}: R² > 0.30")
print(f"   {'Achieved':>15}: R² = {avg_test_r2:.6f}")
print(f"   {'Status':>15}: {'✅ SUCCESS' if avg_test_r2 > 0.30 else '⚠️  BELOW TARGET'}")

