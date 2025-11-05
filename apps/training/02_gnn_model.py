#!/usr/bin/env python3
"""
🚀 PHASE 2.2: GNN Model (GAT + TransformerConv)
================================================

Graph Neural Network using PyTorch Geometric.
Architecture: GAT + TransformerConv + Global Attention Pooling
Multi-task learning for Fu, Vd, and Clearance.

Target: R² > 0.45

Author: Dr. Demetrios Chiuratto Agourakis
Date: October 28, 2025
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, TransformerConv, global_mean_pool, global_add_pool
from torch_geometric.loader import DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("🚀 GNN MODEL - GAT + TRANSFORMERCONV")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 128  # Smaller for GNN (memory intensive)
LEARNING_RATE = 1e-4  # Reduced for stability
NUM_EPOCHS = 200  # More epochs
PATIENCE = 30  # More patience
HIDDEN_DIM = 256
NUM_HEADS = 4
NUM_LAYERS = 3
DROPOUT = 0.2

print(f"\n⚙️  Configuration:")
print(f"   Device: {DEVICE}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Hidden dim: {HIDDEN_DIM}")
print(f"   Attention heads: {NUM_HEADS}")
print(f"   GNN layers: {NUM_LAYERS}")

if DEVICE.type == 'cuda':
    print(f"\n🔥 GPU Info:")
    print(f"   Name: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"   Memory: {props.total_memory / (1024**3):.1f} GB")

# Paths
GRAPHS_FILE = Path('data/processed/molecular_graphs/molecular_graphs.pkl')
TRAIN_FILE = Path('data/processed/splits/train.parquet')
VAL_FILE = Path('data/processed/splits/val.parquet')
TEST_FILE = Path('data/processed/splits/test.parquet')
OUTPUT_DIR = Path('models/gnn_model')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# LOAD DATA
# ============================================================================
print(f"\n📁 Loading data...")

# Load molecular graphs
with open(GRAPHS_FILE, 'rb') as f:
    all_graphs = pickle.load(f)
print(f"  ✓ Loaded {len(all_graphs):,} molecular graphs")

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

def log1p_transform(x):
    """Log1p transform for Vd and Clearance"""
    return np.log1p(x)

# ============================================================================
# DATASET
# ============================================================================
def prepare_graph_dataset(df, graphs):
    """Prepare graph dataset with PBPK targets"""
    dataset = []
    
    for idx in range(len(df)):
        # Create a copy to avoid modifying original
        graph = graphs[idx].clone()
        
        # Get targets
        fu = df.iloc[idx]['fu']
        vd = df.iloc[idx]['vd']
        clearance = df.iloc[idx]['clearance']
        
        # Transform
        fu_transformed = logit_transform(fu) if not np.isnan(fu) else 0.0
        vd_transformed = log1p_transform(vd) if not np.isnan(vd) else 0.0
        clearance_transformed = log1p_transform(clearance) if not np.isnan(clearance) else 0.0
        
        # Masks
        fu_mask = not np.isnan(fu)
        vd_mask = not np.isnan(vd)
        clearance_mask = not np.isnan(clearance)
        
        # Add to graph as graph-level attributes (not node-level!)
        graph.y = torch.FloatTensor([[fu_transformed, vd_transformed, clearance_transformed]])
        graph.mask = torch.BoolTensor([[fu_mask, vd_mask, clearance_mask]])
        
        dataset.append(graph)
    
    return dataset

print(f"\n📦 Preparing graph datasets...")
train_graphs = prepare_graph_dataset(df_train, all_graphs[:len(df_train)])
val_graphs = prepare_graph_dataset(df_val, all_graphs[len(df_train):len(df_train)+len(df_val)])
test_graphs = prepare_graph_dataset(df_test, all_graphs[len(df_train)+len(df_val):])

# Create dataloaders
train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_graphs, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

print(f"  ✓ Train batches: {len(train_loader)}")
print(f"  ✓ Val batches:   {len(val_loader)}")
print(f"  ✓ Test batches:  {len(test_loader)}")

# ============================================================================
# MODEL
# ============================================================================
class GNNModel(nn.Module):
    def __init__(self, node_features=10, edge_features=4, hidden_dim=256, 
                 num_heads=4, num_layers=3, dropout=0.2):
        super().__init__()
        
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.edge_embedding = nn.Linear(edge_features, hidden_dim)
        
        # GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                       dropout=dropout, edge_dim=hidden_dim)
            )
        
        # TransformerConv layers
        self.transformer_layers = nn.ModuleList()
        for i in range(num_layers):
            self.transformer_layers.append(
                TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads,
                              dropout=dropout, edge_dim=hidden_dim)
            )
        
        # Batch norm layers
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers * 2)
        ])
        
        # Attention pooling
        self.attention_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # MLP for final prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 because we concat mean and add pooling
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Task-specific heads
        self.fu_head = nn.Linear(hidden_dim // 2, 1)
        self.vd_head = nn.Linear(hidden_dim // 2, 1)
        self.clearance_head = nn.Linear(hidden_dim // 2, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Embed nodes and edges
        x = self.node_embedding(x)
        edge_attr_embedded = self.edge_embedding(edge_attr)
        
        # GAT layers
        layer_idx = 0
        for gat_layer in self.gat_layers:
            x_residual = x
            x = gat_layer(x, edge_index, edge_attr_embedded)
            x = self.batch_norms[layer_idx](x)
            x = torch.relu(x)
            x = self.dropout(x)
            x = x + x_residual  # Residual connection
            layer_idx += 1
        
        # TransformerConv layers
        for transformer_layer in self.transformer_layers:
            x_residual = x
            x = transformer_layer(x, edge_index, edge_attr_embedded)
            x = self.batch_norms[layer_idx](x)
            x = torch.relu(x)
            x = self.dropout(x)
            x = x + x_residual  # Residual connection
            layer_idx += 1
        
        # Attention-based pooling
        attention_weights = self.attention_gate(x)
        x_attention = x * attention_weights
        
        # Global pooling (both mean and add)
        x_mean = global_mean_pool(x_attention, batch)
        x_add = global_add_pool(x_attention, batch)
        x_pooled = torch.cat([x_mean, x_add], dim=1)
        
        # MLP
        features = self.mlp(x_pooled)
        
        # Task-specific predictions
        fu_pred = self.fu_head(features).squeeze(-1)
        vd_pred = self.vd_head(features).squeeze(-1)
        clearance_pred = self.clearance_head(features).squeeze(-1)
        
        return torch.stack([fu_pred, vd_pred, clearance_pred], dim=1)

# Get node and edge feature dimensions from first graph
example_graph = train_graphs[0]
node_features = example_graph.x.shape[1]
edge_features = example_graph.edge_attr.shape[1]

print(f"\n📊 Graph statistics:")
print(f"   Node features: {node_features}")
print(f"   Edge features: {edge_features}")

model = GNNModel(
    node_features=node_features,
    edge_features=edge_features,
    hidden_dim=HIDDEN_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
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
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)

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
        for batch in dataloader:
            batch = batch.to(DEVICE)
            
            predictions = model(batch)
            
            # Reshape batch.y and batch.mask
            targets = batch.y.squeeze(1)
            masks = batch.mask.squeeze(1)
            
            # Compute loss
            losses = criterion(predictions, targets)
            task_weights = torch.tensor([1.0, 1.5, 3.0], device=predictions.device)
            weighted_losses = losses * masks.float() * task_weights
            loss = weighted_losses.sum() / (masks.float() * task_weights).sum()
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

print(f"\n🏋️  Training GNN...")
print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Val Loss':>10} | {'Val R² (Fu)':>12} | {'Val R² (Vd)':>12} | {'Val R² (CL)':>12} | {'Avg R²':>8}")
print("-" * 100)

best_val_r2 = -float('inf')
patience_counter = 0
history = []

for epoch in range(NUM_EPOCHS):
    # Training
    model.train()
    train_loss = 0
    
    for batch in train_loader:
        batch = batch.to(DEVICE)
        
        optimizer.zero_grad()
        predictions = model(batch)
        
        # Reshape batch.y and batch.mask to match predictions
        targets = batch.y.squeeze(1)  # (batch_size, 3)
        masks = batch.mask.squeeze(1)  # (batch_size, 3)
        
        losses = criterion(predictions, targets)
        
        # Weighted loss: give more weight to tasks with more data
        task_weights = torch.tensor([1.0, 1.5, 3.0], device=predictions.device)
        weighted_losses = losses * masks.float() * task_weights
        
        loss = weighted_losses.sum() / (masks.float() * task_weights).sum()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
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
    'model': 'GNN (GAT + TransformerConv)',
    'node_features': int(node_features),
    'edge_features': int(edge_features),
    'hidden_dim': HIDDEN_DIM,
    'num_heads': NUM_HEADS,
    'num_layers': NUM_LAYERS,
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
print("🎉 GNN MODEL COMPLETE!")
print("="*80)
print(f"\n✅ Final Test R²: {avg_test_r2:.6f}")
print(f"   {'Target':>15}: R² > 0.45")
print(f"   {'Achieved':>15}: R² = {avg_test_r2:.6f}")
print(f"   {'Status':>15}: {'✅ SUCCESS' if avg_test_r2 > 0.45 else '⚠️  BELOW TARGET'}")

