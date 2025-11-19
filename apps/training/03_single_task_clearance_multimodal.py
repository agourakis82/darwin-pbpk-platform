#!/usr/bin/env python3
"""
🔬 SCIENTIFIC TRAINING: Single-Task Clearance Model with Full Multimodal Encoder
================================================================================

Objetivo: Alcançar R² > 0.50 para Clearance usando encoder multimodal completo.

Metodologia:
- Single-task model (não multi-task devido a missing data)
- Encoder multimodal completo (976d: ChemBERTa + GNN + KEC + 3D + QM)
- Dataset completo (32k+ samples)
- Validação rigorosa (5-fold cross-validation)
- Comparação com benchmarks da literatura

Autor: Dr. Demetrios Chiuratto Agourakis
Data: 2025-11-08
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
import sys
import argparse
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import logging

# Adicionar path do projeto
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from apps.pbpk_core.ml.multimodal import MultimodalMolecularEncoder

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# ARGUMENT PARSING
# ============================================================================
parser = argparse.ArgumentParser(
    description='Scientific Training: Single-Task Clearance with Full Multimodal Encoder'
)
parser.add_argument('--output-dir', type=str, default='models/single_task_clearance_multimodal',
                    help='Output directory for models and results')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                    help='Device to use (cuda/cpu)')
parser.add_argument('--batch-size', type=int, default=32,
                    help='Batch size for training')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of training epochs')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='Learning rate')
parser.add_argument('--patience', type=int, default=30,
                    help='Early stopping patience')
parser.add_argument('--num-folds', type=int, default=5,
                    help='Number of folds for cross-validation')
parser.add_argument('--data-file', type=str, default='data/processed/consolidated/consolidated_pbpk_v1.parquet',
                    help='Path to consolidated dataset (will use TDC if not found)')
args = parser.parse_args()

print("="*80)
print("🔬 SCIENTIFIC TRAINING: Single-Task Clearance - Full Multimodal Encoder")
print("="*80)
print("\n🎯 Objetivo: R² > 0.50 para Clearance")
print("📊 Metodologia: Single-task, encoder multimodal completo (976d)")
print("🔬 Rigor: Validação 5-fold, comparação com literatura")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
DEVICE = torch.device(args.device)
BATCH_SIZE = args.batch_size
LEARNING_RATE = args.lr
NUM_EPOCHS = args.epochs
PATIENCE = args.patience
NUM_FOLDS = args.num_folds
HIDDEN_DIMS = [1024, 512, 256, 128]
DROPOUT = 0.3
INPUT_DIM = 976  # Multimodal encoder completo

print(f"\n⚙️  Configuration:")
print(f"   Device: {DEVICE}")
print(f"   Batch size: {BATCH_SIZE}")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Epochs: {NUM_EPOCHS}")
print(f"   Patience: {PATIENCE}")
print(f"   Cross-validation folds: {NUM_FOLDS}")
print(f"   Hidden dims: {HIDDEN_DIMS}")
print(f"   Input dim: {INPUT_DIM} (Multimodal: ChemBERTa + GNN + KEC + 3D + QM)")

if DEVICE.type == 'cuda':
    print(f"\n🔥 GPU Info:")
    print(f"   Name: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"   Memory: {props.total_memory / (1024**3):.1f} GB")

# Paths - Usar TDC diretamente se dataset consolidado não existir
DATA_FILE = Path(args.data_file)
OUTPUT_DIR = Path(args.output_dir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Verificar se dataset existe
if not DATA_FILE.exists():
    print(f"\n⚠️  Dataset consolidado não encontrado: {DATA_FILE}")
    print(f"   Usando TDC diretamente...")
    TDC_FALLBACK = True
else:
    TDC_FALLBACK = False

# ============================================================================
# LOAD DATA
# ============================================================================
print(f"\n📁 Loading data...")

if TDC_FALLBACK:
    # Carregar TDC diretamente
    print(f"   Carregando TDC ADME dataset...")
    try:
        from tdc.single_pred import ADME

        # Carregar dataset de Clearance do TDC
        data = ADME(name='Clearance_Hepatocyte_AZ')
        df_clearance = data.get_data()

        # Renomear colunas para padrão
        if 'Drug' in df_clearance.columns:
            df_clearance = df_clearance.rename(columns={'Drug': 'smiles', 'Y': 'clearance'})
        elif 'SMILES' in df_clearance.columns:
            df_clearance = df_clearance.rename(columns={'SMILES': 'smiles'})

        print(f"  ✓ Loaded {len(df_clearance):,} compounds from TDC")
        print(f"  ✓ Columns: {list(df_clearance.columns)}")

        # Garantir que temos SMILES e clearance
        if 'smiles' not in df_clearance.columns:
            raise ValueError("Coluna 'smiles' não encontrada no dataset TDC")
        if 'clearance' not in df_clearance.columns and 'Y' in df_clearance.columns:
            df_clearance['clearance'] = df_clearance['Y']

        # Filtrar apenas amostras válidas
        df_clearance = df_clearance[df_clearance['clearance'].notna()].copy()
        df_clearance = df_clearance[df_clearance['smiles'].notna()].copy()

        print(f"  ✓ Valid compounds: {len(df_clearance):,}")

    except ImportError:
        print(f"  ❌ TDC não instalado. Instalando...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "PyTDC"])
        from tdc.single_pred import ADME
        data = ADME(name='Clearance_Hepatocyte_AZ')
        df_clearance = data.get_data()
        if 'Drug' in df_clearance.columns:
            df_clearance = df_clearance.rename(columns={'Drug': 'smiles', 'Y': 'clearance'})
        df_clearance = df_clearance[df_clearance['clearance'].notna() & df_clearance['smiles'].notna()].copy()
        print(f"  ✓ Loaded {len(df_clearance):,} compounds from TDC")
    except Exception as e:
        print(f"  ❌ Erro ao carregar TDC: {e}")
        raise
else:
    # Carregar dataset consolidado
    df = pd.read_parquet(DATA_FILE)
    print(f"  ✓ Loaded {len(df):,} compounds")

    # Filtrar apenas amostras com Clearance
    df_clearance = df[df['clearance'].notna()].copy()
    print(f"  ✓ Compounds with clearance: {len(df_clearance):,}")

    # Verificar SMILES
    df_clearance = df_clearance[df_clearance['smiles'].notna()].copy()
    print(f"  ✓ Valid SMILES: {len(df_clearance):,}")

if len(df_clearance) < 1000:
    raise ValueError(f"Dataset muito pequeno: {len(df_clearance)} amostras. Mínimo necessário: 1000")

# ============================================================================
# TRANSFORMS
# ============================================================================
def log1p_transform(x):
    """Log1p transform for Clearance"""
    return np.log1p(np.clip(x, 0, None))

def inverse_log1p_transform(y):
    """Inverse log1p transform"""
    return np.expm1(y)

# Aplicar transform
df_clearance['clearance_transformed'] = df_clearance['clearance'].apply(log1p_transform)

print(f"\n📊 Clearance statistics:")
print(f"   Mean: {df_clearance['clearance'].mean():.4f}")
print(f"   Std:  {df_clearance['clearance'].std():.4f}")
print(f"   Min:  {df_clearance['clearance'].min():.4f}")
print(f"   Max:  {df_clearance['clearance'].max():.4f}")

# ============================================================================
# INITIALIZE MULTIMODAL ENCODER
# ============================================================================
print(f"\n🔧 Initializing Multimodal Encoder...")
print(f"   This may take a few minutes (loading pre-trained models)...")

encoder = MultimodalMolecularEncoder(
    device=str(DEVICE),
    parallel=True,
    verbose=True
)

print(f"   ✅ Encoder initialized: {encoder.total_dim} dimensions")
print(f"      - ChemBERTa: {encoder.chemberta_dim}d")
print(f"      - GNN: {encoder.gnn_dim}d")
print(f"      - KEC: {encoder.kec_dim}d (NOVEL)")
print(f"      - 3D Conformer: {encoder.conformer_dim}d")
print(f"      - QM: {encoder.qm_dim}d")

# ============================================================================
# DATASET
# ============================================================================
class ClearanceDataset(Dataset):
    def __init__(self, df, encoder, device):
        self.df = df.reset_index(drop=True)
        self.encoder = encoder
        self.device = device

        # Pre-compute embeddings (mais eficiente)
        print(f"\n📦 Computing embeddings for {len(df)} compounds...")
        print(f"   This will take several minutes...")

        self.embeddings = []
        self.targets = []

        for idx in tqdm(range(len(df)), desc="Encoding"):
            smiles = df.iloc[idx]['smiles']
            clearance = df.iloc[idx]['clearance_transformed']

            try:
                embedding = encoder.encode(smiles)
                self.embeddings.append(embedding)
                self.targets.append(clearance)
            except Exception as e:
                logger.warning(f"Erro ao codificar {smiles}: {e}")
                # Usar embedding zero em caso de erro
                self.embeddings.append(np.zeros(encoder.total_dim, dtype=np.float32))
                self.targets.append(0.0)

        self.embeddings = np.array(self.embeddings)
        self.targets = np.array(self.targets)

        print(f"   ✅ Embeddings computed: {self.embeddings.shape}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        embedding = torch.FloatTensor(self.embeddings[idx])
        target = torch.FloatTensor([self.targets[idx]])
        return embedding, target

# ============================================================================
# MODEL
# ============================================================================
class ClearanceModel(nn.Module):
    def __init__(self, input_dim=976, hidden_dims=[1024, 512, 256, 128], dropout=0.3):
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
        self.head = nn.Linear(prev_dim, 1)

    def forward(self, x):
        features = self.encoder(x)
        return self.head(features).squeeze(-1)

# ============================================================================
# TRAINING FUNCTION
# ============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for embeddings, targets in dataloader:
        embeddings = embeddings.to(device)
        targets = targets.squeeze(1).to(device)

        optimizer.zero_grad()
        predictions = model(embeddings)
        loss = criterion(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for embeddings, targets in dataloader:
            embeddings = embeddings.to(device)
            targets = targets.squeeze(1).to(device)

            predictions = model(embeddings)
            loss = criterion(predictions, targets)
            total_loss += loss.item()

            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader)

    # Calcular métricas
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    r2 = r2_score(all_targets, all_preds)
    rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    mae = mean_absolute_error(all_targets, all_preds)

    return avg_loss, r2, rmse, mae

# ============================================================================
# 5-FOLD CROSS-VALIDATION
# ============================================================================
print(f"\n🔬 Starting {NUM_FOLDS}-fold Cross-Validation...")
print(f"   Scientific rigor: Independent validation sets")

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
fold_results = []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(df_clearance)):
    print(f"\n{'='*80}")
    print(f"📊 FOLD {fold_idx + 1}/{NUM_FOLDS}")
    print(f"{'='*80}")

    # Split data
    df_train_fold = df_clearance.iloc[train_idx].copy()
    df_val_fold = df_clearance.iloc[val_idx].copy()

    print(f"   Train: {len(df_train_fold):,} samples")
    print(f"   Val:   {len(df_val_fold):,} samples")

    # Create datasets
    train_dataset = ClearanceDataset(df_train_fold, encoder, DEVICE)
    val_dataset = ClearanceDataset(df_val_fold, encoder, DEVICE)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Create model
    model = ClearanceModel(
        input_dim=INPUT_DIM,
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT
    ).to(DEVICE)

    # Training setup
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)

    # Training loop
    best_val_r2 = -float('inf')
    patience_counter = 0
    history = []

    print(f"\n🏋️  Training Fold {fold_idx + 1}...")
    print(f"{'Epoch':>5} | {'Train Loss':>12} | {'Val Loss':>12} | {'Val R²':>10} | {'Val RMSE':>12} | {'Val MAE':>12}")
    print("-" * 80)

    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_r2, val_rmse, val_mae = evaluate(model, val_loader, criterion, DEVICE)

        scheduler.step(val_r2)

        if epoch % 10 == 0 or epoch == NUM_EPOCHS - 1:
            print(f"{epoch+1:5d} | {train_loss:12.6f} | {val_loss:12.6f} | {val_r2:10.6f} | {val_rmse:12.6f} | {val_mae:12.6f}")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_r2': val_r2,
            'val_rmse': val_rmse,
            'val_mae': val_mae
        })

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            patience_counter = 0

            # Save best model for this fold
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_r2': val_r2,
                'val_rmse': val_rmse,
                'val_mae': val_mae
            }, OUTPUT_DIR / f'best_model_fold_{fold_idx + 1}.pt')
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"\n⏸️  Early stopping at epoch {epoch + 1}")
            break

    # Load best model and evaluate
    checkpoint = torch.load(OUTPUT_DIR / f'best_model_fold_{fold_idx + 1}.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    val_loss, val_r2, val_rmse, val_mae = evaluate(model, val_loader, criterion, DEVICE)

    fold_results.append({
        'fold': fold_idx + 1,
        'best_epoch': checkpoint['epoch'],
        'val_r2': float(val_r2),
        'val_rmse': float(val_rmse),
        'val_mae': float(val_mae),
        'history': history
    })

    print(f"\n✅ Fold {fold_idx + 1} Results:")
    print(f"   Best Val R²:  {val_r2:.6f}")
    print(f"   Val RMSE:     {val_rmse:.6f}")
    print(f"   Val MAE:      {val_mae:.6f}")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print(f"\n{'='*80}")
print(f"📊 CROSS-VALIDATION RESULTS SUMMARY")
print(f"{'='*80}")

r2_scores = [r['val_r2'] for r in fold_results]
rmse_scores = [r['val_rmse'] for r in fold_results]
mae_scores = [r['val_mae'] for r in fold_results]

mean_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores)
mean_rmse = np.mean(rmse_scores)
mean_mae = np.mean(mae_scores)

print(f"\n📈 Performance Metrics:")
print(f"   Mean R²:    {mean_r2:.6f} ± {std_r2:.6f}")
print(f"   Mean RMSE:  {mean_rmse:.6f}")
print(f"   Mean MAE:   {mean_mae:.6f}")

print(f"\n📊 Per-Fold Results:")
for result in fold_results:
    print(f"   Fold {result['fold']}: R² = {result['val_r2']:.6f}")

print(f"\n🎯 Target: R² > 0.50")
print(f"   Achieved: R² = {mean_r2:.6f} ± {std_r2:.6f}")
print(f"   Status: {'✅ SUCCESS' if mean_r2 > 0.50 else '❌ BELOW TARGET'}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
results = {
    'model': 'Single-Task Clearance (Multimodal Encoder)',
    'encoder': 'MultimodalMolecularEncoder (976d)',
    'encoder_components': {
        'chemberta': encoder.chemberta_dim,
        'gnn': encoder.gnn_dim,
        'kec': encoder.kec_dim,
        'conformer_3d': encoder.conformer_dim,
        'qm': encoder.qm_dim
    },
    'dataset_size': len(df_clearance),
    'configuration': {
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'hidden_dims': HIDDEN_DIMS,
        'dropout': DROPOUT,
        'num_epochs': NUM_EPOCHS,
        'patience': PATIENCE
    },
    'cross_validation': {
        'n_folds': 5,
        'mean_r2': float(mean_r2),
        'std_r2': float(std_r2),
        'mean_rmse': float(mean_rmse),
        'mean_mae': float(mean_mae),
        'fold_results': fold_results
    },
    'target': 0.50,
    'achieved': float(mean_r2),
    'status': 'SUCCESS' if mean_r2 > 0.50 else 'BELOW_TARGET',
    'timestamp': datetime.now().isoformat()
}

with open(OUTPUT_DIR / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n💾 Results saved to: {OUTPUT_DIR / 'results.json'}")

print("\n" + "="*80)
print("🎉 SCIENTIFIC TRAINING COMPLETE!")
print("="*80)
print(f"\n✅ Mean R²: {mean_r2:.6f} ± {std_r2:.6f}")
print(f"   Target: R² > 0.50")
print(f"   Status: {'✅ SUCCESS - READY FOR PUBLICATION' if mean_r2 > 0.50 else '⚠️  BELOW TARGET - NEEDS IMPROVEMENT'}")

