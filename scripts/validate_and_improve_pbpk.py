#!/usr/bin/env python3
"""
🎯 VALIDAÇÃO EXTERNA + MELHORIA DE PBPK
========================================

Sistema completo para:
1. Validar Trial 84 em datasets externos (DrugBank, PK-DB, FDA)
2. Implementar 3 estratégias de melhoria
3. Comparar com benchmarks clínicos (fold error < 2x)
4. Gerar relatório com métricas clínicas

Author: Dr. Demetrios Chiuratto Agourakis
Date: October 28, 2025
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import pickle
from typing import Dict, List, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (15, 10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== PATHS ====================
TRIAL84_MODEL = BASE_DIR / 'results' / 'trial84_evaluation' / 'trial84_best.pt'
KEC_DATA = BASE_DIR / 'data' / 'processed' / 'kec_dataset_split.pkl'
EMBEDDINGS = BASE_DIR / 'data' / 'embeddings' / 'rich_embeddings_788d.npz'

# External validation datasets
EXTERNAL_DATASETS = {
    'drugbank': Path('/mnt/f/datasets/pbpk/drugbank'),
    'pk-db': Path('/mnt/f/datasets/pbpk/pk-db'),
    'fda': Path('/mnt/f/datasets/pbpk/fda_real_data'),
    'validation_100': Path('/mnt/f/datasets/pbpk/validation_drugs_100.json'),
    'tdc': Path('/mnt/f/datasets/pbpk/tdc')
}

OUTPUT_DIR = BASE_DIR / 'results' / 'pbpk_validation_complete'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("🎯 VALIDAÇÃO EXTERNA + MELHORIA DE PBPK")
print("="*80)
print(f"Device: {device}")
print(f"Trial 84: {TRIAL84_MODEL}")
print(f"Output: {OUTPUT_DIR}")
print()


# ==================== MODEL DEFINITION ====================

class FlexiblePKModel(nn.Module):
    """Trial 84 architecture"""
    def __init__(
        self,
        input_dim: int = 788,
        hidden_dims: List[int] = [512, 256, 128],
        output_dim: int = 3,
        dropout: float = 0.3,
        use_residual: bool = True,
        use_attention: bool = True
    ):
        super().__init__()
        self.use_residual = use_residual
        self.use_attention = use_attention
        
        # Encoder
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
        
        # Attention
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=hidden_dims[-1],
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
        
        # Output heads
        self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
    def forward(self, x):
        h = self.encoder(x)
        
        if self.use_attention:
            h_att, _ = self.attention(
                h.unsqueeze(1), 
                h.unsqueeze(1), 
                h.unsqueeze(1)
            )
            h = h_att.squeeze(1)
        
        if self.use_residual and h.shape[-1] == x.shape[-1]:
            h = h + x
        
        out = self.output_layer(h)
        return out


# ==================== DATA LOADING ====================

def load_kec_test_data():
    """Load KEC test set"""
    print("📁 Loading KEC test data...")
    
    # Load split
    with open(KEC_DATA, 'rb') as f:
        data = pickle.load(f)
    
    test_df = data['test']['dataframe']
    test_smiles = test_df['canonical_smiles'].tolist()
    
    # Load embeddings
    embs = np.load(EMBEDDINGS)
    all_embeddings = embs['embeddings']
    
    # Match test embeddings by SMILES
    train_df = data['train']['dataframe']
    val_df = data['val']['dataframe']
    all_smiles = (
        train_df['canonical_smiles'].tolist() +
        val_df['canonical_smiles'].tolist() +
        test_smiles
    )
    
    smiles_to_idx = {s: i for i, s in enumerate(all_smiles)}
    test_indices = [smiles_to_idx[s] for s in test_smiles]
    test_embeddings = all_embeddings[test_indices]
    
    # Extract targets
    y_test = test_df[['fu', 'vd', 'clearance']].values
    
    print(f"  ✓ Test samples: {len(test_df)}")
    print(f"  ✓ Embeddings shape: {test_embeddings.shape}")
    
    return test_embeddings, y_test, test_df


def load_external_validation_set():
    """Load external validation drugs (100 drugs with clinical data)"""
    print("\n📁 Loading external validation set...")
    
    val_file = EXTERNAL_DATASETS['validation_100']
    if not val_file.exists():
        print(f"  ⚠️  File not found: {val_file}")
        return None, None, None
    
    with open(val_file, 'r') as f:
        drugs = json.load(f)
    
    print(f"  ✓ Loaded {len(drugs)} drugs with clinical data")
    
    # Extract SMILES and targets
    smiles_list = []
    fu_list, vd_list, cl_list = [], [], []
    drug_names = []
    
    for drug in drugs:
        if 'smiles' in drug and 'pk_parameters' in drug:
            pk = drug['pk_parameters']
            if 'fu' in pk or 'vd' in pk or 'clearance' in pk:
                smiles_list.append(drug['smiles'])
                fu_list.append(pk.get('fu', np.nan))
                vd_list.append(pk.get('vd', np.nan))
                cl_list.append(pk.get('clearance', np.nan))
                drug_names.append(drug.get('drug_name', 'Unknown'))
    
    if len(smiles_list) == 0:
        print("  ⚠️  No valid drugs found")
        return None, None, None
    
    df = pd.DataFrame({
        'drug_name': drug_names,
        'smiles': smiles_list,
        'fu': fu_list,
        'vd': vd_list,
        'clearance': cl_list
    })
    
    print(f"  ✓ Valid drugs: {len(df)}")
    print(f"  ✓ Fu available: {df['fu'].notna().sum()}")
    print(f"  ✓ Vd available: {df['vd'].notna().sum()}")
    print(f"  ✓ CL available: {df['clearance'].notna().sum()}")
    
    return df, smiles_list, df[['fu', 'vd', 'clearance']].values


# ==================== TRANSFORMS ====================

def apply_transforms_simple(y):
    """Apply simple transforms (logit for fu, log1p for vd/clearance)"""
    y_transformed = np.zeros_like(y)
    
    # Fu: logit
    fu = np.clip(y[:, 0], 1e-6, 1-1e-6)
    fu_transformed = np.log(fu / (1 - fu))
    fu_transformed = np.clip(fu_transformed, -10, 10)
    y_transformed[:, 0] = fu_transformed
    
    # Vd: log1p
    vd_transformed = np.log1p(np.clip(y[:, 1], 0, 1e10))
    vd_transformed = np.clip(vd_transformed, -10, 10)
    y_transformed[:, 1] = vd_transformed
    
    # Clearance: log1p
    cl_transformed = np.log1p(np.clip(y[:, 2], 0, 1e10))
    cl_transformed = np.clip(cl_transformed, -10, 10)
    y_transformed[:, 2] = cl_transformed
    
    return y_transformed


def inverse_transform_simple(y_transformed):
    """Inverse simple transforms"""
    y_original = np.zeros_like(y_transformed)
    
    # Fu: inverse logit
    fu_original = 1 / (1 + np.exp(-y_transformed[:, 0]))
    y_original[:, 0] = np.clip(fu_original, 0, 1)
    
    # Vd: expm1
    y_original[:, 1] = np.expm1(y_transformed[:, 1])
    y_original[:, 1] = np.clip(y_original[:, 1], 0, 1e10)
    
    # Clearance: expm1
    y_original[:, 2] = np.expm1(y_transformed[:, 2])
    y_original[:, 2] = np.clip(y_original[:, 2], 0, 1e10)
    
    return y_original


# ==================== EVALUATION ====================

def compute_clinical_metrics(y_true, y_pred, param_name):
    """Compute clinical evaluation metrics"""
    # Remove NaNs
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) < 5:
        return None
    
    # Basic metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Clinical metrics: Fold error
    fold_error = y_pred / (y_true + 1e-10)
    within_2fold = np.mean((fold_error >= 0.5) & (fold_error <= 2.0))
    within_3fold = np.mean((fold_error >= 1/3) & (fold_error <= 3.0))
    
    # Bias
    bias = np.mean(y_pred - y_true)
    
    # Correlation
    corr, corr_p = stats.pearsonr(y_true, y_pred)
    
    # Bland-Altman
    mean_vals = (y_true + y_pred) / 2
    diff_vals = y_pred - y_true
    ba_bias = np.mean(diff_vals)
    ba_sd = np.std(diff_vals)
    ba_loa_lower = ba_bias - 1.96 * ba_sd
    ba_loa_upper = ba_bias + 1.96 * ba_sd
    
    return {
        'parameter': param_name,
        'n_samples': len(y_true),
        'r2': float(r2),
        'mae': float(mae),
        'rmse': float(rmse),
        'correlation': float(corr),
        'correlation_p': float(corr_p),
        'bias': float(bias),
        'within_2fold': float(within_2fold),
        'within_3fold': float(within_3fold),
        'bland_altman': {
            'bias': float(ba_bias),
            'sd': float(ba_sd),
            'loa_lower': float(ba_loa_lower),
            'loa_upper': float(ba_loa_upper)
        }
    }


def evaluate_model(model, X, y_true, dataset_name="Test"):
    """Evaluate model and return clinical metrics"""
    print(f"\n📊 Evaluating on {dataset_name}...")
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        y_pred_transformed = model(X_tensor).cpu().numpy()
    
    # Inverse transform
    y_pred = inverse_transform_simple(y_pred_transformed)
    
    # Compute metrics for each parameter
    results = {}
    param_names = ['fu', 'vd', 'clearance']
    
    for i, param in enumerate(param_names):
        metrics = compute_clinical_metrics(
            y_true[:, i],
            y_pred[:, i],
            param
        )
        if metrics:
            results[param] = metrics
            
            print(f"\n  {param.upper()}:")
            print(f"    R²:            {metrics['r2']:.4f}")
            print(f"    MAE:           {metrics['mae']:.4f}")
            print(f"    Within 2-fold: {metrics['within_2fold']*100:.1f}%")
            print(f"    Within 3-fold: {metrics['within_3fold']*100:.1f}%")
            print(f"    Bias:          {metrics['bias']:.4f}")
    
    # Overall average R²
    avg_r2 = np.mean([m['r2'] for m in results.values()])
    print(f"\n  📈 Average R²: {avg_r2:.4f}")
    
    return results, y_pred


# ==================== VISUALIZATION ====================

def plot_validation_results(results_dict, output_path):
    """Plot comprehensive validation results"""
    print("\n📊 Creating validation plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('PBPK Model Validation - Clinical Metrics', fontsize=16, fontweight='bold')
    
    datasets = list(results_dict.keys())
    params = ['fu', 'vd', 'clearance']
    
    # 1. R² comparison
    ax = axes[0, 0]
    data = []
    for dataset in datasets:
        for param in params:
            if param in results_dict[dataset]:
                data.append({
                    'Dataset': dataset,
                    'Parameter': param,
                    'R²': results_dict[dataset][param]['r2']
                })
    df = pd.DataFrame(data)
    if len(df) > 0:
        df_pivot = df.pivot(index='Parameter', columns='Dataset', values='R²')
        df_pivot.plot(kind='bar', ax=ax)
        ax.set_title('R² Score by Parameter', fontweight='bold')
        ax.set_ylabel('R²')
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.legend(title='Dataset')
        ax.grid(True, alpha=0.3)
    
    # 2. Within 2-fold accuracy
    ax = axes[0, 1]
    data = []
    for dataset in datasets:
        for param in params:
            if param in results_dict[dataset]:
                data.append({
                    'Dataset': dataset,
                    'Parameter': param,
                    'Within 2-fold': results_dict[dataset][param]['within_2fold'] * 100
                })
    df = pd.DataFrame(data)
    if len(df) > 0:
        df_pivot = df.pivot(index='Parameter', columns='Dataset', values='Within 2-fold')
        df_pivot.plot(kind='bar', ax=ax)
        ax.set_title('Clinical Accuracy (Within 2-fold)', fontweight='bold')
        ax.set_ylabel('% Within 2-fold')
        ax.axhline(50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
        ax.legend(title='Dataset')
        ax.grid(True, alpha=0.3)
    
    # 3. MAE comparison
    ax = axes[0, 2]
    data = []
    for dataset in datasets:
        for param in params:
            if param in results_dict[dataset]:
                data.append({
                    'Dataset': dataset,
                    'Parameter': param,
                    'MAE': results_dict[dataset][param]['mae']
                })
    df = pd.DataFrame(data)
    if len(df) > 0:
        df_pivot = df.pivot(index='Parameter', columns='Dataset', values='MAE')
        df_pivot.plot(kind='bar', ax=ax)
        ax.set_title('Mean Absolute Error', fontweight='bold')
        ax.set_ylabel('MAE')
        ax.legend(title='Dataset')
        ax.grid(True, alpha=0.3)
    
    # 4. Bias analysis
    ax = axes[1, 0]
    data = []
    for dataset in datasets:
        for param in params:
            if param in results_dict[dataset]:
                data.append({
                    'Dataset': dataset,
                    'Parameter': param,
                    'Bias': results_dict[dataset][param]['bias']
                })
    df = pd.DataFrame(data)
    if len(df) > 0:
        df_pivot = df.pivot(index='Parameter', columns='Dataset', values='Bias')
        df_pivot.plot(kind='bar', ax=ax)
        ax.set_title('Prediction Bias', fontweight='bold')
        ax.set_ylabel('Bias')
        ax.axhline(0, color='red', linestyle='--', alpha=0.5)
        ax.legend(title='Dataset')
        ax.grid(True, alpha=0.3)
    
    # 5. Sample size
    ax = axes[1, 1]
    data = []
    for dataset in datasets:
        for param in params:
            if param in results_dict[dataset]:
                data.append({
                    'Dataset': dataset,
                    'Parameter': param,
                    'N': results_dict[dataset][param]['n_samples']
                })
    df = pd.DataFrame(data)
    if len(df) > 0:
        df_pivot = df.pivot(index='Parameter', columns='Dataset', values='N')
        df_pivot.plot(kind='bar', ax=ax)
        ax.set_title('Sample Size per Parameter', fontweight='bold')
        ax.set_ylabel('N')
        ax.legend(title='Dataset')
        ax.grid(True, alpha=0.3)
    
    # 6. Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = "📊 VALIDATION SUMMARY\n" + "="*40 + "\n\n"
    
    for dataset in datasets:
        avg_r2 = np.mean([results_dict[dataset][p]['r2'] 
                          for p in params if p in results_dict[dataset]])
        avg_2fold = np.mean([results_dict[dataset][p]['within_2fold'] 
                             for p in params if p in results_dict[dataset]])
        
        summary_text += f"{dataset.upper()}:\n"
        summary_text += f"  Average R²: {avg_r2:.3f}\n"
        summary_text += f"  2-fold acc: {avg_2fold*100:.1f}%\n\n"
    
    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {output_path}")
    plt.close()


# ==================== MAIN ====================

def main():
    """Main execution"""
    
    # Load model
    print("🔧 Loading Trial 84 model...")
    if not TRIAL84_MODEL.exists():
        print(f"  ❌ Model not found: {TRIAL84_MODEL}")
        return
    
    checkpoint = torch.load(TRIAL84_MODEL, map_location=device)
    
    model = FlexiblePKModel(
        input_dim=788,
        hidden_dims=[512, 256, 128],
        output_dim=3,
        dropout=0.3,
        use_residual=True,
        use_attention=True
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("  ✓ Model loaded successfully")
    
    # Evaluate on all datasets
    results_all = {}
    
    # 1. KEC Test Set
    X_test, y_test, test_df = load_kec_test_data()
    results_kec, preds_kec = evaluate_model(model, X_test, y_test, "KEC Test")
    results_all['KEC Test'] = results_kec
    
    # 2. External Validation Set
    ext_df, ext_smiles, ext_targets = load_external_validation_set()
    if ext_df is not None:
        # TODO: Generate embeddings for external drugs
        # For now, skip this
        print("\n⚠️  External validation requires embedding generation")
        print("    Skipping for now...")
    
    # Plot results
    plot_path = OUTPUT_DIR / 'validation_results.png'
    plot_validation_results(results_all, plot_path)
    
    # Save JSON report
    report = {
        'model': 'Trial 84',
        'timestamp': pd.Timestamp.now().isoformat(),
        'results': results_all
    }
    
    report_path = OUTPUT_DIR / 'validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Validation complete!")
    print(f"   Report: {report_path}")
    print(f"   Plots: {plot_path}")
    
    # Print summary
    print("\n" + "="*80)
    print("📊 FINAL SUMMARY")
    print("="*80)
    
    for dataset, results in results_all.items():
        print(f"\n{dataset}:")
        avg_r2 = np.mean([r['r2'] for r in results.values()])
        avg_2fold = np.mean([r['within_2fold'] for r in results.values()])
        
        print(f"  Average R²:        {avg_r2:.4f}")
        print(f"  2-fold accuracy:   {avg_2fold*100:.1f}%")
        
        # Check if meets clinical standard
        if avg_2fold >= 0.5:
            print(f"  ✅ CLINICALLY ACCEPTABLE (>50% within 2-fold)")
        else:
            print(f"  ⚠️  Below clinical standard (<50% within 2-fold)")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

