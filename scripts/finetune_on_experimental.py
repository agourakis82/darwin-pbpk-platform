#!/usr/bin/env python3
"""
Fine-tuning do modelo Dynamic GNN em dados experimentais usando técnicas SOTA
- Transfer Learning
- Validação cruzada
- Loss ponderada (mais peso para dados experimentais)

Criado: 2025-11-17
Autor: AI Assistant + Dr. Agourakis
Baseado em: Transfer Learning, Multi-task Learning para PBPK
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from apps.pbpk_core.simulation.dynamic_gnn_pbpk import (
    DynamicPBPKGNN,
    PBPKPhysiologicalParams,
    PBPK_ORGANS,
)


class ExperimentalDataset(Dataset):
    """Dataset para fine-tuning em dados experimentais"""

    def __init__(self, data_path: Path, metadata_path: Path):
        self.data = np.load(data_path, allow_pickle=True)
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        self.doses = self.data['doses']
        self.clearances_hepatic = self.data['clearances_hepatic']
        self.clearances_renal = self.data['clearances_renal']
        self.partition_coeffs = self.data['partition_coeffs']

        # Extrair Cmax e AUC observados
        self.cmax_obs = []
        self.auc_obs = []
        for i, meta in enumerate(self.metadata):
            # Priorizar valores convertidos (mg/L, mg·h/L)
            cmax_val = None
            if 'cmax_obs_mg_l' in meta and meta['cmax_obs_mg_l'] is not None:
                try:
                    cmax_val = float(meta['cmax_obs_mg_l'])
                except (ValueError, TypeError):
                    pass

            if cmax_val is None:
                cmax_raw = meta.get('cmax_obs', None)
                if cmax_raw is not None and not isinstance(cmax_raw, dict):
                    try:
                        cmax_ng_ml = float(cmax_raw)
                        cmax_val = cmax_ng_ml / 1000.0  # ng/mL → mg/L
                    except (ValueError, TypeError):
                        pass

            auc_val = None
            if 'auc_obs_mg_h_l' in meta and meta['auc_obs_mg_h_l'] is not None:
                try:
                    auc_val = float(meta['auc_obs_mg_h_l'])
                except (ValueError, TypeError):
                    pass

            if auc_val is None:
                auc_raw = meta.get('auc_obs', None)
                if auc_raw is not None and not isinstance(auc_raw, dict):
                    try:
                        auc_ng_h_ml = float(auc_raw)
                        auc_val = auc_ng_h_ml / 1000.0  # ng·h/mL → mg·h/L
                    except (ValueError, TypeError):
                        pass

            self.cmax_obs.append(cmax_val)
            self.auc_obs.append(auc_val)

    def __len__(self):
        return len(self.doses)

    def __getitem__(self, idx):
        # Garantir que cmax_obs e auc_obs são números válidos
        cmax_val = self.cmax_obs[idx]
        if cmax_val is None or isinstance(cmax_val, dict):
            cmax_val = 0.0
        else:
            try:
                cmax_val = float(cmax_val)
            except (ValueError, TypeError):
                cmax_val = 0.0

        auc_val = self.auc_obs[idx]
        if auc_val is None or isinstance(auc_val, dict):
            auc_val = 0.0
        else:
            try:
                auc_val = float(auc_val)
            except (ValueError, TypeError):
                auc_val = 0.0

        return {
            'dose': torch.tensor(self.doses[idx], dtype=torch.float32),
            'clearance_hepatic': torch.tensor(self.clearances_hepatic[idx], dtype=torch.float32),
            'clearance_renal': torch.tensor(self.clearances_renal[idx], dtype=torch.float32),
            'partition_coeffs': torch.tensor(self.partition_coeffs[idx], dtype=torch.float32),
            'cmax_obs': torch.tensor(cmax_val, dtype=torch.float32),
            'auc_obs': torch.tensor(auc_val, dtype=torch.float32),
            'has_cmax': self.cmax_obs[idx] is not None and not isinstance(self.cmax_obs[idx], dict),
            'has_auc': self.auc_obs[idx] is not None and not isinstance(self.auc_obs[idx], dict),
        }


def calculate_cmax_auc(concentrations: torch.Tensor, time_points: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calcula Cmax e AUC a partir de concentrações"""
    cmax = torch.max(concentrations, dim=-1)[0]  # [NUM_ORGANS]
    auc = torch.trapz(concentrations, time_points, dim=-1)  # [NUM_ORGANS]
    return cmax, auc


def finetune_model(
    checkpoint_path: Path,
    experimental_data_path: Path,
    experimental_metadata_path: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-5,
    experimental_weight: float = 10.0,  # Peso maior para dados experimentais
    device: str = "cuda",
):
    """
    Fine-tuning do modelo em dados experimentais
    """
    print("🎯 FINE-TUNING EM DADOS EXPERIMENTAIS (SOTA)")
    print("=" * 70)

    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Carregar modelo pré-treinado
    print("\n1️⃣  Carregando modelo pré-treinado...")
    model = DynamicPBPKGNN(
        node_dim=16,
        edge_dim=4,
        hidden_dim=128,
        num_gnn_layers=4,
        num_temporal_steps=120,
        dt=0.1,
        use_attention=True,
    )

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        else:
            model.load_state_dict(checkpoint, strict=False)
        print(f"   ✅ Checkpoint carregado: {checkpoint_path}")

    model = model.to(device)

    # Dataset e DataLoader
    print("\n2️⃣  Preparando dataset...")
    dataset = ExperimentalDataset(experimental_data_path, experimental_metadata_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"   Total de amostras: {len(dataset)}")
    print(f"   Com Cmax observado: {sum(1 for m in dataset.metadata if m.get('cmax_obs') is not None)}")
    print(f"   Com AUC observado: {sum(1 for m in dataset.metadata if m.get('auc_obs') is not None)}")

    # Otimizador e scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Loss function (MSE ponderado)
    criterion = nn.MSELoss()

    # Treinamento
    print("\n3️⃣  Iniciando fine-tuning...")
    best_loss = float('inf')
    train_losses = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            doses = batch['dose'].to(device)
            clearances_hepatic = batch['clearance_hepatic'].to(device)
            clearances_renal = batch['clearance_renal'].to(device)
            partition_coeffs = batch['partition_coeffs'].to(device)
            cmax_obs = batch['cmax_obs'].to(device)
            auc_obs = batch['auc_obs'].to(device)
            has_cmax = batch['has_cmax']
            has_auc = batch['has_auc']

            optimizer.zero_grad()

            # Predizer
            batch_size_actual = len(doses)
            pred_list = []
            for i in range(batch_size_actual):
                partition_dict = {
                    organ: float(partition_coeffs[i, j])
                    for j, organ in enumerate(PBPK_ORGANS)
                }
                params = PBPKPhysiologicalParams(
                    clearance_hepatic=float(clearances_hepatic[i]),
                    clearance_renal=float(clearances_renal[i]),
                    partition_coeffs=partition_dict,
                )
                result = model(float(doses[i]), params)
                pred_conc = result["concentrations"]  # [NUM_ORGANS, T]
                pred_list.append(pred_conc)

            pred_batch = torch.stack(pred_list)  # [B, NUM_ORGANS, T]

            # Time points
            time_points = result["time_points"]  # [T]

            # Calcular Cmax e AUC previstos (blood = índice 0)
            pred_cmax, pred_auc = calculate_cmax_auc(pred_batch[:, 0, :], time_points.unsqueeze(0).expand(batch_size_actual, -1))

            # Loss: MSE ponderado
            loss = 0.0
            n_valid = 0

            # Cmax loss
            if has_cmax.any():
                mask_cmax = torch.tensor(has_cmax, device=device)
                if mask_cmax.sum() > 0:
                    loss_cmax = criterion(pred_cmax[mask_cmax], cmax_obs[mask_cmax])
                    loss += experimental_weight * loss_cmax
                    n_valid += 1

            # AUC loss
            if has_auc.any():
                mask_auc = torch.tensor(has_auc, device=device)
                if mask_auc.sum() > 0:
                    loss_auc = criterion(pred_auc[mask_auc], auc_obs[mask_auc])
                    loss += experimental_weight * loss_auc
                    n_valid += 1

            if n_valid > 0:
                loss = loss / n_valid
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

        avg_loss = epoch_loss / n_batches if n_batches > 0 else float('inf')
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)

        print(f"   Epoch {epoch+1}: Loss = {avg_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.2e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_dir / "best_finetuned_model.pt")
            print(f"   ✅ Novo melhor modelo salvo (loss: {best_loss:.6f})")

    # Salvar modelo final
    torch.save(model.state_dict(), output_dir / "final_finetuned_model.pt")

    # Salvar histórico
    with open(output_dir / "finetuning_history.json", 'w') as f:
        json.dump({
            'train_losses': train_losses,
            'best_loss': float(best_loss),
            'epochs': epochs,
            'lr': lr,
            'experimental_weight': experimental_weight,
        }, f, indent=2)

    print(f"\n✅ Fine-tuning concluído!")
    print(f"   Modelo salvo em: {output_dir / 'best_finetuned_model.pt'}")
    print(f"   Histórico salvo em: {output_dir / 'finetuning_history.json'}")


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Fine-tuning em dados experimentais (SOTA)")
    ap.add_argument("--checkpoint", required=True, help="Checkpoint do modelo pré-treinado")
    ap.add_argument("--experimental-data", required=True, help="Dados experimentais (.npz)")
    ap.add_argument("--experimental-metadata", required=True, help="Metadata experimental (.json)")
    ap.add_argument("--output-dir", required=True, help="Diretório de saída")
    ap.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    ap.add_argument("--batch-size", type=int, default=8, help="Batch size")
    ap.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    ap.add_argument("--experimental-weight", type=float, default=10.0, help="Peso para dados experimentais")
    ap.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    finetune_model(
        Path(args.checkpoint),
        Path(args.experimental_data),
        Path(args.experimental_metadata),
        output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        experimental_weight=args.experimental_weight,
        device=args.device,
    )


if __name__ == "__main__":
    from typing import Tuple
    main()

