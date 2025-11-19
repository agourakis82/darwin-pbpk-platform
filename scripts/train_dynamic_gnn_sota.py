#!/usr/bin/env python3
"""
Treinamento Dynamic GNN para PBPK SOTA

Autor: Dr. Demetrios Chiuratto Agourakis
Data: Novembro 2025

Notas de dependência:
- Requer PyTorch >= 2.8.0
- Para execução em GPU, instale extensões PyG compatíveis (torch-scatter, torch-sparse).
  Veja README ou execute:
      pip install --no-cache-dir --upgrade torch_scatter torch_sparse \
          --find-links https://data.pyg.org/whl/torch-2.8.0+cu128.html
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import json
import argparse
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import time
from datetime import datetime

# Adicionar raiz do projeto ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from apps.pbpk_core.simulation.dynamic_gnn_pbpk import (
    DynamicPBPKGNN,
    PBPKPhysiologicalParams,
    PBPK_ORGANS,
    NUM_ORGANS
)


class PBPKDataset(Dataset):
    """Dataset para treinamento do Dynamic GNN."""

    def __init__(self, data_path: Path):
        data = np.load(data_path)

        self.doses = torch.tensor(data["doses"], dtype=torch.float32)
        self.clearances_hepatic = torch.tensor(data["clearances_hepatic"], dtype=torch.float32)
        self.clearances_renal = torch.tensor(data["clearances_renal"], dtype=torch.float32)
        self.partition_coeffs = torch.tensor(data["partition_coeffs"], dtype=torch.float32)
        self.concentrations = torch.tensor(data["concentrations"], dtype=torch.float32)
        self.time_points = torch.tensor(data["time_points"], dtype=torch.float32)

        self.num_samples = len(self.doses)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "dose": self.doses[idx],
            "clearance_hepatic": self.clearances_hepatic[idx],
            "clearance_renal": self.clearances_renal[idx],
            "partition_coeffs": self.partition_coeffs[idx],
            "concentrations": self.concentrations[idx],
            "time_points": self.time_points.clone()
        }


def compute_loss(pred_conc: torch.Tensor, true_conc: torch.Tensor, organ_weights: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computa loss com pesos por órgão.

    Args:
        pred_conc: [NUM_ORGANS, T] ou [B, NUM_ORGANS, T]
        true_conc: [NUM_ORGANS, T] ou [B, NUM_ORGANS, T]
        organ_weights: [NUM_ORGANS]

    Returns:
        loss: Tensor escalar
        losses_per_organ: Dict com loss por órgão
    """
    # MSE por órgão
    if pred_conc.dim() == 3:
        mse_per_organ = torch.mean((pred_conc - true_conc) ** 2, dim=2).mean(dim=0)
    else:
        mse_per_organ = torch.mean((pred_conc - true_conc) ** 2, dim=1)

    # Loss ponderada
    loss = torch.sum(mse_per_organ * organ_weights) / torch.sum(organ_weights)

    # Perdas por órgão para logging
    losses_per_organ = {
        PBPK_ORGANS[i]: mse_per_organ[i].item()
        for i in range(NUM_ORGANS)
    }

    return loss, losses_per_organ


def _create_physiological_params(
    dose: torch.Tensor,
    clearance_hepatic: torch.Tensor,
    clearance_renal: torch.Tensor,
    partition_coeffs: torch.Tensor
) -> Tuple[float, PBPKPhysiologicalParams]:
    """Helper para converter tensores em parâmetros fisiológicos."""
    partition_dict = {
        organ: float(partition_coeffs[idx].item())
        for idx, organ in enumerate(PBPK_ORGANS)
    }
    params = PBPKPhysiologicalParams(
        clearance_hepatic=float(clearance_hepatic.item()),
        clearance_renal=float(clearance_renal.item()),
        partition_coeffs=partition_dict
    )
    return float(dose.item()), params


def train_epoch_sota(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    organ_weights: torch.Tensor,
    scaler: GradScaler,
    gradient_clip: float = 1.0
) -> Tuple[float, Dict[str, float]]:
    """Treina uma época com mixed precision e gradient clipping."""
    model.train()
    total_loss = 0.0
    all_organ_losses = {organ: [] for organ in PBPK_ORGANS}

    for batch in tqdm(train_loader, desc="Training"):
        # Mover para device
        dose = batch["dose"].to(device)
        clearance_hepatic = batch["clearance_hepatic"].to(device)
        clearance_renal = batch["clearance_renal"].to(device)
        partition_coeffs = batch["partition_coeffs"].to(device)
        true_conc = batch["concentrations"].to(device)
        time_points = batch["time_points"].to(device)

        # Garantir time_points 1D
        if time_points.dim() > 1:
            time_points = time_points[0]

        optimizer.zero_grad()

        # Mixed precision training
        amp_device_type = "cuda" if device.type == "cuda" else "cpu"
        amp_enabled = device.type == "cuda"
        with autocast(device_type=amp_device_type, enabled=amp_enabled):
            batch_size = len(dose)
            pred_conc_list = []

            for i in range(batch_size):
                sample_dose, params = _create_physiological_params(
                    dose=dose[i],
                    clearance_hepatic=clearance_hepatic[i],
                    clearance_renal=clearance_renal[i],
                    partition_coeffs=partition_coeffs[i].detach().cpu()
                )
                model_output = model(
                    dose=sample_dose,
                    physiological_params=params,
                    time_points=time_points
                )
                pred_conc_list.append(model_output["concentrations"])  # [NUM_ORGANS, T]

            pred_conc = torch.stack(pred_conc_list, dim=0)  # [B, NUM_ORGANS, T]

            loss, organ_losses = compute_loss(pred_conc, true_conc, organ_weights)

        # Backward com scaler
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip)

        # Optimizer step
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        for organ, loss_val in organ_losses.items():
            all_organ_losses[organ].append(loss_val)

    avg_loss = total_loss / len(train_loader)
    avg_organ_losses = {
        organ: np.mean(losses) if losses else 0.0
        for organ, losses in all_organ_losses.items()
    }

    return avg_loss, avg_organ_losses


def validate_sota(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    organ_weights: torch.Tensor
) -> Tuple[float, Dict[str, float]]:
    """Valida com mixed precision."""
    model.eval()
    total_loss = 0.0
    all_organ_losses = {organ: [] for organ in PBPK_ORGANS}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            dose = batch["dose"].to(device)
            clearance_hepatic = batch["clearance_hepatic"].to(device)
            clearance_renal = batch["clearance_renal"].to(device)
            partition_coeffs = batch["partition_coeffs"].to(device)
            true_conc = batch["concentrations"].to(device)
            time_points = batch["time_points"].to(device)

            if time_points.dim() > 1:
                time_points = time_points[0]

            amp_device_type = "cuda" if device.type == "cuda" else "cpu"
            amp_enabled = device.type == "cuda"
            with autocast(device_type=amp_device_type, enabled=amp_enabled):
                batch_size = len(dose)
                pred_conc_list = []

                for i in range(batch_size):
                    sample_dose, params = _create_physiological_params(
                        dose=dose[i],
                        clearance_hepatic=clearance_hepatic[i],
                        clearance_renal=clearance_renal[i],
                        partition_coeffs=partition_coeffs[i].detach().cpu()
                    )
                    model_output = model(
                        dose=sample_dose,
                        physiological_params=params,
                        time_points=time_points
                    )
                    pred_conc_list.append(model_output["concentrations"])  # [NUM_ORGANS, T]

                pred_conc = torch.stack(pred_conc_list, dim=0)

                loss, organ_losses = compute_loss(pred_conc, true_conc, organ_weights)

            total_loss += loss.item()
            for organ, loss_val in organ_losses.items():
                all_organ_losses[organ].append(loss_val)

    avg_loss = total_loss / len(val_loader)
    avg_organ_losses = {
        organ: np.mean(losses) if losses else 0.0
        for organ, losses in all_organ_losses.items()
    }

    return avg_loss, avg_organ_losses


def train_sota(
    data_path: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    device: str = "cuda",
    num_workers: int = 4,
    gradient_clip: float = 1.0,
    warmup_epochs: int = 5,
    patience: int = 10,
    min_lr: float = 1e-6
):
    """
    Treinamento SOTA com todas as técnicas avançadas.

    Técnicas SOTA implementadas:
    - Mixed Precision Training (AMP)
    - Gradient Clipping
    - Cosine Annealing LR Schedule com Warmup
    - Early Stopping
    - Checkpointing
    - Organ-weighted loss
    """
    device = torch.device(device)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("TREINAMENTO SOTA: Dynamic GNN para PBPK")
    print("=" * 80)
    print(f"\nConfiguração SOTA:")
    print(f"   Dataset: {data_path}")
    print(f"   Output: {output_dir}")
    print(f"   Épocas: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {lr}")
    print(f"   Device: {device}")
    print(f"   Gradient clipping: {gradient_clip}")
    print(f"   Warmup epochs: {warmup_epochs}")
    print(f"   Early stopping patience: {patience}")
    print(f"   Mixed Precision: ✅")
    print(f"   Cosine Annealing: ✅")

    # Dataset
    print(f"\n📦 Carregando dataset...")
    dataset = PBPKDataset(data_path)

    # Split train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"   Train: {len(train_dataset)} amostras")
    print(f"   Val: {len(val_dataset)} amostras")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device.type == "cuda" else False
    )

    # Modelo
    print(f"\n🏗️  Criando modelo...")
    model = DynamicPBPKGNN(
        node_dim=16,
        edge_dim=4,
        hidden_dim=64,
        num_gnn_layers=3,
        num_temporal_steps=100,
        dt=0.1
    ).to(device)

    # Multi-GPU support
    if device.type == "cuda" and torch.cuda.device_count() > 1:
        print(f"   🚀 Usando {torch.cuda.device_count()} GPUs (DataParallel)")
        model = nn.DataParallel(model)
        effective_batch_size = batch_size * torch.cuda.device_count()
        print(f"   Effective batch size: {effective_batch_size}")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Parâmetros: {num_params:,}")

    # Optimizer e Scheduler SOTA
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # Cosine Annealing com Warmup (garante domínio válido)
    effective_warmup = int(max(1, min(warmup_epochs, epochs - 1))) if epochs > 1 else 1
    total_decay_steps = max(1, epochs - effective_warmup)

    def lr_lambda(epoch: int) -> float:
        if epoch < effective_warmup:
            return (epoch + 1) / effective_warmup
        progress = (epoch - effective_warmup) / total_decay_steps
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed Precision Scaler
    amp_device_type = "cuda" if device.type == "cuda" else "cpu"
    amp_enabled = device.type == "cuda"
    scaler = GradScaler(device_type=amp_device_type, enabled=amp_enabled)

    # Organ weights
    organ_weights = torch.ones(NUM_ORGANS, device=device)

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': []
    }

    print(f"\n🚀 Iniciando treinamento SOTA...\n")
    start_time = time.time()

    for epoch in range(epochs):
        # Train
        train_loss, train_org_losses = train_epoch_sota(
            model, train_loader, optimizer, device, organ_weights, scaler, gradient_clip
        )

        # Validate
        val_loss, val_org_losses = validate_sota(model, val_loader, device, organ_weights)

        # Update LR
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        next_lr = optimizer.param_groups[0]['lr']

        # Log
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  LR (atual): {current_lr:.2e}")
        print(f"  LR (próximo): {next_lr:.2e}")

        # History
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Salvar checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'history': history
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
            print(f"  ⭐ Novo melhor modelo salvo! (Val Loss: {val_loss:.6f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping após {epoch+1} épocas")
            break

    # Salvar história
    elapsed_time = time.time() - start_time
    history['elapsed_time'] = elapsed_time

    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Plot
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Curves')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['lr'], label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.legend()
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curve.png", dpi=150)
    print(f"\n✅ Treinamento concluído!")
    print(f"   Tempo total: {elapsed_time/3600:.2f} horas")
    print(f"   Melhor Val Loss: {best_val_loss:.6f}")
    print(f"   Modelo salvo: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treinamento SOTA Dynamic GNN")
    parser.add_argument("--data", type=str, required=True, help="Caminho para dataset .npz")
    parser.add_argument("--output", type=str, required=True, help="Diretório de output")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--num-workers", type=int, default=4, help="Num workers")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Gradient clipping")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Warmup epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    args = parser.parse_args()

    # Auto-detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
            print(f"✅ CUDA disponível: {torch.cuda.device_count()} GPU(s)")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            device = "cpu"
            print("⚠️  CUDA não disponível, usando CPU")
    else:
        device = args.device

    train_sota(
        data_path=Path(args.data),
        output_dir=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        num_workers=args.num_workers,
        gradient_clip=args.gradient_clip,
        warmup_epochs=args.warmup_epochs,
        patience=args.patience
    )

