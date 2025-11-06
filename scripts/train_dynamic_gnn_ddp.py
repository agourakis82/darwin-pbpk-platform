#!/usr/bin/env python3
"""
Treinamento Dynamic GNN com DistributedDataParallel (DDP)

Suporta multi-GPU e multi-node training.

Autor: Dr. Demetrios Chiuratto Agourakis
Data: Novembro 2025
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
import argparse
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional
import os

# Adicionar raiz do projeto ao path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from apps.pbpk_core.simulation.dynamic_gnn_pbpk import (
    DynamicPBPKGNN,
    PBPKPhysiologicalParams,
    PBPK_ORGANS,
    NUM_ORGANS
)

# Importar classes do script original
from scripts.train_dynamic_gnn_pbpk import (
    PBPKDataset,
    create_physiological_params,
    compute_loss
)


def setup(rank: int, world_size: int, master_addr: str = "localhost", master_port: str = "12355"):
    """Inicializa o processo distribuído."""
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    
    # Inicializar processo group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Set device
    torch.cuda.set_device(rank)


def cleanup():
    """Limpa o processo distribuído."""
    dist.destroy_process_group()


def train_epoch_ddp(
    model: DDP,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    rank: int,
    organ_weights: Optional[torch.Tensor] = None
) -> Tuple[float, Dict[str, float]]:
    """Treina por uma época com DDP."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    losses_per_organ = {organ: 0.0 for organ in PBPK_ORGANS}
    
    # Progress bar apenas no rank 0
    if rank == 0:
        pbar = tqdm(dataloader, desc="Training", leave=False)
    else:
        pbar = dataloader
    
    for batch in pbar:
        # Mover para device
        dose = batch["dose"].to(device)
        clearance_hepatic = batch["clearance_hepatic"].to(device)
        clearance_renal = batch["clearance_renal"].to(device)
        partition_coeffs = batch["partition_coeffs"].to(device)
        true_conc = batch["concentrations"].to(device)
        time_points = batch["time_points"].to(device)
        
        # DataLoader pode criar batch de time_points incorretamente
        if time_points.dim() > 1:
            time_points = time_points[0]
        
        batch_size = len(dose)
        batch_loss = 0.0
        
        optimizer.zero_grad()
        
        # Processar cada amostra do batch
        for i in range(batch_size):
            params = create_physiological_params(
                dose[i].item(),
                clearance_hepatic[i].item(),
                clearance_renal[i].item(),
                partition_coeffs[i]
            )
            
            # Forward pass
            results = model.module(dose[i].item(), params, time_points)
            pred_conc = results["concentrations"]
            true_conc_i = true_conc[i]
            
            # Garantir shape correto
            if pred_conc.shape[0] != NUM_ORGANS:
                if pred_conc.shape[1] == NUM_ORGANS:
                    pred_conc = pred_conc.t()
            
            if true_conc_i.shape[0] != NUM_ORGANS:
                if true_conc_i.shape[1] == NUM_ORGANS:
                    true_conc_i = true_conc_i.t()
            
            if pred_conc.shape[1] != true_conc_i.shape[1]:
                min_t = min(pred_conc.shape[1], true_conc_i.shape[1])
                pred_conc = pred_conc[:, :min_t]
                true_conc_i = true_conc_i[:, :min_t]
            
            loss = compute_loss(pred_conc, true_conc_i, organ_weights)
            batch_loss += loss
        
        batch_loss = batch_loss / batch_size
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    return avg_loss, losses_per_organ


def validate_ddp(
    model: DDP,
    dataloader: DataLoader,
    device: torch.device,
    rank: int,
    organ_weights: Optional[torch.Tensor] = None
) -> Tuple[float, Dict[str, float]]:
    """Valida com DDP."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    losses_per_organ = {organ: 0.0 for organ in PBPK_ORGANS}
    
    with torch.no_grad():
        if rank == 0:
            pbar = tqdm(dataloader, desc="Validation", leave=False)
        else:
            pbar = dataloader
        
        for batch in pbar:
            dose = batch["dose"].to(device)
            clearance_hepatic = batch["clearance_hepatic"].to(device)
            clearance_renal = batch["clearance_renal"].to(device)
            partition_coeffs = batch["partition_coeffs"].to(device)
            true_conc = batch["concentrations"].to(device)
            time_points = batch["time_points"].to(device)
            
            if time_points.dim() > 1:
                time_points = time_points[0]
            
            batch_size = len(dose)
            batch_loss = 0.0
            
            for i in range(batch_size):
                params = create_physiological_params(
                    dose[i].item(),
                    clearance_hepatic[i].item(),
                    clearance_renal[i].item(),
                    partition_coeffs[i]
                )
                
                results = model.module(dose[i].item(), params, time_points)
                pred_conc = results["concentrations"]
                true_conc_i = true_conc[i]
                
                if pred_conc.shape[0] != NUM_ORGANS:
                    if pred_conc.shape[1] == NUM_ORGANS:
                        pred_conc = pred_conc.t()
                
                if true_conc_i.shape[0] != NUM_ORGANS:
                    if true_conc_i.shape[1] == NUM_ORGANS:
                        true_conc_i = true_conc_i.t()
                
                if pred_conc.shape[1] != true_conc_i.shape[1]:
                    min_t = min(pred_conc.shape[1], true_conc_i.shape[1])
                    pred_conc = pred_conc[:, :min_t]
                    true_conc_i = true_conc_i[:, :min_t]
                
                loss = compute_loss(pred_conc, true_conc_i, organ_weights)
                batch_loss += loss
            
            batch_loss = batch_loss / batch_size
            total_loss += batch_loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    return avg_loss, losses_per_organ


def train_ddp(
    rank: int,
    world_size: int,
    data_path: Path,
    output_dir: Path,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    master_addr: str,
    master_port: str,
    train_split: float = 0.8,
    weight_organs: bool = True
):
    """Treina com DDP."""
    # Setup
    setup(rank, world_size, master_addr, master_port)
    device = torch.device(f"cuda:{rank}")
    
    if rank == 0:
        print("=" * 80)
        print("TREINAMENTO: Dynamic GNN para PBPK (DDP)")
        print("=" * 80)
        print(f"\nConfiguração:")
        print(f"   World size: {world_size}")
        print(f"   Dataset: {data_path}")
        print(f"   Output: {output_dir}")
        print(f"   Épocas: {num_epochs}")
        print(f"   Batch size: {batch_size} (por GPU)")
        print(f"   Learning rate: {learning_rate}")
        print()
    
    # Carregar dataset
    if rank == 0:
        print("📦 Carregando dataset...")
    full_dataset = PBPKDataset(data_path)
    
    # Split train/val
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=0
    )
    
    if rank == 0:
        print(f"   Train: {len(train_dataset):,} amostras")
        print(f"   Val: {len(val_dataset):,} amostras")
    
    # Criar modelo
    if rank == 0:
        print("\n🏗️  Criando modelo...")
    model = DynamicPBPKGNN(
        node_dim=16,
        edge_dim=4,
        hidden_dim=64,
        num_gnn_layers=3,
        num_temporal_steps=100,
        dt=0.1
    ).to(device)
    
    # DDP
    model = DDP(model, device_ids=[rank])
    
    if rank == 0:
        num_params = sum(p.numel() for p in model.module.parameters())
        print(f"   Parâmetros: {num_params:,}")
        print(f"   🚀 Usando {world_size} GPUs (DDP)")
    
    # Organ weights
    if weight_organs:
        organ_weights = torch.ones(NUM_ORGANS, device=device)
        critical_organs = ["blood", "liver", "kidney", "brain"]
        for organ in critical_organs:
            idx = PBPK_ORGANS.index(organ)
            organ_weights[idx] = 2.0
    else:
        organ_weights = None
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    
    # Training loop
    if rank == 0:
        print("\n🚀 Iniciando treinamento...")
        print()
    
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        # Set epoch for sampler (importante para DDP)
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, _ = train_epoch_ddp(
            model, train_loader, optimizer, device, rank, organ_weights
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss, _ = validate_ddp(model, val_loader, device, rank, organ_weights)
        val_losses.append(val_loss)
        
        # Scheduler
        scheduler.step(val_loss)
        
        # Print (apenas rank 0)
        if rank == 0:
            print(f"   Train Loss: {train_loss:.6f}")
            print(f"   Val Loss: {val_loss:.6f}")
            
            # Salvar melhor modelo (apenas rank 0)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }, output_dir / "best_model.pt")
                print(f"   ✅ Novo melhor modelo salvo! (Val Loss: {val_loss:.6f})")
            print()
    
    # Salvar modelo final (apenas rank 0)
    if rank == 0:
        torch.save({
            "epoch": num_epochs,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_losses": train_losses,
            "val_losses": val_losses,
        }, output_dir / "final_model.pt")
        
        # Plot losses
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training Progress (DDP)")
        plt.savefig(output_dir / "training_curve.png", dpi=150)
        plt.close()
        
        print("=" * 80)
        print("✅ TREINAMENTO CONCLUÍDO!")
        print("=" * 80)
        print(f"\nMelhor Val Loss: {best_val_loss:.6f}")
        print(f"Modelos salvos em: {output_dir}")
    
    cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina Dynamic GNN com DDP")
    parser.add_argument("--data", type=str, required=True, help="Caminho para dataset .npz")
    parser.add_argument("--output", type=str, default="models/dynamic_gnn_ddp", help="Diretório de saída")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size por GPU")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--world-size", type=int, default=2, help="Número de GPUs")
    parser.add_argument("--master-addr", type=str, default="localhost", help="Master address")
    parser.add_argument("--master-port", type=str, default="12355", help="Master port")
    parser.add_argument("--train-split", type=float, default=0.8, help="Fração de treinamento")
    parser.add_argument("--no-organ-weights", action="store_true", help="Não usar pesos por órgão")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Spawn processos
    mp.spawn(
        train_ddp,
        args=(
            args.world_size,
            Path(args.data),
            output_dir,
            args.epochs,
            args.batch_size,
            args.lr,
            args.master_addr,
            args.master_port,
            args.train_split,
            not args.no_organ_weights
        ),
        nprocs=args.world_size,
        join=True
    )

