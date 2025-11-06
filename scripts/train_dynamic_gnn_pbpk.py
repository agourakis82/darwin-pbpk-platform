"""
Treinamento do Dynamic GNN para PBPK

Autor: Dr. Demetrios Chiuratto Agourakis
Data: Novembro 2025
"""

import torch
import torch.nn as nn
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
        """
        Args:
            data_path: Caminho para arquivo .npz com dados
        """
        data = np.load(data_path)
        
        self.doses = torch.tensor(data["doses"], dtype=torch.float32)
        self.clearances_hepatic = torch.tensor(data["clearances_hepatic"], dtype=torch.float32)
        self.clearances_renal = torch.tensor(data["clearances_renal"], dtype=torch.float32)
        self.partition_coeffs = torch.tensor(data["partition_coeffs"], dtype=torch.float32)
        self.concentrations = torch.tensor(data["concentrations"], dtype=torch.float32)  # [N, NUM_ORGANS, T]
        self.time_points = torch.tensor(data["time_points"], dtype=torch.float32)
        
        self.num_samples = len(self.doses)
        self.num_time_points = len(self.time_points)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Garantir shape correto: [NUM_ORGANS, T]
        conc = self.concentrations[idx]  # [NUM_ORGANS, T] do dataset
        if conc.shape[0] != NUM_ORGANS:
            conc = conc.t()  # Transpor se necessário
        
        return {
            "dose": self.doses[idx],
            "clearance_hepatic": self.clearances_hepatic[idx],
            "clearance_renal": self.clearances_renal[idx],
            "partition_coeffs": self.partition_coeffs[idx],  # [NUM_ORGANS]
            "concentrations": conc,  # [NUM_ORGANS, T]
            "time_points": self.time_points
        }


def create_physiological_params(
    dose: float,
    clearance_hepatic: float,
    clearance_renal: float,
    partition_coeffs: torch.Tensor
) -> PBPKPhysiologicalParams:
    """Cria PBPKPhysiologicalParams a partir de tensores."""
    partition_dict = {organ: float(partition_coeffs[i]) for i, organ in enumerate(PBPK_ORGANS)}
    
    return PBPKPhysiologicalParams(
        clearance_hepatic=float(clearance_hepatic),
        clearance_renal=float(clearance_renal),
        partition_coeffs=partition_dict
    )


def compute_loss(
    pred_conc: torch.Tensor,
    true_conc: torch.Tensor,
    organ_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Computa loss (MSE com pesos opcionais por órgão).
    
    Args:
        pred_conc: [NUM_ORGANS, T] ou [T, NUM_ORGANS]
        true_conc: [NUM_ORGANS, T] ou [T, NUM_ORGANS]
        organ_weights: [NUM_ORGANS] (opcional)
    
    Returns:
        Loss escalar
    """
    # Garantir shape correto: [NUM_ORGANS, T]
    if pred_conc.shape[0] != NUM_ORGANS:
        pred_conc = pred_conc.t()  # Transpor se necessário
    if true_conc.shape[0] != NUM_ORGANS:
        true_conc = true_conc.t()
    
    # MSE por órgão (média sobre tempo)
    mse_per_organ = torch.mean((pred_conc - true_conc) ** 2, dim=1)  # [NUM_ORGANS]
    
    # Aplicar pesos se fornecidos
    if organ_weights is not None:
        mse_per_organ = mse_per_organ * organ_weights
    
    # Loss total
    loss = torch.mean(mse_per_organ)
    
    return loss


def train_epoch(
    model: DynamicPBPKGNN,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    organ_weights: Optional[torch.Tensor] = None
) -> Tuple[float, Dict[str, float]]:
    """Treina por uma época."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    losses_per_organ = {organ: 0.0 for organ in PBPK_ORGANS}
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
        # Mover para device
        dose = batch["dose"].to(device)
        clearance_hepatic = batch["clearance_hepatic"].to(device)
        clearance_renal = batch["clearance_renal"].to(device)
        partition_coeffs = batch["partition_coeffs"].to(device)
        true_conc = batch["concentrations"].to(device)  # [B, NUM_ORGANS, T]
        time_points = batch["time_points"].to(device)
        
        batch_size = len(dose)
        batch_loss = 0.0
        
        optimizer.zero_grad()
        
        # Processar cada amostra do batch
        for i in range(batch_size):
            # Criar parâmetros fisiológicos
            params = create_physiological_params(
                dose[i].item(),
                clearance_hepatic[i].item(),
                clearance_renal[i].item(),
                partition_coeffs[i]
            )
            
            # Forward pass
            results = model(dose[i].item(), params, time_points)
            pred_conc = results["concentrations"]  # [NUM_ORGANS, T]
            true_conc_i = true_conc[i]  # [NUM_ORGANS, T]
            
            # Debug shapes (primeira iteração)
            if i == 0 and num_batches == 1:
                print(f"   Debug shapes:")
                print(f"      pred_conc: {pred_conc.shape}")
                print(f"      true_conc_i: {true_conc_i.shape}")
                print(f"      time_points: {time_points.shape}")
                print(f"      NUM_ORGANS: {NUM_ORGANS}")
            
            # Garantir que shapes batem
            if pred_conc.shape[1] != true_conc_i.shape[1]:
                # Interpolar ou truncar se necessário
                min_t = min(pred_conc.shape[1], true_conc_i.shape[1])
                pred_conc = pred_conc[:, :min_t]
                true_conc_i = true_conc_i[:, :min_t]
            
            # Loss
            loss = compute_loss(pred_conc, true_conc_i, organ_weights)
            batch_loss += loss
        
        # Backward
        batch_loss = batch_loss / batch_size
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss.item()
        num_batches += 1
        
        # Per-organ losses (último batch)
        if num_batches == len(dataloader):
            with torch.no_grad():
                for j, organ in enumerate(PBPK_ORGANS):
                    organ_loss = torch.mean((pred_conc[j] - true_conc[-1, j]) ** 2).item()
                    losses_per_organ[organ] = organ_loss
    
    avg_loss = total_loss / num_batches
    
    return avg_loss, losses_per_organ


def validate(
    model: DynamicPBPKGNN,
    dataloader: DataLoader,
    device: torch.device,
    organ_weights: Optional[torch.Tensor] = None
) -> Tuple[float, Dict[str, float]]:
    """Valida o modelo."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    losses_per_organ = {organ: 0.0 for organ in PBPK_ORGANS}
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            dose = batch["dose"].to(device)
            clearance_hepatic = batch["clearance_hepatic"].to(device)
            clearance_renal = batch["clearance_renal"].to(device)
            partition_coeffs = batch["partition_coeffs"].to(device)
            true_conc = batch["concentrations"].to(device)
            time_points = batch["time_points"].to(device)
            
            batch_size = len(dose)
            batch_loss = 0.0
            
            for i in range(batch_size):
                params = create_physiological_params(
                    dose[i].item(),
                    clearance_hepatic[i].item(),
                    clearance_renal[i].item(),
                    partition_coeffs[i]
                )
                
                results = model(dose[i].item(), params, time_points)
                pred_conc = results["concentrations"]
                
                loss = compute_loss(pred_conc, true_conc[i], organ_weights)
                batch_loss += loss
            
            batch_loss = batch_loss / batch_size
            total_loss += batch_loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    return avg_loss, losses_per_organ


def train(
    data_path: Path,
    output_dir: Path,
    num_epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    train_split: float = 0.8,
    weight_organs: bool = True
) -> None:
    """
    Treina o Dynamic GNN.
    
    Args:
        data_path: Caminho para dataset
        output_dir: Diretório de saída
        num_epochs: Número de épocas
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device (cuda/cpu)
        train_split: Fração de treinamento
        weight_organs: Se True, dá mais peso a órgãos críticos
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(device)
    
    print("=" * 80)
    print("TREINAMENTO: Dynamic GNN para PBPK")
    print("=" * 80)
    print(f"\nConfiguração:")
    print(f"   Dataset: {data_path}")
    print(f"   Output: {output_dir}")
    print(f"   Épocas: {num_epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Device: {device}")
    print()
    
    # Carregar dataset
    print("📦 Carregando dataset...")
    full_dataset = PBPKDataset(data_path)
    
    # Split train/val
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"   Train: {len(train_dataset):,} amostras")
    print(f"   Val: {len(val_dataset):,} amostras")
    
    # Criar modelo
    print("\n🏗️  Criando modelo...")
    model = DynamicPBPKGNN(
        node_dim=16,
        edge_dim=4,
        hidden_dim=64,
        num_gnn_layers=3,
        num_temporal_steps=100,
        dt=0.1
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   Parâmetros: {num_params:,}")
    
    # Organ weights (dar mais peso a órgãos críticos)
    if weight_organs:
        organ_weights = torch.ones(NUM_ORGANS, device=device)
        critical_organs = ["blood", "liver", "kidney", "brain"]
        for organ in critical_organs:
            idx = PBPK_ORGANS.index(organ)
            organ_weights[idx] = 2.0  # 2x peso
    else:
        organ_weights = None
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    
    # Training loop
    print("\n🚀 Iniciando treinamento...")
    print()
    
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss, train_org_losses = train_epoch(
            model, train_loader, optimizer, device, organ_weights
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_org_losses = validate(model, val_loader, device, organ_weights)
        val_losses.append(val_loss)
        
        # Scheduler
        scheduler.step(val_loss)
        
        # Print
        print(f"   Train Loss: {train_loss:.6f}")
        print(f"   Val Loss: {val_loss:.6f}")
        
        # Salvar melhor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, output_dir / "best_model.pt")
            print(f"   ✅ Novo melhor modelo salvo! (Val Loss: {val_loss:.6f})")
        
        print()
    
    # Salvar modelo final
    torch.save({
        "epoch": num_epochs,
        "model_state_dict": model.state_dict(),
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
    plt.title("Training Progress")
    plt.savefig(output_dir / "training_curve.png", dpi=150)
    plt.close()
    
    print("=" * 80)
    print("✅ TREINAMENTO CONCLUÍDO!")
    print("=" * 80)
    print(f"\nMelhor Val Loss: {best_val_loss:.6f}")
    print(f"Modelos salvos em: {output_dir}")
    print(f"   - best_model.pt (melhor validação)")
    print(f"   - final_model.pt (última época)")
    print(f"   - training_curve.png (curva de treinamento)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina Dynamic GNN para PBPK")
    parser.add_argument("--data", type=str, required=True, help="Caminho para dataset .npz")
    parser.add_argument("--output", type=str, default="models/dynamic_gnn", help="Diretório de saída")
    parser.add_argument("--epochs", type=int, default=50, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/cpu/auto)")
    parser.add_argument("--train-split", type=float, default=0.8, help="Fração de treinamento")
    parser.add_argument("--no-organ-weights", action="store_true", help="Não usar pesos por órgão")
    
    args = parser.parse_args()
    
    # Auto-detect device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    train(
        data_path=Path(args.data),
        output_dir=Path(args.output),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=device,
        train_split=args.train_split,
        weight_organs=not args.no_organ_weights
    )

