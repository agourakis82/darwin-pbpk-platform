"""
Training Pipeline for PINN
==========================

Week 2 - MOONSHOT Implementation
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

from .pinn_core import PBPKPhysicsInformedNN, PINNConfig
from .physics_loss import PhysicsLoss

logger = logging.getLogger(__name__)


def create_data_loaders(
    embeddings: np.ndarray,
    targets: np.ndarray,
    batch_size: int = 32,
    val_split: float = 0.2,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val data loaders.
    
    Args:
        embeddings: [N, 981] array of molecular embeddings
        targets: [N, 3] array of [fu, vd, cl] values
        batch_size: Batch size
        val_split: Validation split ratio
        seed: Random seed
        
    Returns:
        train_loader, val_loader
    """
    np.random.seed(seed)
    n_samples = len(embeddings)
    indices = np.random.permutation(n_samples)
    
    n_val = int(n_samples * val_split)
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    # Convert to tensors
    X_train = torch.FloatTensor(embeddings[train_indices])
    y_train = torch.FloatTensor(targets[train_indices])
    X_val = torch.FloatTensor(embeddings[val_indices])
    y_val = torch.FloatTensor(targets[val_indices])
    
    # Create datasets
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info(f"Created data loaders: train={len(train_dataset)}, val={len(val_dataset)}")
    
    return train_loader, val_loader


class PINNTrainer:
    """Trainer for PINN models"""
    
    def __init__(
        self,
        model: PBPKPhysicsInformedNN,
        config: PINNConfig,
        save_dir: str = "checkpoints",
        device: Optional[str] = None
    ):
        """
        Args:
            model: PINN model
            config: Model configuration
            save_dir: Directory for checkpoints
            device: Device ('cuda' or 'cpu')
        """
        self.model = model
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Loss function
        self.loss_fn = PhysicsLoss(alpha=1.0, beta=0.1, gamma=0.05)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        self.best_val_loss = float('inf')
        
        logger.info(f"PINNTrainer initialized on {self.device}")
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {'data': [], 'physics': [], 'boundary': [], 'total': []}
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            predictions = self.model(X_batch)
            
            # Prepare targets - map columns to active parameters only
            targets = {}
            active_params = []
            if self.config.predict_fu:
                active_params.append('fu')
            if self.config.predict_vd:
                active_params.append('vd')
            if self.config.predict_cl:
                active_params.append('cl')
            
            for i, param in enumerate(active_params):
                if i < y_batch.shape[1]:
                    targets[param] = y_batch[:, i:i+1]
            
            # Compute loss
            losses = self.loss_fn(predictions, targets, compute_physics=True)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track losses
            for key in epoch_losses:
                epoch_losses[key].append(losses[key].item())
        
        # Average losses
        return {k: np.mean(v) for k, v in epoch_losses.items()}
    
    def validate(self, val_loader: DataLoader) -> Tuple[Dict[str, float], Optional[Dict[str, float]]]:
        """Validate model"""
        self.model.eval()
        epoch_losses = {'data': [], 'physics': [], 'boundary': [], 'total': []}
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                # Forward pass
                predictions = self.model(X_batch)
                
                # Prepare targets - map columns to active parameters only
                targets = {}
                active_params = []
                if self.config.predict_fu:
                    active_params.append('fu')
                if self.config.predict_vd:
                    active_params.append('vd')
                if self.config.predict_cl:
                    active_params.append('cl')
                
                for i, param in enumerate(active_params):
                    if i < y_batch.shape[1]:
                        targets[param] = y_batch[:, i:i+1]
                
                # Compute loss
                losses = self.loss_fn(predictions, targets, compute_physics=True)
                
                # Track losses
                for key in epoch_losses:
                    epoch_losses[key].append(losses[key].item())
                
                # Collect for metrics - handle single parameter models
                pred_list = []
                if 'fu' in predictions:
                    pred_list.append(predictions['fu'])
                if 'vd' in predictions:
                    pred_list.append(predictions['vd'])
                if 'cl' in predictions:
                    pred_list.append(predictions['cl'])
                
                if pred_list:
                    all_predictions.append(torch.cat(pred_list, dim=1).cpu())
                    all_targets.append(y_batch.cpu())
        
        # Average losses
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        
        # Compute metrics
        if all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0).numpy()
            all_targets = torch.cat(all_targets, dim=0).numpy()
            
            metrics = {}
            # Determine which parameters are being predicted based on config
            active_params = []
            if self.config.predict_fu:
                active_params.append('fu')
            if self.config.predict_vd:
                active_params.append('vd')
            if self.config.predict_cl:
                active_params.append('cl')
            
            for i, param in enumerate(active_params):
                if i < all_predictions.shape[1]:
                    mse = np.mean((all_predictions[:, i] - all_targets[:, i])**2)
                    mae = np.mean(np.abs(all_predictions[:, i] - all_targets[:, i]))
                    
                    # R² score
                    ss_res = np.sum((all_targets[:, i] - all_predictions[:, i])**2)
                    ss_tot = np.sum((all_targets[:, i] - np.mean(all_targets[:, i]))**2)
                    r2 = 1 - (ss_res / (ss_tot + 1e-10))
                    
                    metrics[f'{param}_mse'] = mse
                    metrics[f'{param}_mae'] = mae
                    metrics[f'{param}_r2'] = r2
        else:
            metrics = {}
        
        return avg_losses, metrics
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        early_stopping_patience: int = 20,
        save_best: bool = True
    ) -> Dict[str, List]:
        """
        Train PINN model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            early_stopping_patience: Patience for early stopping
            save_best: Whether to save best model
            
        Returns:
            Training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        patience_counter = 0
        
        logger.info(f"Starting training for {epochs} epochs")
        
        for epoch in range(epochs):
            # Train
            train_losses = self.train_epoch(train_loader)
            
            # Validate
            val_losses, val_metrics = self.validate(val_loader)
            
            # Update scheduler
            self.scheduler.step(val_losses['total'])
            
            # Track history
            history['train_loss'].append(train_losses)
            history['val_loss'].append(val_losses)
            history['val_metrics'].append(val_metrics)
            
            # Log
            logger.info(
                f"Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {train_losses['total']:.4f}, "
                f"Val Loss: {val_losses['total']:.4f}"
            )
            
            # Log metrics
            if val_metrics:
                for key, value in val_metrics.items():
                    if 'r2' in key:
                        logger.info(f"  {key}: {value:.4f}")
            
            # Save best model
            if save_best and val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint('best_model.pt', {'epoch': epoch, 'val_loss': val_losses['total']})
                logger.info(f"  ⭐ New best model saved (val_loss={val_losses['total']:.4f})")
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        logger.info("Training complete!")
        return history
    
    def save_checkpoint(self, filename: str, metadata: Optional[Dict] = None):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'best_val_loss': self.best_val_loss,
        }
        
        if metadata:
            checkpoint['metadata'] = metadata
        
        path = self.save_dir / filename
        torch.save(checkpoint, path)
        logger.debug(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint"""
        path = self.save_dir / filename
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Checkpoint loaded from {path}")

