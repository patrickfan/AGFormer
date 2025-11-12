import os
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from utils import TrainingLogger, seed_everything


class PretrainDataset(Dataset):
    """
    Pretrain dataset: supports multi-reservoir random sampling
    Extracts 30-day windows from each reservoir's pretrain data
    """
    def __init__(self, data_dict, scaler_type="global"):
        self.data_dict = data_dict
        self.scaler_type = scaler_type
        self.reservoir_names = []
        self.reservoir_data = []
        
        # Collect all reservoirs with pretrain data
        for reservoir_name, data in data_dict.items():
            if isinstance(data, dict) and 'pretrain' in data:
                pretrain_data = data['pretrain']
                if len(pretrain_data['X']) > 0:
                    self.reservoir_names.append(reservoir_name)
                    self.reservoir_data.append(pretrain_data)
                    
        print(f"Found {len(self.reservoir_names)} reservoirs with pretrain data")
        
        # Create reservoir ID mapping
        self.reservoir_to_id = {name: idx for idx, name in enumerate(self.reservoir_names)}
        
        # Calculate total number of windows
        self.windows_per_reservoir = []
        self.total_windows = 0
        for data in self.reservoir_data:
            num_windows = len(data['X'])
            self.windows_per_reservoir.append(num_windows)
            self.total_windows += num_windows
            
    def __len__(self):
        return self.total_windows
    
    def __getitem__(self, idx):
        # Find corresponding reservoir and window index
        reservoir_idx = 0
        window_idx = idx
        
        for i, num_windows in enumerate(self.windows_per_reservoir):
            if window_idx < num_windows:
                reservoir_idx = i
                break
            window_idx -= num_windows
            
        # Get data
        reservoir_name = self.reservoir_names[reservoir_idx]
        reservoir_id = self.reservoir_to_id[reservoir_name]
        pretrain_data = self.reservoir_data[reservoir_idx]
        
        # Window features: (30 days, 3 features)
        window = torch.tensor(pretrain_data['X'][window_idx], dtype=torch.float32)
        
        # Target: (7 days)
        target = torch.tensor(pretrain_data['y'][window_idx], dtype=torch.float32)
        
        return {
            'window': window,           # (30, 3)
            'reservoir_id': reservoir_id,  # scalar
            'target': target           # (7,)
        }


class WindowEmbedding(nn.Module):
    """
    Window embedding module using shared MLP encoder
    """
    def __init__(self, input_dim=3, hidden_dim=64, dropout=0.2):
        super().__init__()
        # Shared MLP encoder (consistent with existing models)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, windows):
        """
        Args:
            windows: (B, 30, 3) - batch of window data
        Returns:
            embeddings: (B, hidden_dim) - window embeddings
        """
        batch_size, seq_len, input_dim = windows.shape
        
        # Flatten windows to (B*30, 3)
        windows_flat = windows.view(-1, input_dim)
        
        # Pass through shared MLP encoder
        daily_embeddings = self.encoder(windows_flat)  # (B*30, hidden_dim)
        
        # Reshape to (B, 30, hidden_dim)
        daily_embeddings = daily_embeddings.view(batch_size, seq_len, -1)
        
        # Average over time dimension
        window_embeddings = daily_embeddings.mean(dim=1)  # (B, hidden_dim)
        
        return window_embeddings


class PrototypeDict(nn.Module):
    """
    Prototype dictionary with momentum-updated embedding means for each reservoir
    """
    def __init__(self, num_reservoirs, embedding_dim, momentum=0.999):
        super().__init__()
        self.num_reservoirs = num_reservoirs
        self.embedding_dim = embedding_dim
        self.momentum = momentum
        
        # Register prototype vector for each reservoir
        self.register_buffer('prototypes', torch.zeros(num_reservoirs, embedding_dim))
        self.register_buffer('initialized', torch.zeros(num_reservoirs, dtype=torch.bool))
        
    def update_prototypes(self, embeddings, reservoir_ids):
        """Update prototype dictionary with momentum"""
        for i, reservoir_id in enumerate(reservoir_ids):
            embedding = embeddings[i]
            
            if not self.initialized[reservoir_id]:
                # First initialization
                self.prototypes[reservoir_id] = embedding.detach()
                self.initialized[reservoir_id] = True
            else:
                # Momentum update
                self.prototypes[reservoir_id] = (
                    self.momentum * self.prototypes[reservoir_id] + 
                    (1 - self.momentum) * embedding.detach()
                )
    
    def get_prototypes(self, reservoir_ids):
        """Get prototype vectors for specified reservoirs"""
        return self.prototypes[reservoir_ids]


class InfoNCELoss(nn.Module):
    """
    InfoNCE contrastive loss: same reservoir windows as positive samples
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, embeddings, reservoir_ids):
        """
        Args:
            embeddings: (B, embedding_dim) - window embeddings
            reservoir_ids: (B,) - reservoir IDs
        Returns:
            loss: InfoNCE contrastive loss
        """
        batch_size = embeddings.shape[0]
        
        # Compute similarity matrix with normalized embeddings
        embeddings_norm = F.normalize(embeddings, dim=1)
        similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.t()) / self.temperature
        
        # Create label matrix: same reservoir samples as positive
        labels = reservoir_ids.unsqueeze(0) == reservoir_ids.unsqueeze(1)  # (B, B)
        
        # Remove diagonal (self)
        mask = torch.eye(batch_size, device=embeddings.device, dtype=torch.bool)
        labels = labels & ~mask
        
        # Compute InfoNCE loss
        losses = []
        for i in range(batch_size):
            # Find positive samples
            positive_mask = labels[i]
            if positive_mask.sum() == 0:
                continue  # No positive samples, skip
                
            # Compute numerator: similarity with positive samples
            positive_similarities = similarity_matrix[i, positive_mask]
            
            # Compute denominator: similarity with all samples (except self)
            all_similarities = similarity_matrix[i, ~mask[i]]
            
            # InfoNCE loss
            for pos_sim in positive_similarities:
                loss = -torch.log(
                    torch.exp(pos_sim) / torch.exp(all_similarities).sum()
                )
                losses.append(loss)
        
        if losses:
            return torch.stack(losses).mean()
        else:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)


class PretrainModel(nn.Module):
    """
    Pretrain model: window embedding + prototype dict + prediction head
    """
    def __init__(self, input_dim=3, hidden_dim=64, num_reservoirs=30, 
                 pred_days=7, dropout=0.2, momentum=0.999):
        super().__init__()
        self.window_embedding = WindowEmbedding(input_dim, hidden_dim, dropout)
        self.prototype_dict = PrototypeDict(num_reservoirs, hidden_dim, momentum)
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_days)
        )
        
    def forward(self, windows, reservoir_ids, update_prototypes=True):
        """
        Args:
            windows: (B, 30, 3) - window data
            reservoir_ids: (B,) - reservoir IDs
            update_prototypes: bool - whether to update prototype dict
        Returns:
            predictions: (B, 7) - predictions
            embeddings: (B, hidden_dim) - window embeddings
        """
        # Get window embeddings
        embeddings = self.window_embedding(windows)
        
        # Update prototype dictionary
        if update_prototypes and self.training:
            self.prototype_dict.update_prototypes(embeddings, reservoir_ids)
        
        # Predictions
        predictions = self.prediction_head(embeddings)
        
        return predictions, embeddings


def create_data_loader(data_dict, batch_size=32, shuffle=True, scaler_type="global"):
    """Create pretrain data loader"""
    dataset = PretrainDataset(data_dict, scaler_type)
    
    def collate_fn(batch):
        windows = torch.stack([item['window'] for item in batch])
        reservoir_ids = torch.tensor([item['reservoir_id'] for item in batch])
        targets = torch.stack([item['target'] for item in batch])
        
        return {
            'windows': windows,
            'reservoir_ids': reservoir_ids,
            'targets': targets
        }
    
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                       collate_fn=collate_fn, num_workers=0)
    
    return loader, dataset.reservoir_names


def get_pretrain_config():
    """Get pretrain configuration"""
    config = {
        'model': {
            'input_dim': 3,
            'hidden_dim': 128,
            'pred_days': 7,
            'dropout': 0.2,
            'momentum': 0.999
        },
        'training': {
            'batch_size': 32,
            'learning_rate': 1e-3,
            'epochs': 5,
            'weight_decay': 1e-4
        },
        'loss': {
            'temperature': 0.1,
            'supervised_weight': 0.8,
            'contrastive_weight': 0.2
        },
        'data': {
            'scaler_type': 'local'  # ['global', 'local']
        }
    }
    
    return config


def train_pretrain_model(config, data_dict, device):
    """Train pretrain model"""
    # Create data loader
    train_loader, reservoir_names = create_data_loader(
        data_dict,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        scaler_type=config['data']['scaler_type']
    )
    
    print(f"Training with {len(reservoir_names)} reservoirs")
    print(f"Reservoir names: {reservoir_names}")
    
    # Create model
    model = PretrainModel(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        num_reservoirs=len(reservoir_names),
        pred_days=config['model']['pred_days'],
        dropout=config['model']['dropout'],
        momentum=config['model']['momentum']
    ).to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    mse_loss = nn.MSELoss()
    contrastive_loss = InfoNCELoss(temperature=config['loss']['temperature'])
    
    # Training logger
    logger = TrainingLogger("PretrainModel", config['data']['scaler_type'])
    
    print(f"Starting pretraining for {config['training']['epochs']} epochs")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    model.train()
    for epoch in range(1, config['training']['epochs'] + 1):
        total_loss = 0
        supervised_losses = []
        contrastive_losses = []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            windows = batch['windows'].to(device)
            reservoir_ids = batch['reservoir_ids'].to(device)
            targets = batch['targets'].to(device)
            optimizer.zero_grad()
            predictions, embeddings = model(windows, reservoir_ids)
            supervised_loss = mse_loss(predictions, targets)
            contrastive_loss_val = contrastive_loss(embeddings, reservoir_ids)
            total_loss_val = (
                config['loss']['supervised_weight'] * supervised_loss +
                config['loss']['contrastive_weight'] * contrastive_loss_val
            )
            total_loss_val.backward()
            optimizer.step()

            total_loss += total_loss_val.item()
            supervised_losses.append(supervised_loss.item())
            contrastive_losses.append(contrastive_loss_val.item())

            pbar.set_postfix({
                'Total': f'{total_loss_val.item():.4f}',
                'Sup': f'{supervised_loss.item():.4f}',
                'Con': f'{contrastive_loss_val.item():.4f}'
            })

        avg_loss = total_loss / len(train_loader)
        avg_supervised = np.mean(supervised_losses)
        avg_contrastive = np.mean(contrastive_losses)

        # Log epoch and save checkpoint if best
        is_best = logger.log_epoch(epoch, avg_loss, avg_loss)
        if is_best:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'reservoir_names': reservoir_names,
                'config': config
            }, logger.log_dir + f"/pretrain_best_model.pth")
        
        print(f"Epoch {epoch}: Total Loss: {avg_loss:.4f}, "
              f"Supervised: {avg_supervised:.4f}, Contrastive: {avg_contrastive:.4f}")

    # Save training results
    results_info = f"Pretrain Model Training Results\n"
    results_info += f"Epochs: {config['training']['epochs']}\n"
    results_info += f"Best Loss: {logger.best_val_loss:.4f}\n"
    results_info += f"Supervised Weight: {config['loss']['supervised_weight']}\n"
    results_info += f"Contrastive Weight: {config['loss']['contrastive_weight']}\n"
    results_info += f"Temperature: {config['loss']['temperature']}\n"
    results_info += f"Reservoirs: {len(reservoir_names)}\n"
    logger.save_results(results_info)
    
    return model, reservoir_names


def main():
    seed_everything(42)
    config = get_pretrain_config()
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    data_path = "./data"
    parsed_path = os.path.join(data_path, 'parsed')
    scaler_type = config['data']['scaler_type']
    all_rsr_data_file = os.path.join(parsed_path, f"all_rsr_data_{scaler_type}.pkl")

    if not os.path.exists(all_rsr_data_file):
        raise FileNotFoundError(f"Preprocessed data not found: {all_rsr_data_file}. "
                               f"Please run _preprocess.py first.")

    with open(all_rsr_data_file, 'rb') as f:
        all_rsr_data = pickle.load(f)

    pretrain_data = {}
    excluded_keys = ['scaler_X', 'scaler_y', 'local_scalers_X', 'local_scalers_y', 'params']

    for reservoir_name, data in all_rsr_data.items():
        if reservoir_name not in excluded_keys:
            if isinstance(data, dict) and 'pretrain' in data:
                if len(data['pretrain']['X']) > 0:
                    pretrain_data[reservoir_name] = data

    print(f"Found {len(pretrain_data)} reservoirs with pretrain data")

    if len(pretrain_data) == 0:
        raise ValueError("No pretrain data found. Please check the data preprocessing.")

    _, reservoir_names = train_pretrain_model(config, pretrain_data, device)

    print("Pretraining completed successfully!")
    print(f"Model saved to: ./logs/PretrainModel/")
    print(f"Trained on {len(reservoir_names)} reservoirs")


if __name__ == "__main__":
    main()
