import os
import random
import numpy as np
import torch
from torch_geometric.data import Data
from torch.utils.data import Dataset


def seed_everything(seed=42):
    """Set random seed for all libraries to ensure reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Random seed set to: {seed}")


class WindowDataset(Dataset):
    def __init__(self, X, y, edge_index): # X: [N, 30days, 30nodes, F], y: [N, 30nodes, 7days]
        self.X, self.y = X, y
        self.edge_index = edge_index

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        graphs = []
        for t in range(self.X.shape[1]): # 30 days
            x_t = self.X[idx, t] # [30 nodes, F]
            graphs.append(Data(x=x_t, edge_index=self.edge_index))
        target = self.y[idx] # [30 nodes, 7]
        return graphs, target # X [30 days, 30 nodes, F], y [30 nodes, 7 days]


class DynamicWindowDataset(Dataset):
    def __init__(self, X, y, edge_indices): # edge_indices: list of 30 edge_index tensors
        self.X, self.y = X, y
        self.edge_indices = edge_indices  # List of refined edge_index for each time step

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        graphs = []
        for t in range(self.X.shape[1]): # 30 days
            x_t = self.X[idx, t] # [30 nodes, F]
            # Use time-specific edge_index
            edge_index_t = self.edge_indices[t] if self.edge_indices and t < len(self.edge_indices) else self.edge_indices[0]
            graphs.append(Data(x=x_t, edge_index=edge_index_t))
        target = self.y[idx] # [30 nodes, 7]
        return graphs, target # X [30 days, 30 nodes, F], y [30 nodes, 7 days]


def inverse_transform_predictions(predictions, targets, scaler_data, encode_map=None):
    """
    Unified inverse transform function for both global and local scalers.
    
    Args:
        predictions (np.ndarray): Predictions array with shape (samples, nodes, 7)
        targets (np.ndarray): Targets array with shape (samples, nodes, 7)
        scaler_data (dict): Dictionary containing scaler information
        encode_map (dict): Optional mapping from reservoir name to node index
    
    Returns:
        tuple: (predictions_original, targets_original) - inverse transformed arrays
    """
    import numpy as np
    
    n_samples, n_nodes, n_days = predictions.shape
    scaler_type = scaler_data.get('params', {}).get('scaler_type', 'global')
    
    if scaler_type == 'global':
        # Use global scaler
        scaler_y = scaler_data['scaler_y']
        
        pred_reshaped = predictions.reshape(-1, 1)
        target_reshaped = targets.reshape(-1, 1)
        
        pred_inversed = scaler_y.inverse_transform(pred_reshaped)
        target_inversed = scaler_y.inverse_transform(target_reshaped)
        
        predictions_original = pred_inversed.reshape(n_samples, n_nodes, n_days)
        targets_original = target_inversed.reshape(n_samples, n_nodes, n_days)
        
    elif scaler_type == 'local':
        # Use local scalers
        local_scalers_y = scaler_data['local_scalers_y']
        
        # Create reverse mapping from node index to reservoir name
        idx_to_reservoir = {}
        if encode_map:
            for reservoir_name, node_idx in encode_map.items():
                idx_to_reservoir[node_idx] = reservoir_name
        
        predictions_original = np.zeros_like(predictions)
        targets_original = np.zeros_like(targets)
        
        for node_idx in range(n_nodes):
            # Get reservoir name using the reverse mapping
            reservoir_name = f"Node_{node_idx}"
            if node_idx in idx_to_reservoir:
                reservoir_name = idx_to_reservoir[node_idx]
            
            # Get the corresponding scaler for this reservoir
            if reservoir_name in local_scalers_y:
                scaler_y = local_scalers_y[reservoir_name]
                
                # Extract predictions and targets for this reservoir
                node_predictions = predictions[:, node_idx, :]  # Shape: (samples, 7)
                node_targets = targets[:, node_idx, :]         # Shape: (samples, 7)
                
                # Reshape for scaler
                pred_reshaped = node_predictions.reshape(-1, 1)
                target_reshaped = node_targets.reshape(-1, 1)
                
                # Inverse transform
                pred_inversed = scaler_y.inverse_transform(pred_reshaped)
                target_inversed = scaler_y.inverse_transform(target_reshaped)
                
                # Reshape back and store
                predictions_original[:, node_idx, :] = pred_inversed.reshape(n_samples, n_days)
                targets_original[:, node_idx, :] = target_inversed.reshape(n_samples, n_days)
            else:
                print(f"Warning: No scaler found for reservoir {reservoir_name}")
                print(f"Available scalers: {list(local_scalers_y.keys())}")
                # Keep original scaled values as fallback
                predictions_original[:, node_idx, :] = predictions[:, node_idx, :]
                targets_original[:, node_idx, :] = targets[:, node_idx, :]
    
    else:
        raise ValueError(f"Invalid scaler_type: {scaler_type}. Must be 'global' or 'local'.")
    
    return predictions_original, targets_original


def load_preprocessed_data(data_path, scaler_type="global"):
    """
    Helper function to load preprocessed data for a specific scaler type.
    
    Args:
        data_path (str): Path to the data directory
        scaler_type (str): Type of scaler to load ("global" or "local")
    
    Returns:
        tuple: (scaler_data, supervised_data) containing the loaded data
    """
    import os
    import pickle
    import torch
    
    parsed_path = os.path.join(data_path, 'parsed')
    
    # Load scaler data
    all_rsr_data_file = os.path.join(parsed_path, f"all_rsr_data_{scaler_type}.pkl")
    if not os.path.exists(all_rsr_data_file):
        raise FileNotFoundError(f"Preprocessed data not found: {all_rsr_data_file}. Please run _preprocess.py first.")
    
    with open(all_rsr_data_file, 'rb') as f:
        all_rsr_data = pickle.load(f)
    
    scaler_data = {
        'scaler_X': all_rsr_data.get('scaler_X'),
        'scaler_y': all_rsr_data.get('scaler_y'),
        'local_scalers_X': all_rsr_data.get('local_scalers_X'),
        'local_scalers_y': all_rsr_data.get('local_scalers_y'),
        'params': all_rsr_data.get('params', {})
    }
    
    # Load supervised data
    supervised_file = os.path.join(parsed_path, f"_GNN_supervise_{scaler_type}.pt")
    if not os.path.exists(supervised_file):
        raise FileNotFoundError(f"Supervised data not found: {supervised_file}. Please run _preprocess.py first.")
    
    supervised_data = torch.load(supervised_file, weights_only=True)
    
    print(f"Successfully loaded {scaler_type} scaler data:")
    print(f"  - Scaler data from: {all_rsr_data_file}")
    print(f"  - Supervised data from: {supervised_file}")
    print(f"  - Data shapes: X_train {supervised_data['X_train'].shape}, y_train {supervised_data['y_train'].shape}")
    
    return scaler_data, supervised_data


def check_available_data_files(data_path):
    """
    Check which scaler type data files are available.
    
    Args:
        data_path (str): Path to the data directory
        
    Returns:
        dict: Dictionary indicating which files are available
    """
    import os
    
    parsed_path = os.path.join(data_path, 'parsed')
    available = {}
    
    for scaler_type in ["global", "local"]:
        scaler_file = os.path.join(parsed_path, f"all_rsr_data_{scaler_type}.pkl")
        supervised_file = os.path.join(parsed_path, f"_GNN_supervise_{scaler_type}.pt")
        
        available[scaler_type] = {
            'scaler_data': os.path.exists(scaler_file),
            'supervised_data': os.path.exists(supervised_file),
            'complete': os.path.exists(scaler_file) and os.path.exists(supervised_file)
        }
    
    return available


def create_logging_directory(model_name, scaler_type="global"):
    """
    Create logging directory structure for training records.
    
    Args:
        model_name (str): Name of the model class
        scaler_type (str): Type of scaler ("global" or "local")
        
    Returns:
        tuple: (log_dir, timestamp) - directory path and timestamp string
    """
    import os
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    log_dir = os.path.join("logs", model_name, scaler_type)
    os.makedirs(log_dir, exist_ok=True)
    
    return log_dir, timestamp


def save_training_results(log_dir, results_text, timestamp):
    """
    Save training results to a file.
    
    Args:
        log_dir (str): Directory to save the results
        results_text (str): Training output text to save
        timestamp (str): Timestamp for the filename
    """
    import os
    
    results_file = os.path.join(log_dir, f"results_{timestamp}.txt")
    with open(results_file, 'w') as f:
        f.write(results_text)
    
    print(f"Training results saved to: {results_file}")


def save_best_checkpoint(log_dir, model, optimizer, epoch, loss, timestamp):
    """
    Save the best model checkpoint.
    
    Args:
        log_dir (str): Directory to save the checkpoint
        model: PyTorch model to save
        optimizer: Optimizer state
        epoch (int): Current epoch number
        loss (float): Current loss value
        timestamp (str): Timestamp for the filename
    """
    import os
    import torch
    
    checkpoint_file = os.path.join(log_dir, f"checkpoint_{timestamp}.pth")
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': timestamp
    }
    
    torch.save(checkpoint, checkpoint_file)
    print(f"Best checkpoint saved to: {checkpoint_file}")


class TrainingLogger:
    """
    A class to handle training logging and checkpointing.
    """
    
    def __init__(self, model_name, scaler_type="global", use_pretrain=False):
        self.model_name = model_name
        self.scaler_type = scaler_type
        self.use_pretrain = use_pretrain
        self.log_dir, self.timestamp = create_logging_directory(model_name, scaler_type)
        
        # Add "_p" suffix to timestamp if using pretrain
        if use_pretrain:
            self.timestamp = self.timestamp + "_p"
        
        self.best_val_loss = float('inf')
        self.training_logs = []
        
    def log_epoch(self, epoch, train_loss, val_loss, lr=None):
        """
        Log epoch information.
        
        Args:
            epoch (int): Current epoch
            train_loss (float): Training loss
            val_loss (float): Validation loss
            lr (float): Learning rate
        """
        log_entry = f'Epoch {epoch:3d}  '
        if lr is not None:
            log_entry += f'Learning Rate: {lr:.6f}  '
        log_entry += f'Train Loss: {train_loss:.6f}  Val Loss: {val_loss:.6f}'
        
        self.training_logs.append(log_entry)
        print(log_entry)
        
        # Save checkpoint if this is the best validation loss
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            return True
        return False
    
    def save_checkpoint(self, model, optimizer, epoch, loss):
        """
        Save model checkpoint.
        """
        save_best_checkpoint(self.log_dir, model, optimizer, epoch, loss, self.timestamp)
    
    def save_results(self, additional_info=""):
        """
        Save all training results to file.
        
        Args:
            additional_info (str): Additional information to include in results
        """
        results_text = f"Training Results for {self.model_name}\n"
        results_text += f"Timestamp: {self.timestamp}\n"
        results_text += f"Best Validation Loss: {self.best_val_loss:.6f}\n"
        results_text += "="*50 + "\n"
        results_text += "\n".join(self.training_logs)
        
        if additional_info:
            results_text += "\n" + "="*50 + "\n"
            results_text += additional_info
        
        save_training_results(self.log_dir, results_text, self.timestamp)


def _adjust_checkpoint_time(model_name, scaler_type, pretrain_time):
    """
    Adjust pretrain_time to find the earliest available checkpoint if the specified one doesn't exist.
    
    Args:
        model_name (str): Name of the model class
        scaler_type (str): Type of scaler ("global" or "local")
        pretrain_time (str): Original pretrain time string
        
    Returns:
        str: Adjusted pretrain time string or None if no checkpoints found
    """
    import os
    import re
    
    # First check if the original pretrain_time exists
    checkpoint_path = os.path.join("logs", model_name, scaler_type, f"checkpoint_{pretrain_time}.pth")
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at specified time: {pretrain_time}")
        return pretrain_time
    
    # If not found, look for available checkpoints in the directory
    checkpoint_dir = os.path.join("logs", model_name, scaler_type)
    if not os.path.exists(checkpoint_dir):
        print(f"Warning: Checkpoint directory does not exist: {checkpoint_dir}")
        return None
    
    # Find all checkpoint files
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.startswith("checkpoint_") and file.endswith(".pth"):
            # Extract timestamp from filename
            match = re.search(r'checkpoint_(\d+)\.pth', file)
            if match:
                timestamp = match.group(1)
                checkpoint_files.append(timestamp)
    
    if not checkpoint_files:
        print(f"Warning: No checkpoint files found in {checkpoint_dir}")
        return None
    
    # Sort timestamps to find the earliest one
    checkpoint_files.sort()
    earliest_time = checkpoint_files[0]
    
    print(f"Original pretrain_time '{pretrain_time}' not found.")
    print(f"Available checkpoints: {checkpoint_files}")
    print(f"Using earliest checkpoint: {earliest_time}")
    
    return earliest_time
