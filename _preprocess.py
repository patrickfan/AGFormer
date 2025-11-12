import numpy as np
import os
import pickle
import json
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
import torch


def create_sliding_windows(data, _days_x=30, _days_y=7, _input_features=3):
    """Create sliding windows for X and y data
    Args:
        data (np.array): Input data array
        _days_x (int): Number of input days
        _days_y (int): Number of prediction days
        _input_features (int): Number of input features
    Returns:
        tuple: (X, y) where X is the input windows and y is the target windows
    """
    X, y = [], []
    for i in range(len(data) - _days_x - _days_y + 1):
        # Input window: _days_x days of input features
        X.append(data[i:i+_days_x, :_input_features])
        # Target window: _days_y days of inflow
        y.append(data[i+_days_x:i+_days_x+_days_y, -1])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_and_split_data(data_path, reservoir_name, _input_features=3, _days_x=30, _days_y=7):
    """Load and split data for a single reservoir based on time periods
    Args:
        data_path (str): Path to the data directory
        reservoir_name (str): Name of the reservoir CSV file
        _input_features (int): Number of input features
        _days_x (int): Number of input days
        _days_y (int): Number of prediction days
    Returns:
        tuple: (pretrain_data, train_data, test_data) as dictionaries containing X and y
    """
    df = pd.read_csv(os.path.join(data_path, reservoir_name))
    df['date'] = pd.to_datetime(df['date'])
    
    # Split data based on time periods
    pretrain_mask = df['date'] < '1999-01-01'
    train_mask = (df['date'] >= '1999-01-01') & (df['date'] <= '2008-12-31')
    test_mask = (df['date'] >= '2009-01-01') & (df['date'] <= '2011-12-31')
    
    # Extract features (include inflow_y exclude date)
    features = df.iloc[:, :_input_features+1].values.astype(np.float32)
    
    # Split and create windows for each period
    pretrain_data = features[pretrain_mask]
    train_data = features[train_mask]
    test_data = features[test_mask]
    
    # Create windows for each period
    pretrain_X, pretrain_y = create_sliding_windows(pretrain_data, _days_x, _days_y, _input_features) if len(pretrain_data) > 0 else (np.array([]), np.array([]))
    train_X, train_y = create_sliding_windows(train_data, _days_x, _days_y, _input_features)
    test_X, test_y = create_sliding_windows(test_data, _days_x, _days_y, _input_features)
    
    return {
        'pretrain': {'X': pretrain_X, 'y': pretrain_y},
        'train': {'X': train_X, 'y': train_y},
        'test': {'X': test_X, 'y': test_y}
    }


def preprocess_reservoir_data(data_path="./graph/data", output_path="./graph/data/parsed", _input_features=3, _days_x=30, _days_y=7, scaler_type="global"):
    """Preprocess all reservoir data with time-based splitting and sliding windows
    Args:
        data_path (str): Path to the input data directory
        output_path (str): Path to save processed data
        _input_features (int): Number of input features
        _days_x (int): Number of input days
        _days_y (int): Number of prediction days
        scaler_type (str): Type of scaler to use. Options: 'global' or 'local'
    Returns:
        dict: Dictionary containing processed data for all reservoirs
    """
    os.makedirs(output_path, exist_ok=True)
    align_path = os.path.join(data_path, "align")
    reservoir_files = [f for f in os.listdir(align_path) if f.endswith('.csv')]

    print("Loading and collecting training data...")
    all_train_X = []
    all_train_y = []
    reservoir_data = {}

    for rsr_file in tqdm(reservoir_files):
        rsr_name = rsr_file.split('.')[0]
        data = load_and_split_data(align_path, rsr_file, _input_features, _days_x, _days_y)
        reservoir_data[rsr_name] = data
        # Collect all training data for scaling
        if len(data['train']['X']) > 0:
            all_train_X.append(data['train']['X'].reshape(-1, _input_features))
            all_train_y.append(data['train']['y'].reshape(-1, 1))

    # Initialize scaler processing based on scaler_type
    if scaler_type == "global":
        # Global scaler: fit on all training data
        all_train_X = np.vstack(all_train_X)
        all_train_y = np.vstack(all_train_y)
        
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        
        # Fit scalers on training data
        scaler_X.fit(all_train_X)
        scaler_y.fit(all_train_y)

        print("Scaling data for each reservoir with global scaler...")
        processed_data = {}
        for rsr_name, data in reservoir_data.items():
            processed_data[rsr_name] = {
                'pretrain': {
                    'X': scaler_X.transform(data['pretrain']['X'].reshape(-1, _input_features)).reshape(data['pretrain']['X'].shape) if len(data['pretrain']['X']) > 0 else np.array([]),
                    'y': scaler_y.transform(data['pretrain']['y'].reshape(-1, 1)).reshape(data['pretrain']['y'].shape) if len(data['pretrain']['y']) > 0 else np.array([])
                },
                'train': {
                    'X': scaler_X.transform(data['train']['X'].reshape(-1, _input_features)).reshape(data['train']['X'].shape),
                    'y': scaler_y.transform(data['train']['y'].reshape(-1, 1)).reshape(data['train']['y'].shape)
                },
                'test': {
                    'X': scaler_X.transform(data['test']['X'].reshape(-1, _input_features)).reshape(data['test']['X'].shape),
                    'y': scaler_y.transform(data['test']['y'].reshape(-1, 1)).reshape(data['test']['y'].shape)
                }
            }
            if len(data['pretrain']['X']) == 0:
                print(f"Note: Reservoir {rsr_name} has no pretrain data (before 1999)")

        processed_data['scaler_X'] = scaler_X
        processed_data['scaler_y'] = scaler_y
    
    elif scaler_type == "local":
        # Local scaler: each reservoir has its own scaler
        print("Scaling data for each reservoir with local scalers...")
        processed_data = {}
        local_scalers_X = {}
        local_scalers_y = {}
        
        for rsr_name, data in reservoir_data.items():
            if len(data['train']['X']) > 0:
                # Create individual scalers for each reservoir
                scaler_X = MinMaxScaler(feature_range=(0, 1))
                scaler_y = MinMaxScaler(feature_range=(0, 1))
                
                # Fit scalers on this reservoir's training data
                train_X_reshaped = data['train']['X'].reshape(-1, _input_features)
                train_y_reshaped = data['train']['y'].reshape(-1, 1)
                
                scaler_X.fit(train_X_reshaped)
                scaler_y.fit(train_y_reshaped)
                
                # Store scalers for this reservoir
                local_scalers_X[rsr_name] = scaler_X
                local_scalers_y[rsr_name] = scaler_y
                
                # Transform data for this reservoir
                processed_data[rsr_name] = {
                    'pretrain': {
                        'X': scaler_X.transform(data['pretrain']['X'].reshape(-1, _input_features)).reshape(data['pretrain']['X'].shape) if len(data['pretrain']['X']) > 0 else np.array([]),
                        'y': scaler_y.transform(data['pretrain']['y'].reshape(-1, 1)).reshape(data['pretrain']['y'].shape) if len(data['pretrain']['y']) > 0 else np.array([])
                    },
                    'train': {
                        'X': scaler_X.transform(data['train']['X'].reshape(-1, _input_features)).reshape(data['train']['X'].shape),
                        'y': scaler_y.transform(data['train']['y'].reshape(-1, 1)).reshape(data['train']['y'].shape)
                    },
                    'test': {
                        'X': scaler_X.transform(data['test']['X'].reshape(-1, _input_features)).reshape(data['test']['X'].shape),
                        'y': scaler_y.transform(data['test']['y'].reshape(-1, 1)).reshape(data['test']['y'].shape)
                    }
                }
                
                if len(data['pretrain']['X']) == 0:
                    print(f"Note: Reservoir {rsr_name} has no pretrain data (before 1999)")
            else:
                print(f"Warning: Reservoir {rsr_name} has no training data, skipping")

        # Store local scalers instead of global ones
        processed_data['local_scalers_X'] = local_scalers_X
        processed_data['local_scalers_y'] = local_scalers_y
        processed_data['scaler_X'] = None  # For backward compatibility
        processed_data['scaler_y'] = None  # For backward compatibility
    
    else:
        raise ValueError(f"Invalid scaler_type: {scaler_type}. Must be 'global' or 'local'.")
    processed_data['params'] = {
        'input_features': _input_features,
        'days_x': _days_x,
        'days_y': _days_y,
        'scaler_type': scaler_type
    }
    
    output_file = os.path.join(output_path, f"all_rsr_data_{scaler_type}.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)

    print(f"Processed data saved to {output_file}")
    return processed_data


def prepare_input_supervised(input_data, encode_map, split): # split ∈ {"train","test"}
    xs, ys = [], []
    
    for name, _ in encode_map.items():
        node_dict = input_data[name][split] # {'X': [N, 30days, F], 'y': [N, 7days]}
        xs.append(torch.tensor(node_dict['X'], dtype=torch.float)) # [N, 30days, F]
        ys.append(torch.tensor(node_dict['y'], dtype=torch.float)) # [N, 7days]
    
    X = torch.stack(xs, dim=1) # [N, 30nodes, 30days, F]
    y = torch.stack(ys, dim=1) # [N, 30nodes, 7days]
    X = X.transpose(1, 2) # [N, 30days, num_nodes, F]
    return X, y


if __name__ == "__main__":
    data_path = "./data"  # ./graph/data - depends on root directory
    parsed_path = os.path.join(data_path, 'parsed')
    os.makedirs(parsed_path, exist_ok=True)
    graph_path = os.path.join(data_path, 'graph')
    os.makedirs(graph_path, exist_ok=True)
    
    # Generate data for both scaler types
    scaler_types = ["global", "local"]
    
    for scaler_type in scaler_types:
        print(f"\n{'='*60}")
        print(f"Processing data with {scaler_type.upper()} scaler...")
        print(f"{'='*60}")
        
        input_data = preprocess_reservoir_data(data_path, scaler_type=scaler_type)
        
        print(f"\nData processing statistics for {scaler_type} scaler:")
        excluded_keys = ['scaler_X', 'scaler_y', 'local_scalers_X', 'local_scalers_y', 'params']
        
        for rsr_name, data in input_data.items():
            if rsr_name not in excluded_keys:
                all_X = np.concatenate([data['pretrain']['X'].flatten(), data['train']['X'].flatten(), data['test']['X'].flatten()])
                all_y = np.concatenate([data['pretrain']['y'].flatten(), data['train']['y'].flatten(), data['test']['y'].flatten()])
                print(f"\n{rsr_name}: X range [{all_X.min():.4f}, {all_X.max():.4f}], y range [{all_y.min():.4f}, {all_y.max():.4f}]")
                print(f"  Pretrain set - X shape: {data['pretrain']['X'].shape}, y shape: {data['pretrain']['y'].shape}")
                print(f"  Train set - X shape: {data['train']['X'].shape}, y shape: {data['train']['y'].shape}")
                print(f"  Test set - X shape: {data['test']['X'].shape}, y shape: {data['test']['y'].shape}")
        
        # Prepare supervised data for each scaler type
        print(f"\nPreparing supervised data for {scaler_type} scaler...")
        
        # Prepare input data for GNN
        graph_cfg = "config3" # config1, config2, config3
        graph_file = os.path.join(graph_path, "%s.pkl" % graph_cfg)
        # Load Graph Data
        with open(graph_file, 'rb') as f:
            graph_data = pickle.load(f)
        # Retrieve Parsed Info
        A = graph_data['A']
        edge_index = graph_data['edge_index']
        encode_map = graph_data['encode_map']
        
        # Only support aligned temporal data - Train and Test
        X_train, y_train = prepare_input_supervised(input_data, encode_map, "train")
        X_test , y_test  = prepare_input_supervised(input_data, encode_map, "test")
        
        supervised_data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }
        
        supervised_file = os.path.join(parsed_path, f"_GNN_supervise_{scaler_type}.pt")
        torch.save(supervised_data, supervised_file)
        print(f"Supervised data saved to {supervised_file}")
    
    print(f"\n{'='*60}")
    print("All data processing completed!")
    print("Generated files:")
    print(f"  - all_rsr_data_global.pkl")
    print(f"  - all_rsr_data_local.pkl") 
    print(f"  - _GNN_supervise_global.pt")
    print(f"  - _GNN_supervise_local.pt")
    print(f"{'='*60}")
