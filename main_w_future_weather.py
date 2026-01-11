import numpy as np
import os
import pickle
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import r2_score

from utils import seed_everything, TrainingLogger, inverse_transform_predictions, _adjust_checkpoint_time
from models.gnn import ReservoirAttentionNet_withFuture


def load_weather_data():
    """Load weather prediction data from _get_weather_prediction.py output"""
    weather_file = './data/weather/weather_data_2009_2011.pkl'
    if not os.path.exists(weather_file):
        raise FileNotFoundError(f"Weather data file not found: {weather_file}")
    
    with open(weather_file, 'rb') as f:
        weather_data = pickle.load(f)
    
    print(f"Loaded weather data: {weather_data.shape}")
    return weather_data


def get_historical_weather_data(data_path, reservoir_coords, years):
    """
    Extract historical weather data from the original reservoir CSV files
    for training period (1999-2008).
    Assumes temperature and precipitation are available in the CSV files.
    """
    historical_weather = {}
    align_path = os.path.join(data_path, "align")
    
    for year in years:
        for reservoir_name in reservoir_coords:
            csv_file = os.path.join(align_path, f"{reservoir_name}.csv")
            if os.path.exists(csv_file):
                df = pd.read_csv(csv_file)
                df['date'] = pd.to_datetime(df['date'])
                
                # Filter for the specific year
                year_mask = df['date'].dt.year == year
                year_data = df[year_mask]
                
                if len(year_data) > 0:
                    # CSV columns are: temperature, perception, inflow_x, inflow_y, date
                    # Extract temperature and precipitation (columns 0 and 1)
                    if year not in historical_weather:
                        historical_weather[year] = {}
                    
                    historical_weather[year][reservoir_name] = {
                        'dates': year_data['date'].values,
                        'temperature': year_data.iloc[:, 0].values,  # Column 0 (temperature)
                        'precipitation': year_data.iloc[:, 1].values  # Column 1 (perception/precipitation)
                    }
    
    return historical_weather


def create_sliding_windows_with_weather(data, weather_data, _days_x=30, _days_y=7, _input_features=3):
    """
    Create sliding windows for X and y data including weather features
    Args:
        data (np.array): Input data array [samples, features] (original reservoir data)
        weather_data (np.array): Weather data array [samples, 2] (temp, precip)
        _days_x (int): Number of input days
        _days_y (int): Number of prediction days  
        _input_features (int): Number of input features (original features, not including weather)
    Returns:
        tuple: (X, y, weather_future) where:
               X is the input windows [samples, _days_x, _input_features]
               y is the target windows [samples, _days_y]
               weather_future is the future weather [samples, _days_y, 2]
    """
    X, y, weather_future = [], [], []
    
    for i in range(len(data) - _days_x - _days_y + 1):
        # Input window: _days_x days of original input features
        X.append(data[i:i+_days_x, :_input_features])
        
        # Target window: _days_y days of inflow
        y.append(data[i+_days_x:i+_days_x+_days_y, -1])
        
        # Future weather window: _days_y days of weather features
        weather_future.append(weather_data[i+_days_x:i+_days_x+_days_y, :])
    
    return (np.array(X, dtype=np.float32), 
            np.array(y, dtype=np.float32), 
            np.array(weather_future, dtype=np.float32))


def load_and_split_data_with_weather(data_path, reservoir_name, historical_weather_dict, 
                                   future_weather_df, _input_features=3, _days_x=30, _days_y=7):
    """
    Load and split data for a single reservoir with weather features
    Args:
        data_path (str): Path to the data directory
        reservoir_name (str): Name of the reservoir CSV file
        historical_weather_dict (dict): Historical weather data for training
        future_weather_df (DataFrame): Future weather predictions for testing
        _input_features (int): Number of input features (original features)
        _days_x (int): Number of input days
        _days_y (int): Number of prediction days
    Returns:
        dict: Dictionary containing processed data with weather features
    """
    # Load original reservoir data
    df = pd.read_csv(os.path.join(data_path, reservoir_name))
    df['date'] = pd.to_datetime(df['date'])
    
    # Split data based on time periods
    train_mask = (df['date'] >= '1999-01-01') & (df['date'] <= '2008-12-31')
    test_mask = (df['date'] >= '2009-01-01') & (df['date'] <= '2011-12-31')
    
    # Extract original features (exclude weather columns that may exist)
    features = df.iloc[:, :_input_features+1].values.astype(np.float32)
    
    train_data = features[train_mask]
    test_data = features[test_mask]
    train_dates = df.loc[train_mask, 'date'].values
    test_dates = df.loc[test_mask, 'date'].values
    
    # Prepare weather data for training period (use historical/true weather)
    train_weather = []
    for date in train_dates:
        date_obj = pd.to_datetime(date)
        year = date_obj.year
        
        # Find weather data for this date from historical data
        if year in historical_weather_dict and reservoir_name in historical_weather_dict[year]:
            hist_data = historical_weather_dict[year][reservoir_name]
            # Find closest date
            date_diffs = np.abs((pd.to_datetime(hist_data['dates']) - date_obj).days)
            closest_idx = np.argmin(date_diffs)
            temp = hist_data['temperature'][closest_idx]
            precip = hist_data['precipitation'][closest_idx]
        else:
            # Use values from the original CSV if available (columns 0, 1) - no unit conversion needed
            original_idx = df.index[df['date'] == date].tolist()
            if original_idx:
                temp = df.iloc[original_idx[0], 0]  # Column 0 (temperature)
                precip = df.iloc[original_idx[0], 1]  # Column 1 (precipitation)
            else:
                temp, precip = 0.0, 0.0  # Fallback
        
        train_weather.append([temp, precip])
    
    train_weather = np.array(train_weather, dtype=np.float32)
    
    # Prepare weather data for testing period (use future weather predictions)
    test_weather = []
    reservoir_weather = future_weather_df[future_weather_df['reservoir'] == reservoir_name]
    
    for date in test_dates:
        date_str = pd.to_datetime(date).strftime('%Y-%m-%d')
        weather_row = reservoir_weather[reservoir_weather['date'] == date_str]
        
        if len(weather_row) > 0:
            # Apply unit conversion for predicted weather data
            temp = weather_row.iloc[0]['temperature'] - 273.15  # Kelvin to Celsius
            precip = weather_row.iloc[0]['precipitation'] * 1000  # m to mm
        else:
            # Fallback to using original data if prediction not available (no unit conversion needed)
            original_idx = df.index[df['date'] == date].tolist()
            if original_idx:
                temp = df.iloc[original_idx[0], 0]  # Column 0 (temperature)
                precip = df.iloc[original_idx[0], 1]  # Column 1 (precipitation)
            else:
                temp, precip = 0.0, 0.0
        
        test_weather.append([temp, precip])
    
    test_weather = np.array(test_weather, dtype=np.float32)
    
    # Create windows with weather features
    if len(train_data) > _days_x + _days_y:
        train_X, train_y, train_weather_future = create_sliding_windows_with_weather(
            train_data, train_weather, _days_x, _days_y, _input_features)
    else:
        train_X = train_y = train_weather_future = np.array([])
    
    if len(test_data) > _days_x + _days_y:
        test_X, test_y, test_weather_future = create_sliding_windows_with_weather(
            test_data, test_weather, _days_x, _days_y, _input_features)
    else:
        test_X = test_y = test_weather_future = np.array([])
    
    return {
        'train': {'X': train_X, 'y': train_y, 'weather_future': train_weather_future},
        'test': {'X': test_X, 'y': test_y, 'weather_future': test_weather_future}
    }


def preprocess_reservoir_data_with_weather(data_path="./data", output_path="./data/parsed", 
                                         _input_features=3, _days_x=30, _days_y=7, scaler_type="global"):
    """
    Preprocess all reservoir data with weather features
    """
    os.makedirs(output_path, exist_ok=True)
    align_path = os.path.join(data_path, "align")
    reservoir_files = [f for f in os.listdir(align_path) if f.endswith('.csv')]
    
    # Load weather prediction data for testing
    future_weather_df = load_weather_data()
    
    # Get reservoir coordinates for historical weather
    reservoir_coords = {}
    for rsr_file in reservoir_files:
        rsr_name = rsr_file.split('.')[0]
        reservoir_coords[rsr_name] = None  # We'll use data from CSV files directly
    
    # Get historical weather data for training (1999-2008)
    print("Loading historical weather data for training period...")
    train_years = list(range(1999, 2009))
    historical_weather_dict = get_historical_weather_data(data_path, reservoir_coords.keys(), train_years)
    
    print("Processing reservoir data with weather features...")
    all_train_X = []
    all_train_y = []
    all_train_weather = []
    reservoir_data = {}
    
    for rsr_file in tqdm(reservoir_files):
        rsr_name = rsr_file.split('.')[0]
        data = load_and_split_data_with_weather(
            align_path, rsr_file, historical_weather_dict, future_weather_df, 
            _input_features, _days_x, _days_y)
        reservoir_data[rsr_name] = data
        
        # Collect all training data for scaling
        if len(data['train']['X']) > 0:
            all_train_X.append(data['train']['X'].reshape(-1, _input_features))
            all_train_y.append(data['train']['y'].reshape(-1, 1))
            all_train_weather.append(data['train']['weather_future'].reshape(-1, 2))
    
    # Initialize scalers
    if scaler_type == "global":
        # Global scalers: fit on all training data
        all_train_X = np.vstack(all_train_X)
        all_train_y = np.vstack(all_train_y)
        all_train_weather = np.vstack(all_train_weather)
        
        scaler_X = MinMaxScaler(feature_range=(0, 1))
        scaler_y = MinMaxScaler(feature_range=(0, 1))
        scaler_weather = MinMaxScaler(feature_range=(0, 1))
        
        # Fit scalers on training data
        scaler_X.fit(all_train_X)
        scaler_y.fit(all_train_y)
        scaler_weather.fit(all_train_weather)
        
        print("Scaling data for each reservoir with global scalers...")
        processed_data = {}
        for rsr_name, data in reservoir_data.items():
            processed_data[rsr_name] = {}
            for split in ['train', 'test']:
                if len(data[split]['X']) > 0:
                    # Scale X
                    X_scaled = scaler_X.transform(data[split]['X'].reshape(-1, _input_features)).reshape(data[split]['X'].shape)
                    # Scale y
                    y_scaled = scaler_y.transform(data[split]['y'].reshape(-1, 1)).reshape(data[split]['y'].shape)
                    # Scale weather features
                    weather_scaled = scaler_weather.transform(data[split]['weather_future'].reshape(-1, 2)).reshape(data[split]['weather_future'].shape)
                    
                    processed_data[rsr_name][split] = {
                        'X': X_scaled,
                        'y': y_scaled,
                        'weather_future': weather_scaled
                    }
                else:
                    processed_data[rsr_name][split] = {
                        'X': np.array([]),
                        'y': np.array([]),
                        'weather_future': np.array([])
                    }
        
        # Store scalers
        processed_data['scaler_X'] = scaler_X
        processed_data['scaler_y'] = scaler_y
        processed_data['scaler_weather'] = scaler_weather
        processed_data['local_scalers_X'] = None
        processed_data['local_scalers_y'] = None
        processed_data['local_scalers_weather'] = None
        
    elif scaler_type == "local":
        # Local scalers: each reservoir has its own scalers
        print("Scaling data for each reservoir with local scalers...")
        processed_data = {}
        local_scalers_X = {}
        local_scalers_y = {}
        local_scalers_weather = {}
        
        for rsr_name, data in reservoir_data.items():
            if len(data['train']['X']) > 0:
                # Create individual scalers for each reservoir
                scaler_X = MinMaxScaler(feature_range=(0, 1))
                scaler_y = MinMaxScaler(feature_range=(0, 1))
                scaler_weather = MinMaxScaler(feature_range=(0, 1))
                
                # Fit scalers on this reservoir's training data
                train_X_reshaped = data['train']['X'].reshape(-1, _input_features)
                train_y_reshaped = data['train']['y'].reshape(-1, 1)
                train_weather_reshaped = data['train']['weather_future'].reshape(-1, 2)
                
                scaler_X.fit(train_X_reshaped)
                scaler_y.fit(train_y_reshaped)
                scaler_weather.fit(train_weather_reshaped)
                
                # Store scalers for this reservoir
                local_scalers_X[rsr_name] = scaler_X
                local_scalers_y[rsr_name] = scaler_y
                local_scalers_weather[rsr_name] = scaler_weather
                
                # Transform data for this reservoir
                processed_data[rsr_name] = {}
                for split in ['train', 'test']:
                    if len(data[split]['X']) > 0:
                        X_scaled = scaler_X.transform(data[split]['X'].reshape(-1, _input_features)).reshape(data[split]['X'].shape)
                        y_scaled = scaler_y.transform(data[split]['y'].reshape(-1, 1)).reshape(data[split]['y'].shape)
                        weather_scaled = scaler_weather.transform(data[split]['weather_future'].reshape(-1, 2)).reshape(data[split]['weather_future'].shape)
                        
                        processed_data[rsr_name][split] = {
                            'X': X_scaled,
                            'y': y_scaled,
                            'weather_future': weather_scaled
                        }
                    else:
                        processed_data[rsr_name][split] = {
                            'X': np.array([]),
                            'y': np.array([]),
                            'weather_future': np.array([])
                        }
            else:
                print(f"Warning: Reservoir {rsr_name} has no training data, skipping")
        
        # Store local scalers
        processed_data['scaler_X'] = None  # For backward compatibility
        processed_data['scaler_y'] = None  # For backward compatibility
        processed_data['scaler_weather'] = None
        processed_data['local_scalers_X'] = local_scalers_X
        processed_data['local_scalers_y'] = local_scalers_y
        processed_data['local_scalers_weather'] = local_scalers_weather
    
    else:
        raise ValueError(f"Invalid scaler_type: {scaler_type}. Must be 'global' or 'local'.")
    
    # Store parameters
    processed_data['params'] = {
        'input_features': _input_features,
        'days_x': _days_x,
        'days_y': _days_y,
        'scaler_type': scaler_type,
        'weather_features': 2
    }
    
    # Save processed data
    output_file = os.path.join(output_path, f"weather_enhanced_data_{scaler_type}.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump(processed_data, f)
    
    print(f"Weather-enhanced data saved to {output_file}")
    return processed_data


class WeatherEnhancedDataset:
    """Dataset class for weather-enhanced reservoir data"""
    def __init__(self, X, y, weather_future, edge_index):
        self.X = X  # [samples, days, nodes, features]
        self.y = y  # [samples, nodes, pred_days]
        self.weather_future = weather_future  # [samples, pred_days, nodes, weather_features]
        self.edge_index = edge_index
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        from torch_geometric.data import Data
        
        # Create graph sequence for this sample
        graphs = []
        for day in range(self.X.shape[1]):  # 30 days
            node_features = self.X[idx, day]  # [nodes, features]
            if isinstance(node_features, torch.Tensor):
                node_features = node_features.clone().detach()
            else:
                node_features = torch.tensor(node_features, dtype=torch.float32)
            graph = Data(x=node_features, edge_index=self.edge_index)
            graphs.append(graph)
        
        if isinstance(self.y[idx], torch.Tensor):
            targets = self.y[idx].clone().detach()
        else:
            targets = torch.tensor(self.y[idx], dtype=torch.float32)
        
        if isinstance(self.weather_future[idx], torch.Tensor):
            weather = self.weather_future[idx].clone().detach()
        else:
            weather = torch.tensor(self.weather_future[idx], dtype=torch.float32)
        
        # weather is [pred_days, nodes, weather_features], need to transpose to [nodes, pred_days, weather_features]
        weather = weather.transpose(0, 1)  # [nodes, pred_days, weather_features]
        
        return graphs, targets, weather


def prepare_weather_enhanced_data(input_data, encode_map, split):
    """Prepare input data with weather features for GNN"""
    xs, ys, weather_futures = [], [], []
    
    for name, _ in encode_map.items():
        node_dict = input_data[name][split]
        xs.append(torch.tensor(node_dict['X'], dtype=torch.float))  # [N, 30days, F]
        ys.append(torch.tensor(node_dict['y'], dtype=torch.float))  # [N, 7days]
        weather_futures.append(torch.tensor(node_dict['weather_future'], dtype=torch.float))  # [N, 7days, 2]
    
    X = torch.stack(xs, dim=1)  # [N, 30nodes, 30days, F]
    y = torch.stack(ys, dim=1)  # [N, 30nodes, 7days]
    weather_future = torch.stack(weather_futures, dim=1)  # [N, 30nodes, 7days, 2]
    
    X = X.transpose(1, 2)  # [N, 30days, num_nodes, F]
    weather_future = weather_future.transpose(1, 2)  # [N, 7days, num_nodes, 2]
    
    return X, y, weather_future


def run_epoch_with_weather(model, loader, criterion, optimizer=None, device='cpu', train=True):
    """Run one epoch with weather-enhanced model"""
    model.train() if train else model.eval()
    total_loss = 0
    phase = "Train" if train else "Val"
    pbar = tqdm(loader, desc=f"{phase} Epoch", leave=False)
    
    for batch_idx, (graph_batch, target_batch, weather_batch) in enumerate(pbar):
        preds, targets = [], []
        
        for graphs, tgt, weather in zip(graph_batch, target_batch, weather_batch):
            graphs = [g.to(device) for g in graphs]
            weather = weather.to(device)  # [nodes, pred_days, weather_features]
            
            # Model forward pass with weather features
            pred = model(graphs, weather_features=weather)
            preds.append(pred)
            targets.append(tgt.to(device))
        
        preds = torch.stack(preds)  # [B, nodes, 7]
        targets = torch.stack(targets)  # [B, nodes, 7]
        loss = criterion(preds, targets)
        
        if train and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item() * preds.size(0)
        current_avg_loss = total_loss / ((batch_idx + 1) * len(graph_batch))
        pbar.set_postfix({"Loss": f"{current_avg_loss:.6f}"})
    
    return total_loss / len(loader.dataset)


def evaluate_weather_enhanced_model(model, test_loader, encode_map, scaler_data, device='cpu'):
    """Evaluate weather-enhanced model"""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for graph_batch, target_batch, weather_batch in test_loader:
            preds, targets = [], []
            
            for graphs, tgt, weather in zip(graph_batch, target_batch, weather_batch):
                graphs = [g.to(device) for g in graphs]
                weather = weather.to(device)  # [nodes, pred_days, weather_features]
                
                pred = model(graphs, weather_features=weather)
                preds.append(pred)
                targets.append(tgt.to(device))
            
            preds = torch.stack(preds)  # [B, nodes, 7]
            targets = torch.stack(targets)  # [B, nodes, 7]
            all_predictions.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0)  # Shape: (samples, nodes, 7)
    all_targets = np.concatenate(all_targets, axis=0)         # Shape: (samples, nodes, 7)
    
    n_samples = all_predictions.shape[0]
    n_nodes = all_predictions.shape[1]
    n_days = all_predictions.shape[2]
    
    # Create reverse mapping from node index to reservoir name
    idx_to_reservoir = {}
    if encode_map:
        for reservoir_name, node_idx in encode_map.items():
            idx_to_reservoir[node_idx] = reservoir_name
    
    print(f"\nWeather-Enhanced Model Evaluation Results:")
    print(f"Total samples: {n_samples}")
    print(f"Total nodes (reservoirs): {n_nodes}")
    print(f"Prediction days: {n_days}")
    
    # Use the unified inverse transform function (modified for weather-enhanced data)
    all_predictions_original, all_targets_original = inverse_transform_predictions(
        all_predictions, all_targets, scaler_data, encode_map
    )
    
    print(f"\nAfter Inverse Transform:")
    print(f"Predictions range: [{all_predictions_original.min():.2f}, {all_predictions_original.max():.2f}]")
    print(f"Targets range: [{all_targets_original.min():.2f}, {all_targets_original.max():.2f}]")
    
    # Calculate R2 scores
    overall_r2_scores = []
    for i in range(n_days):
        day_targets = all_targets_original[:, :, i].reshape(-1)
        day_predictions = all_predictions_original[:, :, i].reshape(-1)
        r2_val = r2_score(day_targets, day_predictions)
        overall_r2_scores.append(r2_val)
        print(f'\t --- Day {i+1} Overall R2 Score: {r2_val:.4f}')
    
    overall_r2 = r2_score(all_targets_original.reshape(-1), all_predictions_original.reshape(-1))
    print(f'\t --- Overall R2 Score (all days, all nodes): {overall_r2:.4f}')
    
    print(f"\nIndividual Reservoir R2 Scores:")
    reservoir_r2_scores = {}
    for node_idx in range(n_nodes):
        node_predictions = all_predictions_original[:, node_idx, :]  # Shape: (samples, 7)
        node_targets = all_targets_original[:, node_idx, :]         # Shape: (samples, 7)
        node_r2 = r2_score(node_targets.reshape(-1), node_predictions.reshape(-1))
        reservoir_name = f"Node_{node_idx}"
        if node_idx in idx_to_reservoir:
            reservoir_name = idx_to_reservoir[node_idx]
        reservoir_r2_scores[reservoir_name] = node_r2
        print(f'\t --- {reservoir_name}: {node_r2:.4f}')
        daily_r2_scores = []
        for day in range(n_days):
            day_r2 = r2_score(node_targets[:, day], node_predictions[:, day])
            daily_r2_scores.append(day_r2)
            print(f'\t\t --- Day {day+1}: {day_r2:.4f}')
    
    return overall_r2, overall_r2_scores, reservoir_r2_scores


if __name__ == "__main__":
    # Set random seed for reproducibility
    SEED = 42
    seed_everything(SEED)
    
    # ===== CONFIGURATION PARAMETERS =====
    SCALER_TYPE = "local"  # ["global", "local"] - scaler type
    load_checkpoint = True  # [True, False]
    checkpoint_time = "202508061835"  # (if load_checkpoint=True)
    USE_SUBSAMPLE = False  # [True, False]
    SUBSAMPLE_SIZE = 100  # (if USE_SUBSAMPLE=True)
    
    print(f"Using scaler type: {SCALER_TYPE}")
    print(f"Load checkpoint: {load_checkpoint}")
    print(f"Use subsample: {USE_SUBSAMPLE} (size: {SUBSAMPLE_SIZE if USE_SUBSAMPLE else 'Full'})")
    
    if not load_checkpoint:
        print("Processing weather-enhanced reservoir data...")
    
    data_path = "./data"
    parsed_path = os.path.join(data_path, 'parsed')
    os.makedirs(parsed_path, exist_ok=True)
    
    # Process or load weather-enhanced data
    weather_enhanced_file = os.path.join(parsed_path, f"weather_enhanced_data_{SCALER_TYPE}.pkl")
    
    if not load_checkpoint and not os.path.exists(weather_enhanced_file):
        # Process weather-enhanced data
        weather_enhanced_data = preprocess_reservoir_data_with_weather(
            data_path=data_path, output_path=parsed_path, scaler_type=SCALER_TYPE)
    else:
        # Load existing weather-enhanced data
        print(f"Loading existing weather-enhanced data from {weather_enhanced_file}")
        with open(weather_enhanced_file, 'rb') as f:
            weather_enhanced_data = pickle.load(f)
    
    # Prepare graph data
    graph_cfg = "k2_config3"
    graph_path = os.path.join(data_path, 'graph')
    graph_file = os.path.join(graph_path, f"{graph_cfg}.pkl")
    
    with open(graph_file, 'rb') as f:
        graph_data = pickle.load(f)
    
    edge_index = graph_data['edge_index']
    encode_map = graph_data['encode_map']
    
    # Prepare supervised data
    X_train, y_train, weather_train = prepare_weather_enhanced_data(weather_enhanced_data, encode_map, "train")
    X_test, y_test, weather_test = prepare_weather_enhanced_data(weather_enhanced_data, encode_map, "test")
    
    # Apply subsampling for testing if enabled
    if USE_SUBSAMPLE:
        X_train = X_train[:SUBSAMPLE_SIZE]
        y_train = y_train[:SUBSAMPLE_SIZE]
        weather_train = weather_train[:SUBSAMPLE_SIZE]
        X_test = X_test[:min(SUBSAMPLE_SIZE//5, X_test.shape[0])]
        y_test = y_test[:min(SUBSAMPLE_SIZE//5, y_test.shape[0])]
        weather_test = weather_test[:min(SUBSAMPLE_SIZE//5, weather_test.shape[0])]
        
        print("Applied subsampling for testing:")
    
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"weather_train: {weather_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    print(f"weather_test: {weather_test.shape}")
    
    # Create datasets and data loaders
    train_dataset = WeatherEnhancedDataset(X_train, y_train, weather_train, edge_index)
    test_dataset = WeatherEnhancedDataset(X_test, y_test, weather_test, edge_index)
    
    # Custom collate function for PyTorch Geometric Data objects
    def collate_fn(batch):
        return list(zip(*batch))
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    
    # Initialize model
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = ReservoirAttentionNet_withFuture(
        in_dim=X_train.shape[-1], 
        hid_dim=128,
        gnn_dim=64, 
        lstm_dim=64, 
        pred_days=7,
        weather_features=2
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    criterion = nn.MSELoss()
    
    print(f"Using device: {device}")
    print(f"Model Summary:")
    print(f"Input dimension: {X_train.shape[-1]}")
    print(f"Weather features: 2")
    print(f"Prediction days: 7")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize training logger
    model_name = model.__class__.__name__
    logger = TrainingLogger(model_name + "_weather", SCALER_TYPE, use_pretrain=False)
    
    # Handle checkpoint loading
    if load_checkpoint:
        adjusted_time = _adjust_checkpoint_time(model_name + "_weather", SCALER_TYPE, checkpoint_time)
        if adjusted_time is None:
            print(f"ERROR: No checkpoint files found for model '{model_name}_weather' with scaler type '{SCALER_TYPE}'")
            exit(1)
        
        checkpoint_path = os.path.join("logs", model_name + "_weather", SCALER_TYPE, f"checkpoint_{adjusted_time}.pth")
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint model from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            stored_val_loss = checkpoint['loss']
            print(f"Loaded model from epoch {checkpoint['epoch']} with validation loss: {stored_val_loss:.6f}")

            print(f"Recalculating validation loss for verification...")
            current_val_loss = run_epoch_with_weather(model, test_loader, criterion, None, device, train=False)
            print(f"Current validation loss: {current_val_loss:.6f} (stored: {stored_val_loss:.6f}, diff: {abs(current_val_loss - stored_val_loss):.6f})")

            print(f"Evaluating loaded checkpoint model...")
            
            # Keep scaler data for evaluation
            scaler_data = {
                'scaler_X': weather_enhanced_data.get('scaler_X'),
                'scaler_y': weather_enhanced_data.get('scaler_y'),
                'local_scalers_X': weather_enhanced_data.get('local_scalers_X'),
                'local_scalers_y': weather_enhanced_data.get('local_scalers_y'),
                'params': weather_enhanced_data.get('params', {})
            }
            
            overall_r2, daily_r2_scores, reservoir_r2_scores = evaluate_weather_enhanced_model(
                model, test_loader, encode_map, scaler_data, device)
            print(f"Evaluation completed.")
            exit()
        else:
            print(f"ERROR: Checkpoint model not found at: {checkpoint_path}")
            print(f"Available checkpoints in logs/{model_name}_weather/{SCALER_TYPE}/:")
            checkpoint_dir = os.path.join("logs", model_name + "_weather", SCALER_TYPE)
            if os.path.exists(checkpoint_dir):
                for file in os.listdir(checkpoint_dir):
                    if file.endswith('.pth'):
                        print(f"  - {file}")
            else:
                print(f"  Directory does not exist: {checkpoint_dir}")
            exit(1)
    
    # Training loop
    print("Training weather-enhanced model...")
    print(f"Training for {3 if USE_SUBSAMPLE else 10} epochs...")
    
    num_epochs = 3 if USE_SUBSAMPLE else 10
    for epoch in range(1, num_epochs + 1):
        train_loss = run_epoch_with_weather(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch_with_weather(model, test_loader, criterion, None, device, train=False)
        
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Log epoch and save checkpoint if best
        is_best = logger.log_epoch(epoch, train_loss, val_loss, current_lr)
        if is_best:
            logger.save_checkpoint(model, optimizer, epoch, val_loss)
    
    print(f"\nEvaluating weather-enhanced model...")
    
    # Keep scaler data for evaluation
    scaler_data = {
        'scaler_X': weather_enhanced_data.get('scaler_X'),
        'scaler_y': weather_enhanced_data.get('scaler_y'),
        'local_scalers_X': weather_enhanced_data.get('local_scalers_X'),
        'local_scalers_y': weather_enhanced_data.get('local_scalers_y'),
        'params': weather_enhanced_data.get('params', {})
    }
    
    overall_r2, daily_r2_scores, reservoir_r2_scores = evaluate_weather_enhanced_model(
        model, test_loader, encode_map, scaler_data, device)
    
    # Save final results
    evaluation_info = f"\nFinal Weather-Enhanced Model Evaluation Results:\n"
    evaluation_info += f"Overall R2 Score: {overall_r2:.4f}\n"
    evaluation_info += f"Daily R2 Scores: {[f'{r:.4f}' for r in daily_r2_scores]}\n"
    evaluation_info += f"Model: {model_name}_weather\n"
    evaluation_info += f"Final Best Validation Loss: {logger.best_val_loss:.6f}\n"
    evaluation_info += f"Weather Features: Temperature + Precipitation\n"
    logger.save_results(evaluation_info)
    
    print("Weather-enhanced model training and evaluation completed!")
