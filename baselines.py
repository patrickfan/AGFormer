import os
import pickle
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm
import torch
from utils import WindowDataset, inverse_transform_predictions, TrainingLogger, seed_everything, _adjust_checkpoint_time
from torch.utils.data import DataLoader
from models.gnn import ReservoirNet, ReservoirNetSeq2Seq, ReservoirAttentionNet
from models.lstm import Seq2SeqLSTM, Seq2SeqLSTM_new, Transformer
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler


def run_epoch(loader, train=True):
    model.train() if train else model.eval()
    total_loss = 0
    phase = "Train" if train else "Val"
    pbar = tqdm(loader, desc=f"{phase} Epoch", leave=False)
    
    for graph_batch, target_batch in pbar:  # loader
        preds, targets = [], []
        for graphs, tgt in zip(graph_batch, target_batch):   # Iterate samples in batch
            graphs = [g.to(device) for g in graphs]
            preds.append(model(graphs))
            targets.append(tgt.to(device))
        preds = torch.stack(preds) # [B, nodes, 7]
        targets = torch.stack(targets) # [B, nodes, 7]
        loss = criterion(preds, targets)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * preds.size(0)
        
        # Update progress bar with current loss
        current_avg_loss = total_loss / (pbar.n + 1) / loader.batch_size
        pbar.set_postfix({"Loss": f"{current_avg_loss:.6f}"})
        
    return total_loss / len(loader.dataset)


def evaluate_model(model, test_loader, encode_map, scaler_data):
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for graph_batch, target_batch in test_loader:
            preds, targets = [], []
            for graphs, tgt in zip(graph_batch, target_batch):
                graphs = [g.to(device) for g in graphs]
                preds.append(model(graphs))
                targets.append(tgt.to(device))
            preds = torch.stack(preds)  # [B, nodes, 7]
            targets = torch.stack(targets)  # [B, nodes, 7]
            all_predictions.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate all batches
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
    
    print(f"\nEvaluation Results:")
    print(f"Total samples: {n_samples}")
    print(f"Total nodes (reservoirs): {n_nodes}")
    print(f"Prediction days: {n_days}")
    
    # Debug: Check scaler and data ranges
    print(f"\nScaler Info:")
    scaler_type = scaler_data.get('params', {}).get('scaler_type', 'global')
    print(f"Scaler type: {scaler_type}")
    
    if scaler_type == 'global':
        scaler_y = scaler_data['scaler_y']
        print(f"Global scaler data_min_: {scaler_y.data_min_}")
        print(f"Global scaler data_max_: {scaler_y.data_max_}")
        print(f"Global scaler feature_range: {scaler_y.feature_range}")
    else:
        print(f"Local scalers: {len(scaler_data['local_scalers_y'])} reservoir-specific scalers")
    
    print(f"\nBefore Inverse Transform:")
    print(f"Predictions range: [{all_predictions.min():.6f}, {all_predictions.max():.6f}]")
    print(f"Targets range: [{all_targets.min():.6f}, {all_targets.max():.6f}]")
    
    # Check if predictions are within scaler range
    if all_predictions.min() < 0 or all_predictions.max() > 1:
        print(f"WARNING: Predictions outside [0,1] range!")
    
    # ===== R2 on SCALED DATA (before inverse transform) =====
    print(f"\n" + "="*50)
    print(f"R2 CALCULATION ON SCALED DATA (BEFORE INVERSE TRANSFORM)")
    print(f"="*50)
    
    # Overall R2 on scaled data
    scaled_overall_r2 = r2_score(all_targets.reshape(-1), all_predictions.reshape(-1))
    print(f"Overall R2 on scaled data: {scaled_overall_r2:.4f}")
    
    # Daily R2 on scaled data
    scaled_daily_r2_scores = []
    for i in range(n_days):
        day_targets = all_targets[:, :, i].reshape(-1)
        day_predictions = all_predictions[:, :, i].reshape(-1)
        day_r2 = r2_score(day_targets, day_predictions)
        scaled_daily_r2_scores.append(day_r2)
        print(f'Day {i+1} R2 on scaled data: {day_r2:.4f}')
    
    # Individual reservoir R2 on scaled data
    print(f"\nIndividual Reservoir R2 on Scaled Data:")
    scaled_reservoir_r2_scores = {}
    
    for node_idx in range(n_nodes):
        node_predictions = all_predictions[:, node_idx, :]  # Shape: (samples, 7)
        node_targets = all_targets[:, node_idx, :]         # Shape: (samples, 7)
        
        node_r2 = r2_score(node_targets.reshape(-1), node_predictions.reshape(-1))
        
        reservoir_name = f"Node_{node_idx}"
        if node_idx in idx_to_reservoir:
            reservoir_name = idx_to_reservoir[node_idx]
        
        scaled_reservoir_r2_scores[reservoir_name] = node_r2
        print(f'{reservoir_name}: {node_r2:.4f}')
    
    # ===== R2 on ORIGINAL DATA (after inverse transform) =====
    print(f"\n" + "="*50)
    print(f"R2 CALCULATION ON ORIGINAL DATA (AFTER INVERSE TRANSFORM)")
    print(f"="*50)
    
    # Use the unified inverse transform function
    all_predictions_original, all_targets_original = inverse_transform_predictions(
        all_predictions, all_targets, scaler_data, encode_map
    )
    
    print(f"\nAfter Inverse Transform:")
    print(f"Predictions range: [{all_predictions_original.min():.2f}, {all_predictions_original.max():.2f}]")
    print(f"Targets range: [{all_targets_original.min():.2f}, {all_targets_original.max():.2f}]")
    
    # Calculate overall R2 scores for each day
    overall_r2_scores = []
    for i in range(n_days):
        # Calculate R2 for each day's predictions across all nodes and samples
        day_targets = all_targets_original[:, :, i].reshape(-1)
        day_predictions = all_predictions_original[:, :, i].reshape(-1)
        
        r2_val = r2_score(day_targets, day_predictions)
        overall_r2_scores.append(r2_val)
        print(f'\t --- Day {i+1} Overall R2 Score: {r2_val:.4f}')
    
    # Calculate overall R2 score across all days and nodes
    overall_r2 = r2_score(all_targets_original.reshape(-1), all_predictions_original.reshape(-1))
    print(f'\t --- Overall R2 Score (all days, all nodes): {overall_r2:.4f}')
    
    # Calculate R2 scores for each reservoir (node) with detailed analysis
    print(f"\nIndividual Reservoir R2 Scores:")
    reservoir_r2_scores = {}
    
    # Analyze the worst performing reservoirs
    worst_reservoirs = []
    
    for node_idx in range(n_nodes):
        node_predictions = all_predictions_original[:, node_idx, :]  # Shape: (samples, 7)
        node_targets = all_targets_original[:, node_idx, :]         # Shape: (samples, 7)
        
        # Calculate R2 for this reservoir across all days
        node_r2 = r2_score(node_targets.reshape(-1), node_predictions.reshape(-1))
        
        # Get reservoir name from encode_map if available
        reservoir_name = f"Node_{node_idx}"
        if node_idx in idx_to_reservoir:
            reservoir_name = idx_to_reservoir[node_idx]
        
        reservoir_r2_scores[reservoir_name] = node_r2
        
        # Store worst performing reservoirs for detailed analysis
        if node_r2 < 0:
            worst_reservoirs.append((reservoir_name, node_r2, node_idx))
        
        print(f'\t --- {reservoir_name}: {node_r2:.4f}')
        
        # Calculate daily R2 scores for this reservoir
        daily_r2_scores = []
        for day in range(n_days):
            day_r2 = r2_score(node_targets[:, day], node_predictions[:, day])
            daily_r2_scores.append(day_r2)
            print(f'\t\t --- Day {day+1}: {day_r2:.4f}')
    
    # ===== COMPARISON SUMMARY =====
    print(f"\n" + "="*50)
    print(f"COMPARISON SUMMARY")
    print(f"="*50)
    print(f"Overall R2 - Scaled data: {scaled_overall_r2:.4f}")
    print(f"Overall R2 - Original data: {overall_r2:.4f}")
    print(f"Difference: {overall_r2 - scaled_overall_r2:.4f}")
    
    # Count negative R2 reservoirs
    scaled_negative_count = sum(1 for r2 in scaled_reservoir_r2_scores.values() if r2 < 0)
    original_negative_count = sum(1 for r2 in reservoir_r2_scores.values() if r2 < 0)
    print(f"Negative R2 reservoirs - Scaled: {scaled_negative_count}, Original: {original_negative_count}")
    
    # Detailed analysis of worst performing reservoirs
    if worst_reservoirs:
        print(f"\nDetailed Analysis of Worst Performing Reservoirs:")
        worst_reservoirs.sort(key=lambda x: x[1])  # Sort by R2 (worst first)
        
        for reservoir_name, r2_score_val, node_idx in worst_reservoirs[:3]:  # Top 3 worst
            node_predictions = all_predictions_original[:, node_idx, :]
            node_targets = all_targets_original[:, node_idx, :]
            
            print(f"\n{reservoir_name} (R2: {r2_score_val:.4f}):")
            print(f"  Target range: [{node_targets.min():.2f}, {node_targets.max():.2f}]")
            print(f"  Pred range: [{node_predictions.min():.2f}, {node_predictions.max():.2f}]")
            print(f"  Target mean: {node_targets.mean():.2f}, std: {node_targets.std():.2f}")
            print(f"  Pred mean: {node_predictions.mean():.2f}, std: {node_predictions.std():.2f}")
            
            # Check if predictions are reasonable
            if node_predictions.min() < 0:
                print(f"  WARNING: Negative predictions detected!")
            if node_predictions.max() > node_targets.max() * 10:
                print(f"  WARNING: Predictions much larger than targets!")
    
    return overall_r2, overall_r2_scores, reservoir_r2_scores


if __name__ == "__main__":
    # Set random seed for reproducibility
    SEED = 42
    seed_everything(SEED)

    SCALER_TYPE = "local"  # ['global', 'local']
    load_checkpoint = False  # [True, False]
    pretrain_time = "202507062121"  # Loading checkpoint time
    print(f"Using scaler type: {SCALER_TYPE}")
    print(f"Load checkpoint: {load_checkpoint}")

    data_path = "./data"  # ./graph/data - depends on root directory
    parsed_path = os.path.join(data_path, 'parsed')
    os.makedirs(parsed_path, exist_ok=True)
    graph_path = os.path.join(data_path, 'graph')

    # Load the original data with scalers for the specified scaler type
    all_rsr_data_file = os.path.join(parsed_path, f"all_rsr_data_{SCALER_TYPE}.pkl")
    if not os.path.exists(all_rsr_data_file):
        raise FileNotFoundError(f"Preprocessed data not found. Please run _preprocess.py first to generate {all_rsr_data_file}")
    
    with open(all_rsr_data_file, 'rb') as f:
        all_rsr_data = pickle.load(f)
    
    # Keep the scaler data for evaluation
    scaler_data = {
        'scaler_X': all_rsr_data.get('scaler_X'),
        'scaler_y': all_rsr_data.get('scaler_y'),
        'local_scalers_X': all_rsr_data.get('local_scalers_X'),
        'local_scalers_y': all_rsr_data.get('local_scalers_y'),
        'params': all_rsr_data.get('params', {})
    }
    del all_rsr_data
    
    # Prepare input data for GNN
    graph_cfg = "k15_config3" # _k_n_nearest[config1, config2, config3]
    graph_path = os.path.join(data_path, 'graph')
    graph_file = os.path.join(graph_path, "%s.pkl" % graph_cfg)
    print(f"Loading graph data from: {graph_file}")
    
    # Load Graph Data
    with open(graph_file, 'rb') as f:
        graph_data = pickle.load(f)
    # Retrieve Parsed Info
    A = graph_data['A']
    edge_index = graph_data['edge_index']
    encode_map = graph_data['encode_map']
    # print(f"Encode map: \n{encode_map}")

    supervised_data = torch.load(os.path.join(parsed_path, f"_GNN_supervise_{SCALER_TYPE}.pt"), weights_only=True)
    X_train, y_train = supervised_data['X_train'], supervised_data['y_train']
    X_test, y_test = supervised_data['X_test'], supervised_data['y_test']
    
    print(f"Loaded data shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    print(f"edge_index: {edge_index.shape}")
    
    train_loader = DataLoader(WindowDataset(X_train, y_train, edge_index),
                          batch_size=4, shuffle=True, collate_fn=lambda b: list(zip(*b)))
    test_loader  = DataLoader(WindowDataset(X_test,  y_test, edge_index),
                            batch_size=4, shuffle=False, collate_fn=lambda b: list(zip(*b)))
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    
    # model = Transformer(input_dim=X_train.shape[-1], hidden_dim=32, dropout=0.).to(device)
    # model = ReservoirAttentionNet(in_dim=X_train.shape[-1], hid_dim=128,
    #                               gnn_dim=64, lstm_dim=64, pred_days=7).to(device)
    model = ReservoirNetSeq2Seq(in_dim=X_train.shape[-1], hid_dim=128,
                                gnn_dim=128, lstm_dim=64, pred_days=7).to(device)
    # model = ReservoirNet(in_dim=X_train.shape[-1], hid_dim=64,
    #                     gnn_dim=64, lstm_dim=128, pred_days=7).to(device)
    # model = Seq2SeqLSTM(input_dim=X_train.shape[-1], hidden_dim=32,  # 128 64
    #                     output_dim=1, num_layers=1, dropout=0.4).to(device)  # , dropout=0.5
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    criterion = nn.MSELoss()
    
    print(f"Using device: {device}")
    print(f"Model Summary:")
    print(f"Input dimension: {X_train.shape[-1]}")
    print(f"Prediction days: 7")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(model)

    # Initialize training logger
    model_name = model.__class__.__name__
    logger = TrainingLogger(model_name, SCALER_TYPE)

    if load_checkpoint:
        adjusted_time = _adjust_checkpoint_time(model_name, SCALER_TYPE, pretrain_time)
        if adjusted_time is None:
            print(f"ERROR: No checkpoint files found for model '{model_name}' with scaler type '{SCALER_TYPE}'")
            exit(1)
        
        checkpoint_path = os.path.join("logs", model_name, SCALER_TYPE, f"checkpoint_{adjusted_time}.pth")
        if os.path.exists(checkpoint_path):
            print(f"Loading pretrained model from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            stored_val_loss = checkpoint['loss']
            print(f"Loaded model from epoch {checkpoint['epoch']} with validation loss: {stored_val_loss:.6f}")

            print(f"Recalculating validation loss for verification...")
            current_val_loss = run_epoch(test_loader, train=False)
            print(f"Current validation loss: {current_val_loss:.6f} (stored: {stored_val_loss:.6f}, diff: {abs(current_val_loss - stored_val_loss):.6f})")

            print(f"Evaluating loaded pretrained model...")
            overall_r2, daily_r2_scores, reservoir_r2_scores = evaluate_model(model, test_loader, encode_map, scaler_data)
            print(f"Evaluation completed.")
            exit()
        else:
            print(f"ERROR: Pretrained model not found at: {checkpoint_path}")
            print(f"Available checkpoints in logs/{model_name}/{SCALER_TYPE}/:")
            checkpoint_dir = os.path.join("logs", model_name, SCALER_TYPE)
            if os.path.exists(checkpoint_dir):
                for file in os.listdir(checkpoint_dir):
                    if file.endswith('.pth'):
                        print(f"  - {file}")
            else:
                print(f"  Directory does not exist: {checkpoint_dir}")
            exit(1)
    
    print(f"Training model...")
    for epoch in range(1, 11):
        train_loss = run_epoch(train_loader, train=True)
        val_loss   = run_epoch(test_loader , train=False)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Log epoch and save checkpoint if best
        is_best = logger.log_epoch(epoch, train_loss, val_loss, current_lr)
        if is_best:
            logger.save_checkpoint(model, optimizer, epoch, val_loss)

    print(f"\nEvaluating model...")
    overall_r2, daily_r2_scores, reservoir_r2_scores = evaluate_model(model, test_loader, encode_map, scaler_data)

    evaluation_info = f"\nFinal Evaluation Results:\n"
    evaluation_info += f"Overall R2 Score: {overall_r2:.4f}\n"
    evaluation_info += f"Daily R2 Scores: {[f'{r:.4f}' for r in daily_r2_scores]}\n"
    evaluation_info += f"Model: {model_name}\n"
    evaluation_info += f"Final Best Validation Loss: {logger.best_val_loss:.6f}\n"
    logger.save_results(evaluation_info)
