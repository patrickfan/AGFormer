import os
import pickle
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm
import torch
from utils import WindowDataset, DynamicWindowDataset, inverse_transform_predictions, TrainingLogger, seed_everything, _adjust_checkpoint_time
from torch.utils.data import DataLoader
from models.gnn import ReservoirAttentionNet
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler


def aggregate_attention_weights(timestep_attention_weights):
    """Aggregate attention weights collected during training.
    Args:
        timestep_attention_weights: List of lists, each containing attention weights for one time step
    Returns:
        List of averaged attention weights for each time step (30 time steps)
    """
    avg_timestep_attention = []
    for t in range(30):
        if timestep_attention_weights[t]:
            # Average attention weights across all batches for this time step
            avg_att = torch.stack(timestep_attention_weights[t]).mean(dim=0)
            avg_timestep_attention.append(avg_att)
        else:
            avg_timestep_attention.append(None)
    
    return avg_timestep_attention


def get_gat_edge_index_with_self_loops(original_edge_index, num_nodes):
    """Get the edge_index that GAT would use (with self-loops added).
    Args:
        original_edge_index: Original edge index tensor [2, num_edges]
        num_nodes: Number of nodes in the graph
    Returns:
        Edge index with self-loops added [2, num_edges + num_nodes]
    """
    device = original_edge_index.device
    self_loops = torch.arange(num_nodes, device=device)
    self_loop_edge_index = torch.stack([self_loops, self_loops], dim=0)
    edge_index_with_self_loops = torch.cat([original_edge_index, self_loop_edge_index], dim=1)
    
    return edge_index_with_self_loops


def refine_edge_index_by_attention(base_edge_indices, timestep_attention_weights, threshold=0.1, num_nodes=None):
    """Refine edge_index by removing edges with attention weights below threshold for each time step.
    IMPORTANT: Self-loops are ALWAYS preserved regardless of attention weights.
    Args:
        base_edge_indices: List of base edge index tensors for each timestep OR single edge_index for all timesteps
        timestep_attention_weights: List of attention weights for each time step
        threshold: Threshold for edge removal
        num_nodes: Number of nodes (needed to reconstruct GAT edge_index)
    Returns:
        List of refined edge_index for each time step
    """
    refined_edge_indices = []

    if isinstance(base_edge_indices, list):
        base_edge_list = base_edge_indices
    else:
        base_edge_list = [base_edge_indices] * len(timestep_attention_weights)

    for t, attention_weights in enumerate(timestep_attention_weights):
        if attention_weights is None:
            refined_edge_indices.append(base_edge_list[t])
        else:
            base_edge_index = base_edge_list[t]
            if num_nodes is None:
                num_nodes = base_edge_index.max().item() + 1

            is_self_loop = base_edge_index[0] == base_edge_index[1]
            self_loop_edges = base_edge_index[:, is_self_loop]
            regular_edges = base_edge_index[:, ~is_self_loop]
            expected_edges = regular_edges.shape[1] + num_nodes

            if expected_edges != attention_weights.shape[0]:
                print(f"Warning: Shape mismatch at timestep {t} - Expected edges: {expected_edges}, attention weights: {attention_weights.shape[0]}")
                print(f"  Regular edges: {regular_edges.shape[1]}, Self-loops in base: {self_loop_edges.shape[1]}, Nodes: {num_nodes}")
                # Truncate attention weights to available size
                attention_weights = attention_weights[:min(expected_edges, attention_weights.shape[0])]
            
            attention_weights = attention_weights.to(base_edge_index.device)
            n_regular = regular_edges.shape[1]
            if attention_weights.shape[0] >= n_regular + num_nodes:
                regular_att_weights = attention_weights[:n_regular]
                self_loop_att_weights = attention_weights[n_regular:n_regular + num_nodes]
            else:
                regular_att_weights = attention_weights[:min(n_regular, attention_weights.shape[0])]
                if regular_att_weights.shape[0] < n_regular:
                    padding = torch.zeros(n_regular - regular_att_weights.shape[0], device=attention_weights.device)
                    regular_att_weights = torch.cat([regular_att_weights, padding])

            regular_mask = regular_att_weights >= threshold
            refined_regular_edges = regular_edges[:, regular_mask]
            
            if self_loop_edges.shape[1] == 0:
                self_loops = torch.arange(num_nodes, device=base_edge_index.device)
                self_loop_edges = torch.stack([self_loops, self_loops], dim=0)
            refined_edge_index = torch.cat([refined_regular_edges, self_loop_edges], dim=1)
            refined_edge_indices.append(refined_edge_index)
            # print(f"Time step {t}: kept {refined_regular_edges.shape[1]}/{regular_edges.shape[1]} regular edges + {self_loop_edges.shape[1]} self-loops (threshold: {threshold})")
    
    return refined_edge_indices


def load_pretrain_weights(model, scaler_type="global"):
    """Load pretrained encoder weights into ReservoirAttentionNet"""
    try:
        pretrain_dir = f"./logs/PretrainModel/{scaler_type}"
        if not os.path.exists(pretrain_dir):
            print(f"Warning: Pretrain directory not found: {pretrain_dir}")
            return False

        pretrain_path = os.path.join(pretrain_dir, "pretrain_best_model.pth")
        if not os.path.exists(pretrain_path):
            print(f"Warning: Pretrain model not found: {pretrain_path}")
            return False

        checkpoint = torch.load(pretrain_path, map_location=model.encoder_mlp[0].weight.device, weights_only=False)
        pretrain_state_dict = checkpoint['model_state_dict']

        encoder_state_dict = {}
        for key, value in pretrain_state_dict.items():
            if key.startswith('window_embedding.encoder.'):
                new_key = key.replace('window_embedding.encoder.', '')
                encoder_state_dict[new_key] = value
        
        if not encoder_state_dict:
            print("Warning: No encoder weights found in pretrained model")
            return False

        model.encoder_mlp.load_state_dict(encoder_state_dict, strict=True)

        print(f"Successfully loaded pretrained encoder weights from: {pretrain_path}")
        print(f"Loaded {len(encoder_state_dict)} parameter tensors")
        return True

    except Exception as e:
        print(f"Error loading pretrained weights: {e}")
        return False


def run_epoch(loader, train=True, collect_attention=False):
    model.train() if train else model.eval()
    total_loss = 0
    phase = "Train" if train else "Val"
    pbar = tqdm(loader, desc=f"{phase} Epoch", leave=False)

    timestep_attention_weights = [[] for _ in range(30)] if collect_attention else None
    for graph_batch, target_batch in pbar:
        preds, targets = [], []
        for graphs, tgt in zip(graph_batch, target_batch):
            graphs = [g.to(device) for g in graphs]

            if collect_attention and train and isinstance(model, ReservoirAttentionNet):
                pred, att_scores = model(graphs, return_gat_att=True)
                preds.append(pred)

                for t, att_tuple in enumerate(att_scores):
                    _, alpha2 = att_tuple
                    attention_weights = alpha2.mean(dim=1)  # Average over attention heads -> [num_edges]
                    timestep_attention_weights[t].append(attention_weights)
            else:
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

        current_avg_loss = total_loss / (pbar.n + 1) / loader.batch_size
        pbar.set_postfix({"Loss": f"{current_avg_loss:.6f}"})

    if collect_attention and timestep_attention_weights:
        return total_loss / len(loader.dataset), timestep_attention_weights
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

    if all_predictions.min() < 0 or all_predictions.max() > 1:
        print(f"WARNING: Predictions outside [0,1] range!")
    
    # ===== R2 on SCALED DATA (before inverse transform) =====
    print(f"\n" + "="*50)
    print(f"R2 CALCULATION ON SCALED DATA (BEFORE INVERSE TRANSFORM)")
    print(f"="*50)

    scaled_overall_r2 = r2_score(all_targets.reshape(-1), all_predictions.reshape(-1))
    print(f"Overall R2 on scaled data: {scaled_overall_r2:.4f}")

    scaled_daily_r2_scores = []
    for i in range(n_days):
        day_targets = all_targets[:, :, i].reshape(-1)
        day_predictions = all_predictions[:, :, i].reshape(-1)
        day_r2 = r2_score(day_targets, day_predictions)
        scaled_daily_r2_scores.append(day_r2)
        print(f'Day {i+1} R2 on scaled data: {day_r2:.4f}')

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
    worst_reservoirs = []
    for node_idx in range(n_nodes):
        node_predictions = all_predictions_original[:, node_idx, :]  # Shape: (samples, 7)
        node_targets = all_targets_original[:, node_idx, :]         # Shape: (samples, 7)
        node_r2 = r2_score(node_targets.reshape(-1), node_predictions.reshape(-1))
        reservoir_name = f"Node_{node_idx}"
        if node_idx in idx_to_reservoir:
            reservoir_name = idx_to_reservoir[node_idx]
        reservoir_r2_scores[reservoir_name] = node_r2
        if node_r2 < 0:
            worst_reservoirs.append((reservoir_name, node_r2, node_idx))
        print(f'\t --- {reservoir_name}: {node_r2:.4f}')
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
    
    SCALER_TYPE = "local"  # ["global", "local"]
    load_checkpoint = False  # [True, False]
    checkpoint_time = "202507061331"
    load_pretrain = True  # [True, False] - Load pretrained encoder weights
    
    print(f"Using scaler type: {SCALER_TYPE}")
    print(f"Load checkpoint: {load_checkpoint}")
    print(f"Load pretrain: {load_pretrain}")

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
    graph_cfg = "config3" # [config1, config2, config3]
    graph_path = os.path.join(data_path, 'graph')
    graph_file = os.path.join(graph_path, "%s.pkl" % graph_cfg)
    # Load Graph Data
    with open(graph_file, 'rb') as f:
        graph_data = pickle.load(f)
    # Retrieve Parsed Info
    A = graph_data['A']
    edge_index = graph_data['edge_index']
    encode_map = graph_data['encode_map']

    supervised_data = torch.load(os.path.join(parsed_path, f"_GNN_supervise_{SCALER_TYPE}.pt"), weights_only=True)
    X_train, y_train = supervised_data['X_train'], supervised_data['y_train']
    X_test, y_test = supervised_data['X_test'], supervised_data['y_test']

    # test_samples = 100
    # X_train = X_train[:test_samples]
    # y_train = y_train[:test_samples]
    # X_test = X_test[:min(test_samples//5, X_test.shape[0])]
    # y_test = y_test[:min(test_samples//5, y_test.shape[0])]

    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}")
    print(f"y_test: {y_test.shape}")
    print(f"edge_index: {edge_index.shape}")

    # Initialize with original edge_index
    train_dataset = WindowDataset(X_train, y_train, edge_index)
    test_dataset = WindowDataset(X_test, y_test, edge_index)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda b: list(zip(*b)))
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=lambda b: list(zip(*b)))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    model = ReservoirAttentionNet(in_dim=X_train.shape[-1], hid_dim=128,
                                  gnn_dim=64, lstm_dim=64, pred_days=7).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    criterion = nn.MSELoss()

    print(f"Using device: {device}")
    print(f"Model Summary:")
    print(f"Input dimension: {X_train.shape[-1]}")
    print(f"Prediction days: 7")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    print(model)

    # Iterative graph refinement parameters
    iter_graph = True  # [True, False]
    gat_threshold = 0.3
    refine_frequency = 4

    # Load pretrained weights if requested and not loading checkpoint
    pretrain_loaded = False
    if load_pretrain and not load_checkpoint:
        pretrain_loaded = load_pretrain_weights(model, SCALER_TYPE)
    
    # Initialize training logger
    model_name = model.__class__.__name__
    logger = TrainingLogger(model_name, SCALER_TYPE, use_pretrain=pretrain_loaded)

    if load_checkpoint:
        adjusted_time = _adjust_checkpoint_time(model_name, SCALER_TYPE, checkpoint_time)
        if adjusted_time is None:
            print(f"ERROR: No checkpoint files found for model '{model_name}' with scaler type '{SCALER_TYPE}'")
            exit(1)
        
        checkpoint_path = os.path.join("logs", model_name, SCALER_TYPE, f"checkpoint_{adjusted_time}.pth")
        if os.path.exists(checkpoint_path):
            print(f"Loading checkpoint model from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            stored_val_loss = checkpoint['loss']
            print(f"Loaded model from epoch {checkpoint['epoch']} with validation loss: {stored_val_loss:.6f}")

            print(f"Recalculating validation loss for verification...")
            current_val_loss = run_epoch(test_loader, train=False)
            print(f"Current validation loss: {current_val_loss:.6f} (stored: {stored_val_loss:.6f}, diff: {abs(current_val_loss - stored_val_loss):.6f})")

            print(f"Evaluating loaded checkpointmodel...")
            overall_r2, daily_r2_scores, reservoir_r2_scores = evaluate_model(model, test_loader, encode_map, scaler_data)
            print(f"Evaluation completed.")
            exit()
        else:
            print(f"ERROR: Loaded checkpoint model not found at: {checkpoint_path}")
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
    print(f"Iterative graph refinement: {'Enabled' if iter_graph else 'Disabled'}")
    if iter_graph:
        print(f"GAT attention threshold: {gat_threshold}")
        print(f"Graph refinement frequency: every {refine_frequency} epochs")

    current_edge_indices = None
    original_edge_count = edge_index.shape[1]

    if not load_checkpoint:
        edge_tracking_data = {
            'epoch_edge_changes': [],
            'original_edge_index': edge_index.clone(),
            'original_edge_count': original_edge_count,
            'refinement_parameters': {
                'iter_graph': iter_graph,
                'gat_threshold': gat_threshold,
                'refine_frequency': refine_frequency
            }
        }

    for epoch in range(1, 11):
        # Training with potentially refined loader
        collect_att = iter_graph and epoch % refine_frequency == 0 and epoch > 0
        
        if collect_att:
            train_result = run_epoch(train_loader, train=True, collect_attention=True)
            train_loss, timestep_attention_weights = train_result
        else:
            train_loss = run_epoch(train_loader, train=True)
        val_loss = run_epoch(test_loader, train=False)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        refinement_info = ""
        if collect_att:
            print(f"\t Aggregating attention weights for graph refinement...")
            avg_attention_weights = aggregate_attention_weights(timestep_attention_weights)
            
            if avg_attention_weights:
                num_nodes = X_train.shape[2]  # [N, 30days, 30nodes, F]
                if current_edge_indices is not None:
                    base_edge_indices = current_edge_indices
                    total_edges_before = sum(ei.shape[1] for ei in current_edge_indices)
                    avg_edges_before = total_edges_before / len(current_edge_indices)
                else:
                    base_edge_indices = edge_index
                    edges_with_self_loops = original_edge_count + num_nodes
                    total_edges_before = edges_with_self_loops * 30
                    avg_edges_before = edges_with_self_loops

                refined_edge_indices = refine_edge_index_by_attention(base_edge_indices, avg_attention_weights, gat_threshold, num_nodes)

                if not load_checkpoint:
                    edge_change_info = {
                        'epoch': epoch,
                        'edge_indices_before': [ei.clone() for ei in (base_edge_indices if isinstance(base_edge_indices, list) else [base_edge_indices])],
                        'edge_indices_after': [ei.clone() for ei in refined_edge_indices],
                        'total_edges_before': total_edges_before,
                        'total_edges_after': sum(ei.shape[1] for ei in refined_edge_indices),
                        'avg_edges_before': avg_edges_before,
                        'avg_edges_after': sum(ei.shape[1] for ei in refined_edge_indices) / len(refined_edge_indices),
                        'attention_weights': [aw.clone() if aw is not None else None for aw in avg_attention_weights],
                        'threshold_used': gat_threshold
                    }
                    edge_tracking_data['epoch_edge_changes'].append(edge_change_info)
                
                current_edge_indices = refined_edge_indices

                train_dataset = DynamicWindowDataset(X_train, y_train, refined_edge_indices)
                train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda b: list(zip(*b)))

                total_edges_after = sum(ei.shape[1] for ei in refined_edge_indices)
                avg_edges_after = total_edges_after / len(refined_edge_indices)
                total_edges_removed = total_edges_before - total_edges_after
                refinement_info = f"{total_edges_removed} edges removed (avg {avg_edges_before:.1f} -> {avg_edges_after:.1f})"
                print(f"\t Refinement info: {refinement_info}")

        # Log epoch and save checkpoint if best
        is_best = logger.log_epoch(epoch, train_loss, val_loss, current_lr)
        if is_best:
            logger.save_checkpoint(model, optimizer, epoch, val_loss)

    if not load_checkpoint:
        edge_tracking_filename = os.path.join("logs", model_name, SCALER_TYPE, f"edge_tracking_{logger.timestamp}.pkl")
        with open(edge_tracking_filename, 'wb') as f:
            pickle.dump(edge_tracking_data, f)
        print(f"Edge tracking data saved to: {edge_tracking_filename}")

    print(f"\nEvaluating model...")
    overall_r2, daily_r2_scores, reservoir_r2_scores = evaluate_model(model, test_loader, encode_map, scaler_data)
    
    # Save final training results
    evaluation_info = f"\nFinal Evaluation Results:\n"
    evaluation_info += f"Overall R2 Score: {overall_r2:.4f}\n"
    evaluation_info += f"Daily R2 Scores: {[f'{r:.4f}' for r in daily_r2_scores]}\n"
    evaluation_info += f"Model: {model_name}\n"
    evaluation_info += f"Final Best Validation Loss: {logger.best_val_loss:.6f}\n"
    evaluation_info += f"Iterative Graph Refinement: {'Enabled' if iter_graph else 'Disabled'}\n"
    if iter_graph:
        evaluation_info += f"GAT Attention Threshold: {gat_threshold}\n"
        evaluation_info += f"Graph Refinement Frequency: every {refine_frequency} epochs\n"
    logger.save_results(evaluation_info)
