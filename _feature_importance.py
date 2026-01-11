import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader

# Import necessary modules from existing codebase
from utils import WindowDataset, seed_everything, _adjust_checkpoint_time
from models.gnn import ReservoirAttentionNet


def create_model_wrapper(model, day_index):
    """Create a wrapper that only outputs predictions for a specific day."""
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model, day_idx):
            super().__init__()
            self.model = model
            self.day_idx = day_idx
        
        def forward(self, graph_list):
            # Get full predictions [N, 7]
            pred = self.model(graph_list)
            # Return only the prediction for the specified day [N]
            day_pred = pred[:, self.day_idx]
            # Sum across all nodes to get a scalar output for IntegratedGradients
            return day_pred.sum()
    
    return ModelWrapper(model, day_index)


def extract_features_from_graphs(graph_list):
    """Extract features from graph list for integrated gradients."""
    # Extract node features from each graph in the sequence
    features_list = []
    for graph in graph_list:
        features_list.append(graph.x)  # [num_nodes, num_features]
    
    # Stack to create [num_days, num_nodes, num_features] = [30, num_nodes, 3]
    stacked_features = torch.stack(features_list, dim=0)
    return stacked_features


def create_feature_forward_func(model, graph_list_template, device):
    """Create a forward function that takes features as input for IntegratedGradients."""
    def forward_func(input_features):
        """
        input_features: [batch_size, num_days, num_nodes, num_features]
        """
        batch_size = input_features.shape[0]
        predictions = []
        
        for b in range(batch_size):
            # Instead of cloning, directly modify the features in-place for efficiency
            # Store original features for restoration
            original_features = [g.x.clone() for g in graph_list_template]
            
            try:
                # Update features in-place
                for t, graph in enumerate(graph_list_template):
                    graph.x = input_features[b, t]  # [num_nodes, num_features]
                
                # Get prediction for this sample (now returns scalar)
                pred = model(graph_list_template)  # scalar
                predictions.append(pred)
            finally:
                # Restore original features
                for t, graph in enumerate(graph_list_template):
                    graph.x = original_features[t]
        
        # Since each prediction is now a scalar, stack them into a 1D tensor
        stacked_preds = torch.stack(predictions, dim=0)
        # If batch_size is 1, return scalar; otherwise return the tensor
        if batch_size == 1:
            return stacked_preds[0]  # Return scalar for IntegratedGradients
        return stacked_preds
    
    return forward_func


def analyze_feature_importance(checkpoint_time="202507101845_p", scaler_type="local", num_samples=50, normalize_scores=True):
    """Analyze feature importance using Integrated Gradients."""
    
    # Set random seed for reproducibility
    SEED = 42
    seed_everything(SEED)
    
    print(f"\nAnalyzing feature importance using Integrated Gradients")
    print(f"Checkpoint: {checkpoint_time}")
    print(f"Scaler type: {scaler_type}")
    print(f"Number of samples: {num_samples}")
    print(f"Normalize scores: {normalize_scores}")
    
    # Setup paths and load data
    data_path = "./data"
    parsed_path = os.path.join(data_path, 'parsed')
    graph_path = os.path.join(data_path, 'graph')
    
    # Load scaler data
    all_rsr_data_file = os.path.join(parsed_path, f"all_rsr_data_{scaler_type}.pkl")
    if not os.path.exists(all_rsr_data_file):
        raise FileNotFoundError(f"Preprocessed data not found: {all_rsr_data_file}")
    
    with open(all_rsr_data_file, 'rb') as f:
        all_rsr_data = pickle.load(f)
    scaler_data = {
        'scaler_X': all_rsr_data.get('scaler_X'),
        'scaler_y': all_rsr_data.get('scaler_y'),
        'local_scalers_X': all_rsr_data.get('local_scalers_X'),
        'local_scalers_y': all_rsr_data.get('local_scalers_y'),
        'params': all_rsr_data.get('params', {})
    }
    del all_rsr_data
    
    # Load graph data
    graph_cfg = "k2_config3"
    graph_file = os.path.join(graph_path, f"{graph_cfg}.pkl")
    if not os.path.exists(graph_file):
        raise FileNotFoundError(f"Graph file not found: {graph_file}")
    
    with open(graph_file, 'rb') as f:
        graph_data = pickle.load(f)
    
    edge_index = graph_data['edge_index']
    encode_map = graph_data['encode_map']
    
    # Load test data
    supervised_file = os.path.join(parsed_path, f"_GNN_supervise_{scaler_type}.pt")
    if not os.path.exists(supervised_file):
        raise FileNotFoundError(f"Supervised data file not found: {supervised_file}")
    
    supervised_data = torch.load(supervised_file, weights_only=True)
    X_test, y_test = supervised_data['X_test'], supervised_data['y_test']
    
    # Limit samples for analysis
    X_test = X_test[:num_samples]
    y_test = y_test[:num_samples]
    
    # Create test dataset and loader
    test_dataset = WindowDataset(X_test, y_test, edge_index)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda b: list(zip(*b)))
    
    # Setup device and model
    device = torch.device("cpu")  # Using CPU to avoid CUDA memory issues
    input_dim = X_test.shape[-1]
    model = ReservoirAttentionNet(in_dim=input_dim, hid_dim=128,
                                  gnn_dim=64, lstm_dim=64, pred_days=7).to(device)
    
    # Load checkpoint
    model_name = model.__class__.__name__
    adjusted_time = _adjust_checkpoint_time(model_name, scaler_type, checkpoint_time)
    if adjusted_time is None:
        raise FileNotFoundError(f"No checkpoint files found for model '{model_name}' with scaler type '{scaler_type}'")
    
    checkpoint_path = os.path.join("logs", model_name, scaler_type, f"checkpoint_{adjusted_time}.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Initialize results storage
    all_attributions = {f'Day{i+1}': [] for i in range(7)}
    feature_names = ['Temperature', 'Precipitation', 'Inflow']
    
    print("Starting Integrated Gradients analysis...")
    
    # Pre-create model wrappers and forward functions to avoid recreation
    model_wrappers = {}
    forward_funcs = {}
    
    sample_count = 0
    # Don't use torch.no_grad() as IntegratedGradients needs gradients
    for graph_batch, target_batch in tqdm(test_loader, desc=f"Analyzing samples", total=num_samples):
            if sample_count >= num_samples:
                break
                
            graphs, targets = graph_batch[0], target_batch[0]  # Get single sample
            graphs = [g.to(device) for g in graphs]
            
            # Extract features from graphs
            input_features = extract_features_from_graphs(graphs)
            input_features = input_features.unsqueeze(0)  # Add batch dimension [1, 30, num_nodes, 3]
            input_features.requires_grad_(True)
            
            # Create baseline once per sample (zeros)
            baseline = torch.zeros_like(input_features)
            
            # For each prediction day, calculate feature importance
            for day_idx in range(7):
                # Create or reuse model wrapper for this specific day
                if day_idx not in model_wrappers:
                    model_wrappers[day_idx] = create_model_wrapper(model, day_idx)
                    forward_funcs[day_idx] = create_feature_forward_func(model_wrappers[day_idx], graphs, device)
                
                # Initialize Integrated Gradients with reused forward function
                ig = IntegratedGradients(forward_funcs[day_idx])
                
                # Calculate attributions (reduced steps for faster computation)
                attributions = ig.attribute(input_features, baseline, n_steps=20)
                
                # Average attributions across nodes and time steps
                # attributions shape: [1, 30, num_nodes, 3]
                # Average across nodes (dim=2) and time steps (dim=1)
                avg_attributions = attributions.squeeze(0).mean(dim=[0, 1])  # [3]
                
                all_attributions[f'Day{day_idx+1}'].append(avg_attributions.detach().cpu().numpy())
            
            sample_count += 1
            
            # Clear cache periodically to manage memory
            if sample_count % 5 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    print(f"\nAnalysis completed! Processing {sample_count} samples...")
    
    # Calculate mean and std across all samples
    daily_importance = {}
    daily_importance_std = {}
    
    for day in all_attributions:
        day_attrs = np.array(all_attributions[day])  # [num_samples, 3]
        daily_importance[day] = np.mean(np.abs(day_attrs), axis=0)  # Take absolute values and average
        daily_importance_std[day] = np.std(np.abs(day_attrs), axis=0)
    
    # Optional normalization to 0-1 scale
    if normalize_scores:
        # Find the maximum importance value across all days and features for normalization
        max_importance = 0
        for day_key in daily_importance:
            max_importance = max(max_importance, daily_importance[day_key].max())
        
        # Normalize all values to 0-1 range
        if max_importance > 0:
            for day_key in daily_importance:
                daily_importance[day_key] = daily_importance[day_key] / max_importance
                daily_importance_std[day_key] = daily_importance_std[day_key] / max_importance
    
    # Create bar chart visualization
    days = list(range(1, 8))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Prepare data for bar chart
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue (Temperature), Orange (Precipitation), Green (Inflow)
    n_days = len(days)
    n_features = len(feature_names)
    
    # Set bar width and positions
    bar_width = 0.25
    day_positions = np.arange(n_days)
    
    # Create bars for each feature
    for i, feature in enumerate(feature_names):
        importance_values = [daily_importance[f'Day{day}'][i] for day in days]
        std_values = [daily_importance_std[f'Day{day}'][i] for day in days]
        
        # Convert to percentage for display - normalize so each day sums to 100%
        importance_percentage = []
        for day_idx, day in enumerate(days):
            day_key = f'Day{day}'
            day_total = sum(daily_importance[day_key])  # Sum all features for this day
            if day_total > 0:
                percentage = (daily_importance[day_key][i] / day_total) * 100
            else:
                percentage = 0
            importance_percentage.append(percentage)
        
        # Position bars for each feature
        bar_positions = day_positions + i * bar_width - bar_width
        
        bars = ax.bar(bar_positions, importance_percentage, 
                     width=bar_width, 
                     color=colors[i], 
                     alpha=0.8,
                     label=feature,
                     edgecolor='white',
                     linewidth=0.8)
        
        # Error bars removed to keep y-axis scale proper
        
        # Value labels removed for cleaner appearance
    
    # Customize the plot
    ax.set_ylabel('Importance of Variables (%)', fontsize=14, fontweight='bold')
    
    # Set x-axis labels and ticks
    ax.set_xticks(day_positions)
    ax.set_xticklabels([f'Day{d}' for d in days], fontsize=16)
    ax.legend(loc='best', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # Set axis limits with some padding
    ax.set_xlim(-0.5, n_days - 0.5)
    ax.set_ylim(0, 100)
    
    plt.tight_layout()
    
    # Save the plot as PDF (vector format)
    output_path = "daily_feature_importance.pdf"
    plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Feature importance plot saved as: {output_path}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE RESULTS")
    print("="*60)
    
    # Calculate overall feature importance (averaged across all days)
    overall_importance = np.zeros(3)
    for day_key in daily_importance:
        overall_importance += daily_importance[day_key]
    overall_importance /= 7
    
    print("\nOverall Feature Importance (averaged across all days):")
    for i, feature in enumerate(feature_names):
        print(f"{feature:>12}: {overall_importance[i]:.6f}")
    
    print("\nDaily Feature Importance:")
    for day in range(1, 8):
        day_key = f'Day{day}'
        print(f"\n{day_key}:")
        for i, feature in enumerate(feature_names):
            mean_val = daily_importance[day_key][i]
            std_val = daily_importance_std[day_key][i]
            print(f"  {feature:>12}: {mean_val:.6f} ± {std_val:.6f}")
    
    return daily_importance, daily_importance_std, overall_importance


if __name__ == "__main__":
    # Configuration
    checkpoint_time = "202507101845_p"  # Adjust this to match your checkpoint
    scaler_type = "local"  # or "global"
    num_samples = 5  # Number of test samples to analyze
    normalize_scores = True  # Whether to normalize importance scores to 0-1 range
    
    try:
        daily_importance, daily_std, overall_importance = analyze_feature_importance(
            checkpoint_time=checkpoint_time,
            scaler_type=scaler_type,
            num_samples=num_samples,
            normalize_scores=normalize_scores
        )
        print("\nFeature importance analysis completed successfully!")
        print("Output saved as: daily_feature_importance.pdf")
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()