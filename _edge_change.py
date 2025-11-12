import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns

class EdgeChangeAnalyzer:
    def __init__(self, edge_tracking_file):
        """
        Initialize the analyzer with edge tracking data.
        
        Args:
            edge_tracking_file: Path to the edge_tracking_*.pkl file
        """
        with open(edge_tracking_file, 'rb') as f:
            self.data = pickle.load(f)
        
        self.original_edge_index = self.data['original_edge_index']
        self.original_edge_count = self.data['original_edge_count']
        self.refinement_params = self.data['refinement_parameters']
        self.epoch_changes = self.data['epoch_edge_changes']
        
        print(f"Loaded edge tracking data:")
        print(f"  - Original edge count: {self.original_edge_count}")
        print(f"  - Refinement epochs: {len(self.epoch_changes)}")
        print(f"  - Refinement parameters: {self.refinement_params}")
    
    def get_edge_index_at_epoch_timestep(self, epoch, timestep):
        """
        Get the edge_index at a specific epoch and timestep.
        
        Args:
            epoch: Training epoch (must be a refinement epoch)
            timestep: Timestep within the sequence (0-29)
            
        Returns:
            edge_index tensor for the specified epoch and timestep
        """
        # Find the epoch data
        epoch_data = None
        for change in self.epoch_changes:
            if change['epoch'] == epoch:
                epoch_data = change
                break
        
        if epoch_data is None:
            print(f"No refinement data found for epoch {epoch}")
            available_epochs = [change['epoch'] for change in self.epoch_changes]
            print(f"Available refinement epochs: {available_epochs}")
            return None
        
        if timestep < 0 or timestep >= len(epoch_data['edge_indices_after']):
            print(f"Invalid timestep {timestep}. Must be 0-{len(epoch_data['edge_indices_after'])-1}")
            return None
        
        return epoch_data['edge_indices_after'][timestep]
    
    def get_edge_changes_summary(self):
        """
        Get a summary of edge changes across all epochs.
        
        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            'epochs': [],
            'total_edges_before': [],
            'total_edges_after': [],
            'edges_removed': [],
            'removal_percentage': []
        }
        
        for change in self.epoch_changes:
            summary['epochs'].append(change['epoch'])
            summary['total_edges_before'].append(change['total_edges_before'])
            summary['total_edges_after'].append(change['total_edges_after'])
            
            removed = change['total_edges_before'] - change['total_edges_after']
            summary['edges_removed'].append(removed)
            summary['removal_percentage'].append(removed / change['total_edges_before'] * 100)
        
        return summary
    
    def visualize_edge_changes(self, save_path=None):
        """
        Visualize edge changes over epochs.
        
        Args:
            save_path: Path to save the plot (optional)
        """
        summary = self.get_edge_changes_summary()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Edge count changes
        ax1.plot(summary['epochs'], summary['total_edges_before'], 'b-o', label='Before refinement')
        ax1.plot(summary['epochs'], summary['total_edges_after'], 'r-o', label='After refinement')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Total Edges')
        ax1.set_title('Edge Count Changes Over Epochs')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Removal percentage
        ax2.bar(summary['epochs'], summary['removal_percentage'], alpha=0.7)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Edges Removed (%)')
        ax2.set_title('Percentage of Edges Removed')
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        
        plt.show()
    
    def analyze_timestep_patterns(self, epoch):
        """
        Analyze edge removal patterns across timesteps for a specific epoch.
        
        Args:
            epoch: Training epoch to analyze
        """
        epoch_data = None
        for change in self.epoch_changes:
            if change['epoch'] == epoch:
                epoch_data = change
                break
        
        if epoch_data is None:
            print(f"No data found for epoch {epoch}")
            return
        
        timestep_edges = []
        for t in range(len(epoch_data['edge_indices_after'])):
            edges_after = epoch_data['edge_indices_after'][t].shape[1]
            timestep_edges.append(edges_after)
        
        print(f"\nTimestep edge analysis for epoch {epoch}:")
        print(f"Threshold used: {epoch_data['threshold_used']}")
        print(f"Total edges before: {epoch_data['total_edges_before']}")
        print(f"Total edges after: {epoch_data['total_edges_after']}")
        
        print(f"\nEdges per timestep:")
        for t, edges in enumerate(timestep_edges):
            print(f"  Timestep {t}: {edges} edges")
        
        # Visualize timestep patterns
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(timestep_edges)), timestep_edges, 'b-o')
        plt.xlabel('Timestep')
        plt.ylabel('Number of Edges')
        plt.title(f'Edge Count per Timestep (Epoch {epoch})')
        plt.grid(True)
        plt.show()
    
    def compare_edge_indices(self, epoch1, epoch2, timestep):
        """
        Compare edge indices between two epochs at a specific timestep.
        
        Args:
            epoch1, epoch2: Epochs to compare
            timestep: Timestep to compare
        """
        edge1 = self.get_edge_index_at_epoch_timestep(epoch1, timestep)
        edge2 = self.get_edge_index_at_epoch_timestep(epoch2, timestep)
        
        if edge1 is None or edge2 is None:
            return
        
        print(f"\nComparison between epoch {epoch1} and epoch {epoch2} at timestep {timestep}:")
        print(f"Epoch {epoch1}: {edge1.shape[1]} edges")
        print(f"Epoch {epoch2}: {edge2.shape[1]} edges")
        
        # Find common and unique edges
        edge1_set = set([(edge1[0][i].item(), edge1[1][i].item()) for i in range(edge1.shape[1])])
        edge2_set = set([(edge2[0][i].item(), edge2[1][i].item()) for i in range(edge2.shape[1])])
        
        common_edges = edge1_set & edge2_set
        unique_to_epoch1 = edge1_set - edge2_set
        unique_to_epoch2 = edge2_set - edge1_set
        
        print(f"Common edges: {len(common_edges)}")
        print(f"Unique to epoch {epoch1}: {len(unique_to_epoch1)}")
        print(f"Unique to epoch {epoch2}: {len(unique_to_epoch2)}")
        
        return {
            'common': common_edges,
            'unique_epoch1': unique_to_epoch1,
            'unique_epoch2': unique_to_epoch2
        }
    
    def get_attention_weights_at_epoch(self, epoch):
        """
        Get attention weights for a specific epoch.
        
        Args:
            epoch: Training epoch
            
        Returns:
            List of attention weights for each timestep
        """
        epoch_data = None
        for change in self.epoch_changes:
            if change['epoch'] == epoch:
                epoch_data = change
                break
        
        if epoch_data is None:
            print(f"No data found for epoch {epoch}")
            return None
        
        return epoch_data['attention_weights']
    
    def analyze_attention_patterns(self, epoch):
        """
        Analyze attention weight patterns for a specific epoch.
        
        Args:
            epoch: Training epoch to analyze
        """
        attention_weights = self.get_attention_weights_at_epoch(epoch)
        
        if attention_weights is None:
            return
        
        print(f"\nAttention analysis for epoch {epoch}:")
        
        # Calculate statistics for each timestep
        for t, weights in enumerate(attention_weights):
            if weights is not None:
                weights_np = weights.detach().cpu().numpy()
                print(f"Timestep {t}:")
                print(f"  Mean attention: {weights_np.mean():.4f}")
                print(f"  Std attention: {weights_np.std():.4f}")
                print(f"  Min attention: {weights_np.min():.4f}")
                print(f"  Max attention: {weights_np.max():.4f}")
                print(f"  Edges above threshold: {(weights_np >= self.refinement_params['gat_threshold']).sum()}")
        
        # Visualize attention distributions
        valid_weights = [w for w in attention_weights if w is not None]
        if valid_weights:
            plt.figure(figsize=(15, 10))
            
            # Create subplots for first few timesteps
            n_plots = min(6, len(valid_weights))
            for i in range(n_plots):
                plt.subplot(2, 3, i+1)
                weights_np = valid_weights[i].detach().cpu().numpy()
                plt.hist(weights_np, bins=50, alpha=0.7)
                plt.axvline(self.refinement_params['gat_threshold'], color='red', linestyle='--', 
                           label=f'Threshold ({self.refinement_params["gat_threshold"]})')
                plt.xlabel('Attention Weight')
                plt.ylabel('Frequency')
                plt.title(f'Timestep {i}')
                plt.legend()
            
            plt.tight_layout()
            plt.show()


def main():
    """
    Main function to demonstrate usage of the EdgeChangeAnalyzer.
    """
    # Example usage - you need to specify the path to your edge tracking file
    
    # Find the most recent edge tracking file
    logs_dir = "logs"
    edge_tracking_files = []
    
    for root, dirs, files in os.walk(logs_dir):
        for file in files:
            if file.startswith("edge_tracking_") and file.endswith(".pkl"):
                edge_tracking_files.append(os.path.join(root, file))
    
    if not edge_tracking_files:
        print("No edge tracking files found. Please run training with load_pretrain=False first.")
        return
    
    # Use the most recent file
    latest_file = max(edge_tracking_files, key=os.path.getmtime)
    print(f"Using edge tracking file: {latest_file}")
    
    # Initialize analyzer
    analyzer = EdgeChangeAnalyzer(latest_file)
    
    # Example analyses
    print("\n" + "="*50)
    print("EDGE CHANGE SUMMARY")
    print("="*50)
    
    # Show summary
    summary = analyzer.get_edge_changes_summary()
    for i, epoch in enumerate(summary['epochs']):
        print(f"Epoch {epoch}:")
        print(f"  Edges before: {summary['total_edges_before'][i]}")
        print(f"  Edges after: {summary['total_edges_after'][i]}")
        print(f"  Removed: {summary['edges_removed'][i]} ({summary['removal_percentage'][i]:.1f}%)")
    
    # Visualize changes
    analyzer.visualize_edge_changes()
    
    # Analyze first refinement epoch in detail
    if summary['epochs']:
        first_epoch = summary['epochs'][0]
        print(f"\n" + "="*50)
        print(f"DETAILED ANALYSIS FOR EPOCH {first_epoch}")
        print("="*50)
        
        analyzer.analyze_timestep_patterns(first_epoch)
        analyzer.analyze_attention_patterns(first_epoch)
        
        # Show how to get specific edge indices
        print(f"\nExample: Getting edge_index for epoch {first_epoch}, timestep 0:")
        edge_index = analyzer.get_edge_index_at_epoch_timestep(first_epoch, 0)
        if edge_index is not None:
            print(f"Shape: {edge_index.shape}")
            print(f"First 5 edges: {edge_index[:, :5]}")


if __name__ == "__main__":
    main()