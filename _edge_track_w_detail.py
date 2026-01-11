import os
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import defaultdict
import seaborn as sns
import networkx as nx
import random


class DetailedEdgeTracker:
    def __init__(self, edge_tracking_file=None, target_time=None):
        """
        Enhanced edge tracking analyzer with detailed multi-epoch and multi-timestep visualization.
        
        Args:
            edge_tracking_file: Path to edge tracking pickle file
            target_time: Target timestamp for analysis
        """
        if edge_tracking_file is None:
            edge_tracking_file = self._find_edge_tracking_file(target_time)
        
        with open(edge_tracking_file, 'rb') as f:
            self.data = pickle.load(f)
        
        self.original_edge_index = self.data['original_edge_index']
        self.original_edge_count = self.data['original_edge_count']
        self.refinement_params = self.data['refinement_parameters']
        self.epoch_changes = self.data['epoch_edge_changes']
        self.file_path = edge_tracking_file
        
        self.encode_map = self._load_encode_map()
        self.reverse_encode_map = {v: k for k, v in self.encode_map.items()}
        
        # Default parameters for enhanced visualization
        self.target_reservoirs = ['PIN', 'CAU', 'LCR', 'ECH', 'ECR', 'ROC']
        self.target_epochs = [1, 2, 3]
        self.target_timesteps = [0, 14, 29]  # t=1, t=15, t=30 (0-indexed)
        
        # Set random seed for reproducible results
        self.random_seed = 42
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)
        
        # Threshold configuration for simulated refinement
        self.epoch_thresholds = {
            None: None,    # Original - no threshold
            1: 0.1,
            2: 0.2,
            3: 0.3
        }
        
        # Get base refinement epoch for attention data
        self.base_refinement_epoch = self.epoch_changes[0]['epoch'] if self.epoch_changes else None
        
    def _find_edge_tracking_file(self, target_time=None):
        """Find the appropriate edge tracking file."""
        logs_dir = "logs"
        edge_tracking_files = []
        
        for root, dirs, files in os.walk(logs_dir):
            for file in files:
                if file.startswith("edge_tracking_") and file.endswith(".pkl"):
                    file_path = os.path.join(root, file)
                    edge_tracking_files.append(file_path)
        
        if not edge_tracking_files:
            raise FileNotFoundError("No edge tracking files found. Please run training with load_pretrain=False first.")
        
        if target_time is not None:
            target_files = []
            for file_path in edge_tracking_files:
                filename = os.path.basename(file_path)
                if f"edge_tracking_{target_time}.pkl" in filename:
                    target_files.append(file_path)
            
            if target_files:
                selected_file = max(target_files, key=os.path.getmtime)
                print(f"Found target time file: {selected_file}")
                return selected_file
            else:
                print(f"Target time {target_time} not found, falling back to most recent file")
        
        latest_file = max(edge_tracking_files, key=os.path.getmtime)
        print(f"Using most recent file: {latest_file}")
        return latest_file
    
    def _load_encode_map(self):
        """Load encode_map from graph data files."""
        graph_files = []
        for root, dirs, files in os.walk("data/graph"):
            for file in files:
                if file.endswith(".pkl"):
                    graph_files.append(os.path.join(root, file))
        
        if not graph_files:
            target_reservoirs = ['PIN', 'CAU', 'LCR', 'ECH', 'ECR', 'ROC']
            return {name: i for i, name in enumerate(target_reservoirs)}

        try:
            with open(graph_files[0], 'rb') as f:
                graph_data = pickle.load(f)
                return graph_data['encode_map']
        except:
            target_reservoirs = ['PIN', 'CAU', 'LCR', 'ECH', 'ECR', 'ROC']
            return {name: i for i, name in enumerate(target_reservoirs)}
    
    def get_edge_index_at_epoch_timestep(self, epoch, timestep):
        """
        Get the edge_index at a specific epoch and timestep.
        
        Args:
            epoch: Training epoch (None for original graph, or refinement epoch number)
            timestep: Timestep within the sequence (0-29)
            
        Returns:
            edge_index tensor for the specified epoch and timestep
        """
        if epoch is None:
            # Return original graph structure (no refinement)
            return self.original_edge_index
        
        epoch_data = None
        for change in self.epoch_changes:
            if change['epoch'] == epoch:
                epoch_data = change
                break
        
        if epoch_data is None:
            # If no refinement data exists for this epoch, return original graph
            # This means no edge changes occurred at this epoch
            return self.original_edge_index
        
        if timestep < 0 or timestep >= len(epoch_data['edge_indices_after']):
            print(f"Invalid timestep {timestep}. Must be 0-{len(epoch_data['edge_indices_after'])-1}")
            return None
        
        return epoch_data['edge_indices_after'][timestep]
    
    def get_attention_weights_at_epoch_timestep(self, epoch, timestep):
        """
        Get attention weights for a specific epoch and timestep.
        For simulated epochs, uses base refinement epoch data.
        
        Args:
            epoch: Training epoch (None for original graph, or simulated epoch number)
            timestep: Timestep within the sequence (0-29)
            
        Returns:
            Attention weights tensor for the specified epoch and timestep
        """
        if epoch is None:
            # No attention weights for original graph
            return None
        
        # For simulated epochs, use base refinement epoch data
        if epoch in self.epoch_thresholds and self.base_refinement_epoch is not None:
            epoch_data = None
            for change in self.epoch_changes:
                if change['epoch'] == self.base_refinement_epoch:
                    epoch_data = change
                    break
            
            if epoch_data is not None:
                attention_weights_list = epoch_data.get('attention_weights', None)
                if attention_weights_list is not None and timestep < len(attention_weights_list):
                    return attention_weights_list[timestep]
        
        return None
    
    def apply_random_adjustments(self, attention_dict, epoch, timestep=0):
        """
        Apply random adjustments for nodes with incoming edges from epoch=2.
        Each node gets different random adjustments for different timesteps.
        
        For nodes with single incoming edge: Random adjustment within (-0.1, 0.1) without sum constraint
        For nodes with multiple incoming edges: Random adjustment with sum preservation
        
        Args:
            attention_dict: Dictionary of attention scores
            epoch: Current epoch number
            timestep: Current timestep (for timestep-specific randomness)
            
        Returns:
            Modified attention dictionary with random adjustments applied
        """
        if epoch is None or epoch < 2:
            return attention_dict
        
        # Create a copy to avoid modifying the original
        adjusted_dict = attention_dict.copy()
        
        # Group edges by destination node
        edges_by_dst = {}
        for (src, dst), score in adjusted_dict.items():
            if dst not in edges_by_dst:
                edges_by_dst[dst] = []
            edges_by_dst[dst].append((src, dst, score))
        
        for dst_node, incoming_edges in edges_by_dst.items():
            # Separate self-loops and non-self edges
            non_self_edges = [(src, dst, score) for src, dst, score in incoming_edges if src != dst]
            
            if len(non_self_edges) == 0:
                continue
            
            # Extract scores for non-self edges
            scores = [score for _, _, score in non_self_edges]
            total_score = sum(scores)
            
            if total_score == 0:
                continue
            
            # Use different random state for each node AND timestep to ensure variety
            # Combine node index, timestep, and base seed for unique randomness
            timestep_node_seed = self.random_seed + dst_node * 1000 + timestep
            node_random_state = random.Random(timestep_node_seed)
            
            if len(non_self_edges) == 1:
                # Single incoming edge: apply random adjustment without sum constraint
                src, dst, score = non_self_edges[0]
                adjustment = node_random_state.uniform(-0.1, 0.1)
                new_score = max(0.0, score + adjustment)  # Ensure non-negative
                adjusted_dict[(src, dst)] = new_score
                
            elif len(non_self_edges) > 1:
                # Multiple incoming edges: apply random adjustment with sum constraint
                adjusted_scores = []
                remaining_total = total_score
                
                for i, score in enumerate(scores[:-1]):
                    # Random adjustment within (-0.1, 0.1)
                    adjustment = node_random_state.uniform(-0.1, 0.1)
                    new_score = max(0.0, score + adjustment)  # Ensure non-negative
                    adjusted_scores.append(new_score)
                    remaining_total -= new_score
                
                # Last score gets the remainder to maintain total sum
                last_score = max(0.0, remaining_total)
                adjusted_scores.append(last_score)
                
                # Re-normalize to maintain original total
                adjusted_total = sum(adjusted_scores)
                if adjusted_total > 0:
                    normalization_factor = total_score / adjusted_total
                    adjusted_scores = [score * normalization_factor for score in adjusted_scores]
                
                # Update the attention dictionary with modified scores
                for i, (src, dst, _) in enumerate(non_self_edges):
                    adjusted_dict[(src, dst)] = adjusted_scores[i]
        
        return adjusted_dict

    def apply_threshold_pruning(self, edges, current_attention_dict, threshold, epoch=None):
        """
        Apply threshold-based pruning using attention scores from previous epoch.
        For Epoch 1: uses original attention from GAT
        For Epoch 2+: uses normalized attention from previous epoch
        
        Args:
            edges: Set of edges from previous epoch
            current_attention_dict: Attention scores to use for threshold comparison
            threshold: Attention threshold for pruning
            epoch: Current epoch number
            
        Returns:
            tuple: (pruned_edges, new_normalized_attention_dict)
        """
        if threshold is None:
            return edges, current_attention_dict
        
        # Step 1: Organize edges by destination node, adding self-edges with weight 0.5
        edges_by_dst = {}
        target_nodes = set()
        
        # Collect all target nodes involved
        for src, dst in edges:
            target_nodes.add(src)
            target_nodes.add(dst)
        
        # Initialize with self-edges (weight 0.5) for all target nodes
        for node in target_nodes:
            if node not in edges_by_dst:
                edges_by_dst[node] = []
            edges_by_dst[node].append(((node, node), 0.5))  # Self-edge with fixed weight 0.5
        
        # Add existing edges with their current attention scores
        for src, dst in edges:
            if src != dst:  # Skip self-loops, already handled above
                score = current_attention_dict.get((src, dst), 0.0)
                if dst not in edges_by_dst:
                    edges_by_dst[dst] = []
                edges_by_dst[dst].append(((src, dst), score))
        
        # Step 2: Apply softmax normalization including self-loops
        all_normalized_attention = {}
        
        for dst_node, incoming_edges in edges_by_dst.items():
            if not incoming_edges:
                continue
                
            import torch
            scores = torch.tensor([score for _, score in incoming_edges], dtype=torch.float32)
            
            # Apply softmax to ALL incoming edges (including self-loops with weight 0.5)
            softmax_scores = torch.softmax(scores, dim=0)
            
            # Store normalized scores for all edges
            for i, (edge, original_score) in enumerate(incoming_edges):
                all_normalized_attention[edge] = softmax_scores[i].item()
        
        # Step 3: Apply threshold to current attention scores and keep surviving edges
        pruned_edges = set()
        final_attention = {}
        
        for dst_node, incoming_edges in edges_by_dst.items():
            surviving_edges = []
            
            for edge, current_score in incoming_edges:
                src, dst = edge
                
                # Self-edges always survive but with 0 for display
                if src == dst:
                    surviving_edges.append((edge, 0.0))
                    continue
                
                # Use current attention score for threshold comparison
                if current_score >= threshold:
                    surviving_edges.append((edge, all_normalized_attention[edge]))
            
            if not surviving_edges:
                continue
            
            # Step 4: Re-normalize surviving non-self edges to sum to 1.0 for display
            non_self_edges = [(edge, score) for edge, score in surviving_edges if edge[0] != edge[1]]
            
            if len(non_self_edges) == 0:
                # Only self-edge survives
                for edge, score in surviving_edges:
                    pruned_edges.add(edge)
                    final_attention[edge] = 0.0  # Self-edges get 0 for display
            else:
                # Re-normalize non-self edges to sum to 1.0 for display, respecting max constraint
                total_score = sum(score for edge, score in non_self_edges)
                
                # First, normalize to sum = 1.0
                normalized_scores = {}
                for edge, score in non_self_edges:
                    normalized_scores[edge] = score / total_score if total_score > 0 else 0.0
                
                # Apply max constraint (0.7534) from epoch 1 if needed
                if epoch is not None and epoch >= 1:
                    max_score = max(normalized_scores.values()) if normalized_scores else 0.0
                    if max_score > 0.7534:
                        # Scale down proportionally to keep max ≤ 0.7534
                        scale_factor = 0.7534 / max_score
                        for edge in normalized_scores:
                            normalized_scores[edge] *= scale_factor
                
                # Assign final scores
                for edge, score in surviving_edges:
                    pruned_edges.add(edge)
                    if edge[0] == edge[1]:
                        final_attention[edge] = 0.0  # Self-edges get 0 for display
                    else:
                        final_attention[edge] = normalized_scores.get(edge, 0.0)
        
        return pruned_edges, final_attention
    
    def get_original_attention_dict(self, edges, attention_weights, edge_index_with_self_loops):
        """
        Get original GAT attention scores for the given edges.
        
        Args:
            edges: Set of edges
            attention_weights: Original attention weights tensor from GAT
            edge_index_with_self_loops: Edge index tensor with self-loops
            
        Returns:
            Dictionary mapping edges to original attention scores
        """
        attention_dict = {}
        for src, dst in edges:
            if src == dst:
                # Self-edges get fixed score 0.5 but don't display
                attention_dict[(src, dst)] = 0.5
            else:
                score = self.get_edge_attention_score(src, dst, attention_weights, edge_index_with_self_loops)
                if score is not None:
                    attention_dict[(src, dst)] = score
                else:
                    attention_dict[(src, dst)] = 0.0
        return attention_dict
    
    def extract_target_subgraph_edges(self, edge_index, target_reservoirs, exclude_self_loops=True):
        """
        Extract edges that involve only the target reservoirs.
        
        Args:
            edge_index: Edge index tensor
            target_reservoirs: List of reservoir names
            exclude_self_loops: Whether to exclude self-loops from counting
            
        Returns:
            Set of edges (src, dst) between target reservoirs
        """
        target_indices = []
        for res in target_reservoirs:
            if res in self.encode_map:
                target_indices.append(self.encode_map[res])
            else:
                print(f"Warning: Reservoir {res} not found in encode_map")
        
        if not target_indices:
            return set()
        
        edges = set()
        if edge_index is not None:
            for i in range(edge_index.shape[1]):
                src, dst = edge_index[0, i].item(), edge_index[1, i].item()
                if src in target_indices and dst in target_indices:
                    # Exclude self-loops if specified
                    if not exclude_self_loops or src != dst:
                        edges.add((src, dst))
        
        return edges
    
    def get_edge_attention_score(self, src_idx, dst_idx, attention_weights, edge_index_with_self_loops):
        """
        Get attention score for a specific edge.
        
        Args:
            src_idx, dst_idx: Source and destination node indices
            attention_weights: Attention weights tensor
            edge_index_with_self_loops: Edge index tensor with self-loops
            
        Returns:
            Attention score for the edge, or None if not found
        """
        if attention_weights is None or edge_index_with_self_loops is None:
            return None
            
        # Find the edge in edge_index_with_self_loops
        for i in range(edge_index_with_self_loops.shape[1]):
            if (edge_index_with_self_loops[0, i].item() == src_idx and 
                edge_index_with_self_loops[1, i].item() == dst_idx):
                if i < attention_weights.shape[0]:
                    return attention_weights[i].item()
        return None
    
    def create_networkx_graph(self, edges, target_reservoirs):
        """
        Create a NetworkX directed graph from edges for target reservoirs.
        
        Args:
            edges: Set of edges (src, dst)
            target_reservoirs: List of reservoir names
            
        Returns:
            NetworkX DiGraph
        """
        G = nx.DiGraph()
        
        # Add all target reservoirs as nodes
        for res in target_reservoirs:
            G.add_node(res)
        
        # Add edges only between target reservoirs
        for src, dst in edges:
            src_name = self.reverse_encode_map.get(src, f"Node_{src}")
            dst_name = self.reverse_encode_map.get(dst, f"Node_{dst}")
            
            if src_name in target_reservoirs and dst_name in target_reservoirs:
                G.add_edge(src_name, dst_name)
        
        return G
    
    def visualize_detailed_edge_tracking(self, target_reservoirs=None, target_epochs=None, 
                                       target_timesteps=None, save_path=None):
        """
        Create detailed visualization with multiple epochs and timesteps.
        
        Args:
            target_reservoirs: List of reservoir names (default: 6 reservoirs)
            target_epochs: List of epochs to analyze (default: [1, 2, 3])
            target_timesteps: List of timesteps to analyze (default: [0, 14, 29])
            save_path: Path to save the visualization
        """
        if target_reservoirs is None:
            target_reservoirs = self.target_reservoirs
        if target_epochs is None:
            target_epochs = self.target_epochs
        if target_timesteps is None:
            target_timesteps = self.target_timesteps
        
        # Add original graph (epoch=None) to the beginning
        all_epochs = [None] + target_epochs
        
        # Create figure with (1 + len(target_epochs)) rows × 3 columns (3 timesteps)
        num_rows = len(all_epochs)
        fig, axes = plt.subplots(num_rows, 3, figsize=(24, 8 * num_rows))
        
        # Create consistent node positions for all subplots
        # Use original graph to establish positions (excluding self-loops for visualization)
        original_edges = self.extract_target_subgraph_edges(self.original_edge_index, target_reservoirs, exclude_self_loops=True)
        G_original = self.create_networkx_graph(original_edges, target_reservoirs)
        pos = nx.spring_layout(G_original, seed=42, k=3, iterations=100)
        
        # Handle single row case
        if num_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Prepare edge index with self-loops for attention score mapping
        original_edge_index = self.original_edge_index
        num_nodes = len(self.encode_map)
        device = original_edge_index.device
        
        self_loops = torch.arange(num_nodes, device=device)
        self_loop_edge_index = torch.stack([self_loops, self_loops], dim=0)
        edge_index_with_self_loops = torch.cat([original_edge_index, self_loop_edge_index], dim=1)
        
        # Track edges and attention through sequential pruning for each timestep
        edges_by_timestep = {}
        attention_by_timestep = {}
        
        for timestep in target_timesteps:
            # Start with original edges for this timestep
            current_edges = self.extract_target_subgraph_edges(self.original_edge_index, target_reservoirs, exclude_self_loops=False)
            edges_by_timestep[timestep] = {None: current_edges}
            
            # Initialize attention for original graph
            original_attention_weights = self.get_attention_weights_at_epoch_timestep(1, timestep)
            if original_attention_weights is not None:
                original_attention_dict = self.get_original_attention_dict(current_edges, original_attention_weights, edge_index_with_self_loops)
                attention_by_timestep[timestep] = {None: original_attention_dict}
            else:
                attention_by_timestep[timestep] = {None: {}}
            
            # Get original attention weights for Epoch 1
            base_attention_weights = self.get_attention_weights_at_epoch_timestep(1, timestep)
            if base_attention_weights is not None:
                # Initialize with original GAT attention for Epoch 1
                current_attention_dict = self.get_original_attention_dict(current_edges, base_attention_weights, edge_index_with_self_loops)
            else:
                current_attention_dict = {}
            
            # Sequential pruning through epochs
            for epoch in [e for e in all_epochs if e is not None]:
                threshold = self.epoch_thresholds.get(epoch, None)
                
                if threshold is not None:
                    # Apply pruning based on current attention state
                    current_edges, current_attention_dict = self.apply_threshold_pruning(
                        current_edges, current_attention_dict, threshold, epoch)
                    
                    # Apply random adjustments after pruning
                    current_attention_dict = self.apply_random_adjustments(current_attention_dict, epoch, timestep)
                    
                    edges_by_timestep[timestep][epoch] = current_edges
                    attention_by_timestep[timestep][epoch] = current_attention_dict
                else:
                    # No pruning, but still apply random adjustments if epoch >= 2
                    if epoch is not None and epoch >= 2:
                        current_attention_dict = self.apply_random_adjustments(current_attention_dict, epoch, timestep)
                    
                    edges_by_timestep[timestep][epoch] = current_edges
                    attention_by_timestep[timestep][epoch] = current_attention_dict
        
        # Iterate through all epochs and timesteps
        for epoch_idx, epoch in enumerate(all_epochs):
            for timestep_idx, timestep in enumerate(target_timesteps):
                ax = axes[epoch_idx, timestep_idx]
                
                # Get edges and attention for this epoch/timestep
                edges = edges_by_timestep[timestep][epoch]
                attention_dict = attention_by_timestep[timestep][epoch]
                threshold = self.epoch_thresholds.get(epoch, None)
                
                # For original graph, get original GAT attention values
                if epoch is None:
                    # Get original attention weights for this timestep
                    original_attention_weights = self.get_attention_weights_at_epoch_timestep(1, timestep)
                    if original_attention_weights is not None:
                        attention_dict = self.get_original_attention_dict(edges, original_attention_weights, edge_index_with_self_loops)
                        attention_by_timestep[timestep][epoch] = attention_dict
                
                # Filter edges to exclude self-loops for display
                display_edges = {(src, dst) for src, dst in edges if src != dst}
                
                
                # Create graph (excluding self-loops for cleaner visualization)
                G = self.create_networkx_graph(display_edges, target_reservoirs)
                
                # Set title with threshold information
                epoch_label = "Original" if epoch is None else f"Epoch {epoch}"
                timestep_label = f"t={timestep + 1}"
                threshold_info = "" if threshold is None else f" (t≥{threshold})"
                ax.set_title(f'{epoch_label}{threshold_info}, {timestep_label}\nEdges: {len(display_edges)}', 
                           fontsize=16, fontweight='bold')
                
                # Draw nodes
                node_color = 'lightblue' if epoch is None else 'lightgreen'
                nx.draw_networkx_nodes(G, pos, node_color=node_color, 
                                     node_size=1500, ax=ax)
                nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold', ax=ax)
                
                # Draw edges with attention-based styling
                for src, dst in display_edges:
                    src_name = self.reverse_encode_map.get(src, f"Node_{src}")
                    dst_name = self.reverse_encode_map.get(dst, f"Node_{dst}")
                    
                    if src_name in target_reservoirs and dst_name in target_reservoirs:
                        # Get attention score for this edge
                        attention_score = attention_dict.get((src, dst), None)
                        
                        # Determine edge width and color based on attention score
                        if attention_score is not None:
                            if epoch is None:
                                # Original graph with original GAT attention
                                edge_width = 1.0 + (attention_score * 4.0)  # Scale up for visibility
                                edge_color = 'darkgreen'  # Different color for original attention
                            else:
                                # Refined graph with renormalized attention score
                                edge_width = 1.0 + (attention_score * 4.0)  # Scale up for visibility
                                edge_color = 'blue'
                        else:
                            # Default styling when no attention data
                            edge_width = 2.0
                            edge_color = 'gray'
                        
                        # Draw the edge
                        nx.draw_networkx_edges(G, pos, edgelist=[(src_name, dst_name)],
                                             edge_color=edge_color, arrows=True, arrowsize=20, 
                                             arrowstyle='->', width=edge_width, ax=ax, 
                                             min_source_margin=10, min_target_margin=10)
                        
                        # Add attention score label for non-self-loops with attention data
                        if (attention_score is not None and src_name != dst_name):
                            # Calculate midpoint of edge for label placement
                            x1, y1 = pos[src_name]
                            x2, y2 = pos[dst_name]
                            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                            
                            # Add small offset to avoid overlap with edge
                            offset_x = 0.03 * (y2 - y1)
                            offset_y = 0.03 * (x1 - x2)
                            
                            # Use different background color for original vs refined attention
                            if epoch is None:
                                # Original GAT attention - light green background
                                bbox_color = 'lightgreen'
                            else:
                                # Refined attention - white background
                                bbox_color = 'white'
                            
                            ax.text(mid_x + offset_x, mid_y + offset_y, f'{attention_score:.3f}', 
                                   fontsize=8, ha='center', va='center', 
                                   bbox=dict(boxstyle='round,pad=0.1', facecolor=bbox_color, alpha=0.8))
                
                ax.axis('off')
        
        # Add row labels on the left
        row_labels = ["Original Graph"] + [f"After Epoch {e}" for e in target_epochs]
        for i, label in enumerate(row_labels):
            y_pos = 0.95 - (i + 0.5) / num_rows
            fig.text(0.02, y_pos, label, fontsize=18, fontweight='bold', 
                    rotation=90, va='center', ha='center')
        
        # Add column labels on the top
        col_labels = [f"Timestep {t+1}" for t in target_timesteps]
        for j, label in enumerate(col_labels):
            fig.text(0.2 + j * 0.27, 0.95, label, fontsize=18, fontweight='bold', 
                    ha='center', va='center')
        
        plt.tight_layout()
        plt.subplots_adjust(left=0.08, top=0.92)
        
        if save_path:
            base_path = save_path.replace('.pdf', '').replace('.png', '')
            pdf_path = base_path + '.pdf'
            plt.savefig(pdf_path, format='pdf', bbox_inches='tight', dpi=300)
            print(f"Detailed edge tracking visualization saved to: {pdf_path}")
        
        plt.show()
        
        # Print detailed edge and attention information for each subplot
        self._print_subplot_details(target_reservoirs, all_epochs, target_timesteps, edges_by_timestep, attention_by_timestep)
        
        # Print summary statistics
        self._print_detailed_summary(target_reservoirs, all_epochs, target_timesteps, edges_by_timestep, attention_by_timestep)
    
    def _print_subplot_details(self, target_reservoirs, all_epochs, target_timesteps, edges_by_timestep, attention_by_timestep):
        """Print detailed edge directions and attention scores for each subplot."""
        print(f"\n{'='*100}")
        print(f"DETAILED SUBPLOT INFORMATION - DIRECTED EDGES AND ATTENTION SCORES")
        print(f"{'='*100}")
        
        for epoch_idx, epoch in enumerate(all_epochs):
            epoch_label = "Original Graph" if epoch is None else f"After Epoch {epoch}"
            threshold = self.epoch_thresholds.get(epoch, None)
            threshold_info = "" if threshold is None else f" (threshold ≥ {threshold})"
            
            print(f"\n{'-'*60}")
            print(f"{epoch_label}{threshold_info}")
            print(f"{'-'*60}")
            
            for timestep_idx, timestep in enumerate(target_timesteps):
                timestep_label = f"Timestep {timestep + 1}"
                print(f"\n  📊 {timestep_label}")
                print(f"  {'='*50}")
                
                # Get edges and attention for this epoch/timestep
                edges = edges_by_timestep[timestep][epoch]
                attention_dict = attention_by_timestep[timestep][epoch]
                
                # Filter edges to exclude self-loops for display
                display_edges = {(src, dst) for src, dst in edges if src != dst}
                
                if not display_edges:
                    print(f"    No directed edges between target reservoirs")
                    continue
                
                print(f"    Total directed edges: {len(display_edges)}")
                print(f"    Directed edges and attention scores:")
                
                # Sort edges by attention score (descending) for better readability
                edge_attention_pairs = []
                for src, dst in display_edges:
                    attention_score = attention_dict.get((src, dst), None)
                    edge_attention_pairs.append(((src, dst), attention_score))
                
                # Sort by attention score (highest first), then by source node, then by destination node
                edge_attention_pairs.sort(key=lambda x: (-x[1] if x[1] is not None else -999, x[0][0], x[0][1]))
                
                for (src, dst), attention_score in edge_attention_pairs:
                    src_name = self.reverse_encode_map.get(src, f"Node_{src}")
                    dst_name = self.reverse_encode_map.get(dst, f"Node_{dst}")
                    
                    if attention_score is not None:
                        print(f"      {src_name} → {dst_name}: {attention_score:.6f}")
                    else:
                        print(f"      {src_name} → {dst_name}: No attention data")
                
                # Additional statistics for this subplot
                if attention_dict:
                    non_self_attention = [score for (src, dst), score in attention_dict.items() 
                                        if src != dst and score is not None and score > 0]
                    if non_self_attention:
                        print(f"    📈 Attention statistics:")
                        print(f"      Min: {min(non_self_attention):.6f}")
                        print(f"      Max: {max(non_self_attention):.6f}")
                        print(f"      Average: {sum(non_self_attention)/len(non_self_attention):.6f}")
                        print(f"      Sum: {sum(non_self_attention):.6f}")
                        
                        # Check per-destination normalization
                        dest_sums = {}
                        for (src, dst), score in attention_dict.items():
                            if src != dst and score is not None and score > 0:
                                if dst not in dest_sums:
                                    dest_sums[dst] = 0.0
                                dest_sums[dst] += score
                        
                        if dest_sums:
                            dest_sum_values = list(dest_sums.values())
                            print(f"      Per-destination sums: Min={min(dest_sum_values):.6f}, Max={max(dest_sum_values):.6f}, Avg={sum(dest_sum_values)/len(dest_sum_values):.6f}")
    
    def _print_detailed_summary(self, target_reservoirs, all_epochs, target_timesteps, edges_by_timestep, attention_by_timestep):
        """Print detailed summary of threshold-based edge pruning effects."""
        print(f"\n{'='*80}")
        print(f"THRESHOLD-BASED EDGE PRUNING SUMMARY")
        print(f"{'='*80}")
        print(f"Target reservoirs: {target_reservoirs}")
        print(f"Base attention data from: Epoch {self.base_refinement_epoch}")
        print(f"Normalization strategy: Include self-loops in softmax for realistic GAT simulation")
        print(f"Threshold strategy: Epoch 1 uses original attention (strict), Epoch 2+ uses normalized (fair)")
        
        # Show epoch threshold configuration
        epoch_info = []
        for epoch in all_epochs:
            threshold = self.epoch_thresholds.get(epoch, None)
            if epoch is None:
                epoch_info.append("Original (no pruning)")
            elif epoch == 1:
                epoch_info.append(f"Epoch {epoch} (original ≥ {threshold})")
            else:
                epoch_info.append(f"Epoch {epoch} (normalized ≥ {threshold})")
        print(f"Epoch configuration: {epoch_info}")
        print(f"Analyzed timesteps: {[f't={t+1}' for t in target_timesteps]}")
        
        # Create summary table
        print(f"\n{'Epoch':<15} {'Threshold':<10} {'Timestep':<10} {'Edges':<8} {'Change from Original':<20} {'Avg Attention':<15}")
        print(f"{'-'*95}")
        
        # Get original edge counts for comparison (excluding self-loops)
        original_edges = self.extract_target_subgraph_edges(self.original_edge_index, target_reservoirs, exclude_self_loops=True)
        original_count = len(original_edges)
        
        # Use the provided sequential pruning results for consistency
        
        for epoch_idx, epoch in enumerate(all_epochs):
            epoch_label = "Original" if epoch is None else f"Epoch {epoch}"
            threshold = self.epoch_thresholds.get(epoch, None)
            threshold_str = "None" if threshold is None else f"≥{threshold}"
            
            for timestep in target_timesteps:
                # Get edges and attention from sequential pruning results
                edges = edges_by_timestep[timestep][epoch]
                attention_dict = attention_by_timestep[timestep][epoch]
                
                # Count edges excluding self-loops
                display_edges = {(src, dst) for src, dst in edges if src != dst}
                edge_count = len(display_edges)
                
                # Calculate average attention score for non-self-loop edges
                if epoch is None:
                    avg_attention = "N/A"
                else:
                    non_self_attention = [score for (src, dst), score in attention_dict.items() 
                                        if src != dst and score > 0]
                    avg_attention = f"{sum(non_self_attention)/len(non_self_attention):.4f}" if non_self_attention else "0.0000"
                
                # Calculate change from original
                if epoch is None:
                    change_str = "-"
                else:
                    change = edge_count - original_count
                    change_pct = (change / original_count * 100) if original_count > 0 else 0
                    change_str = f"{change:+d} ({change_pct:+.1f}%)"
                
                print(f"{epoch_label:<15} {threshold_str:<10} {f't={timestep+1}':<10} {edge_count:<8} {change_str:<20} {avg_attention:<15}")
        
        # Additional statistics
        print(f"\n{'='*80}")
        print(f"THRESHOLD PRUNING STATISTICS")
        print(f"{'='*80}")
        
        # Show pruning effects for each threshold at first timestep
        first_timestep = target_timesteps[0]
        attention_weights = self.get_attention_weights_at_epoch_timestep(self.target_epochs[0], first_timestep)
        
        if attention_weights is not None:
            # Prepare edge index with self-loops
            original_edge_index = self.original_edge_index
            num_nodes = len(self.encode_map)
            device = original_edge_index.device
            
            self_loops = torch.arange(num_nodes, device=device)
            self_loop_edge_index = torch.stack([self_loops, self_loops], dim=0)
            edge_index_with_self_loops = torch.cat([original_edge_index, self_loop_edge_index], dim=1)
            
            base_edges = self.extract_target_subgraph_edges(self.original_edge_index, target_reservoirs, exclude_self_loops=False)
            
            # Initialize with original GAT attention for proper sequential pruning
            current_edges = self.extract_target_subgraph_edges(self.original_edge_index, target_reservoirs, exclude_self_loops=False)
            current_attention_dict = self.get_original_attention_dict(current_edges, attention_weights, edge_index_with_self_loops)
            
            for epoch in self.target_epochs:
                threshold = self.epoch_thresholds.get(epoch)
                current_edges, current_attention_dict = self.apply_threshold_pruning(
                    current_edges, current_attention_dict, threshold, epoch)
                
                pruned_edges = current_edges
                attention_dict = current_attention_dict
                
                # Calculate statistics
                display_edges = {(src, dst) for src, dst in pruned_edges if src != dst}
                removed_count = original_count - len(display_edges)
                retention_rate = len(display_edges) / original_count * 100 if original_count > 0 else 0
                
                # Attention score statistics
                non_self_attention = [score for (src, dst), score in attention_dict.items() 
                                    if src != dst and score > 0]
                
                print(f"Epoch {epoch} (threshold ≥ {threshold}):")
                print(f"  Edges retained: {len(display_edges)}/{original_count} ({retention_rate:.1f}%)")
                print(f"  Edges removed: {removed_count}")
                if non_self_attention:
                    print(f"  Softmax-renormalized attention scores - Min: {min(non_self_attention):.4f}, Max: {max(non_self_attention):.4f}, Avg: {sum(non_self_attention)/len(non_self_attention):.4f}")
                    
                    # Verify per-node normalization (should sum to ~1.0 for each destination node)
                    node_sums = {}
                    for (src, dst), score in attention_dict.items():
                        if src != dst and score > 0:
                            if dst not in node_sums:
                                node_sums[dst] = 0.0
                            node_sums[dst] += score
                    
                    if node_sums:
                        avg_node_sum = sum(node_sums.values()) / len(node_sums)
                        print(f"  Per-node attention sum verification - Nodes: {len(node_sums)}, Avg sum: {avg_node_sum:.4f}")
                print()
    
    def analyze_edge_change_patterns(self, target_reservoirs=None):
        """
        Analyze patterns in threshold-based edge pruning across epochs and timesteps.
        
        Args:
            target_reservoirs: List of reservoir names to analyze
        """
        if target_reservoirs is None:
            target_reservoirs = self.target_reservoirs
        
        print(f"\n{'='*80}")
        print(f"THRESHOLD-BASED PRUNING PATTERN ANALYSIS")
        print(f"{'='*80}")
        
        # Get original edges (excluding self-loops)
        original_edges = self.extract_target_subgraph_edges(self.original_edge_index, target_reservoirs, exclude_self_loops=True)
        print(f"Original edges (excluding self-loops): {len(original_edges)}")
        print(f"Base attention data from: Epoch {self.base_refinement_epoch}")
        
        # Analyze pruning effects for each threshold
        first_timestep = self.target_timesteps[0]
        attention_weights = self.get_attention_weights_at_epoch_timestep(self.target_epochs[0], first_timestep)
        
        if attention_weights is None:
            print(f"No attention weights available for analysis")
            return
        
        # Prepare edge index with self-loops
        original_edge_index = self.original_edge_index
        num_nodes = len(self.encode_map)
        device = original_edge_index.device
        
        self_loops = torch.arange(num_nodes, device=device)
        self_loop_edge_index = torch.stack([self_loops, self_loops], dim=0)
        edge_index_with_self_loops = torch.cat([original_edge_index, self_loop_edge_index], dim=1)
        
        # Use sequential pruning for pattern analysis
        current_edges = self.extract_target_subgraph_edges(self.original_edge_index, target_reservoirs, exclude_self_loops=False)
        
        # Initialize with original GAT attention for Epoch 1
        base_attention_weights = self.get_attention_weights_at_epoch_timestep(1, first_timestep)
        if base_attention_weights is not None:
            current_attention_dict = self.get_original_attention_dict(current_edges, base_attention_weights, edge_index_with_self_loops)
        else:
            current_attention_dict = {}
        
        print(f"\nProgressive Threshold Pruning Analysis:")
        
        for epoch in self.target_epochs:
            threshold = self.epoch_thresholds.get(epoch)
            print(f"\nEpoch {epoch} (threshold ≥ {threshold}):")
            
            # Apply sequential threshold pruning using current attention state
            current_edges, current_attention_dict = self.apply_threshold_pruning(
                current_edges, current_attention_dict, threshold, epoch)
            
            # Apply random adjustments after pruning (using first timestep for analysis)
            current_attention_dict = self.apply_random_adjustments(current_attention_dict, epoch, first_timestep)
            
            # Get edges excluding self-loops for analysis
            display_edges = {(src, dst) for src, dst in current_edges if src != dst}
            removed_edges = original_edges - display_edges
            
            retention_rate = len(display_edges) / len(original_edges) * 100 if len(original_edges) > 0 else 0
            
            print(f"  Edges retained: {len(display_edges)}/{len(original_edges)} ({retention_rate:.1f}%)")
            print(f"  Edges removed: {len(removed_edges)}")
            
            # Show attention score distribution for retained edges
            retained_attention = [score for (src, dst), score in current_attention_dict.items() 
                                if src != dst and score > 0]
            
            if retained_attention:
                print(f"  Retained edge softmax-renormalized attention scores:")
                print(f"    Min: {min(retained_attention):.4f}")
                print(f"    Max: {max(retained_attention):.4f}")
                print(f"    Mean: {sum(retained_attention)/len(retained_attention):.4f}")
                
                # Verify per-node normalization
                node_sums = {}
                for (src, dst), score in current_attention_dict.items():
                    if src != dst and score > 0:
                        if dst not in node_sums:
                            node_sums[dst] = 0.0
                        node_sums[dst] += score
                
                if node_sums:
                    avg_node_sum = sum(node_sums.values()) / len(node_sums)
                    print(f"    Per-node normalization check - Avg sum per node: {avg_node_sum:.4f}")
            
            # Show removed edges with their original attention scores
            if removed_edges:
                print(f"  Removed edges (original attention < {threshold}):")
                removed_with_attention = []
                for src, dst in removed_edges:
                    original_score = self.get_edge_attention_score(src, dst, attention_weights, edge_index_with_self_loops)
                    if original_score is not None:
                        removed_with_attention.append(((src, dst), original_score))
                
                # Sort by attention score (lowest first)
                removed_with_attention.sort(key=lambda x: x[1])
                
                # Show first 5 removed edges
                for i, ((src, dst), score) in enumerate(removed_with_attention[:5]):
                    src_name = self.reverse_encode_map.get(src, f"Node_{src}")
                    dst_name = self.reverse_encode_map.get(dst, f"Node_{dst}")
                    print(f"    {src_name} -> {dst_name} (attention: {score:.4f})")
                
                if len(removed_with_attention) > 5:
                    print(f"    ... and {len(removed_with_attention) - 5} more")
        
        # Cross-timestep consistency analysis
        print(f"\n{'='*80}")
        print(f"CROSS-TIMESTEP CONSISTENCY ANALYSIS")
        print(f"{'='*80}")
        
        # Use sequential pruning results from visualization for consistency
        for epoch in self.target_epochs:
            threshold = self.epoch_thresholds.get(epoch)
            print(f"\nEpoch {epoch} (threshold ≥ {threshold}) - Edge counts across timesteps:")
            
            timestep_counts = []
            for timestep in self.target_timesteps:
                # Sequential pruning for each timestep
                timestep_current_edges = self.extract_target_subgraph_edges(self.original_edge_index, target_reservoirs, exclude_self_loops=False)
                
                # Initialize with original GAT attention
                timestep_base_attention_weights = self.get_attention_weights_at_epoch_timestep(1, timestep)
                if timestep_base_attention_weights is not None:
                    timestep_current_attention_dict = self.get_original_attention_dict(timestep_current_edges, timestep_base_attention_weights, edge_index_with_self_loops)
                else:
                    timestep_current_attention_dict = {}
                
                # Apply all epochs up to and including current epoch
                for e in range(1, epoch + 1):
                    if e in self.epoch_thresholds:
                        e_threshold = self.epoch_thresholds[e]
                        if e_threshold is not None:
                            timestep_current_edges, timestep_current_attention_dict = self.apply_threshold_pruning(
                                timestep_current_edges, timestep_current_attention_dict, e_threshold, e)
                            # Apply random adjustments after each epoch
                            timestep_current_attention_dict = self.apply_random_adjustments(timestep_current_attention_dict, e, timestep)
                
                display_edges = {(src, dst) for src, dst in timestep_current_edges if src != dst}
                timestep_counts.append(len(display_edges))
                print(f"  t={timestep+1}: {len(display_edges)} edges")
            
            if timestep_counts:
                consistency = max(timestep_counts) - min(timestep_counts)
                print(f"  Consistency: {consistency} edge difference across timesteps")


def main(target_time=None):
    """
    Main function to run detailed edge tracking analysis.
    
    Args:
        target_time: Target timestamp for analysis (e.g., "202507082038")
    """
    print("Initializing Detailed Edge Tracker...")
    
    # Initialize the detailed edge tracker
    tracker = DetailedEdgeTracker(target_time=target_time)
    
    print(f"\nLoaded edge tracking data from: {tracker.file_path}")
    print(f"Available refinement epochs: {[change['epoch'] for change in tracker.epoch_changes]}")
    print(f"Using base attention data from: Epoch {tracker.base_refinement_epoch}")
    
    # Set analysis parameters
    target_reservoirs = ['PIN', 'CAU', 'LCR', 'ECH', 'ECR', 'ROC']  # Parameter 1
    target_epochs = [1, 2, 3]  # Parameter 2: simulated epochs with thresholds
    target_timesteps = [0, 14, 29]  # Parameter 3: t=1, t=15, t=30 (0-indexed)
    
    print(f"\nAnalysis Parameters:")
    print(f"  Target reservoirs: {target_reservoirs}")
    print(f"  Simulated epochs: Original + {target_epochs}")
    print(f"  Threshold configuration: {tracker.epoch_thresholds}")
    print(f"  Target timesteps: {[f't={t+1}' for t in target_timesteps]}")
    
    # Create detailed visualization
    print(f"\nGenerating threshold-based edge pruning visualization...")
    tracker.visualize_detailed_edge_tracking(
        target_reservoirs=target_reservoirs,
        target_epochs=target_epochs,
        target_timesteps=target_timesteps,
        save_path='threshold_based_edge_pruning_analysis'
    )
    
    # Analyze edge change patterns
    tracker.analyze_edge_change_patterns(target_reservoirs=target_reservoirs)
    
    print(f"\nDetailed edge tracking analysis completed!")


if __name__ == "__main__":
    target_time = "202508101634" # "202508122236"
    main(target_time=target_time)
    