import torch
import networkx as nx
from collections import defaultdict
import random
from utils import compute_adaptive_similarity

def adaptive_label_propagation(data, labels, structure_similarity=None, location_similarity=None, max_iter=1000, verbose=False, order='random'):
    """
    Performs Adaptive Label Propagation algorithm based on asyn_lpa_communities.
    Updates edge weights by recalculating adaptive similarity at each iteration.
    
    Args:
        data: Data object (requires edge_index and num_nodes)
        labels: Node labels (used as initial labels)
        structure_similarity: Structural similarity dictionary {(u,v): similarity_score}
        location_similarity: Location similarity dictionary {(u,v): similarity_score}
        max_iter: Maximum number of iterations (default: 1000)
        verbose: Whether to print detailed output (default: False)
        order: Node update order (default: 'random')
    Returns:
        pred_labels: Predicted labels
        last_similarity: Final similarity dictionary
        iter_info: Iteration process information
    """
    if structure_similarity is None or location_similarity is None:
        raise ValueError("structure_similarity and location_similarity must be provided.")

    n = labels.size(0)
    alpha_info = []
    iter_info = []
    last_similarity = None
    
    # Create NetworkX graph (basic structure)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # Set node position information (using coordinates from data.x)
    pos_dict = {}
    if hasattr(data, 'x') and data.x is not None:
        coordinates = data.x.cpu().numpy() if hasattr(data.x, 'cpu') else data.x
        for i in range(n):
            pos_dict[i] = (coordinates[i, 0], coordinates[i, 1])  # Set in (lon, lat) order
    nx.set_node_attributes(G, pos_dict, 'pos')
    
    # Add basic edges (weights will be updated at each iteration)
    edge_array = data.edge_index.cpu().numpy() if hasattr(data.edge_index, 'cpu') else data.edge_index
    edges = [(int(edge_array[0, i]), int(edge_array[1, i])) for i in range(edge_array.shape[1])]
    G.add_edges_from(edges)
    
    # Set initial labels (unique label for each node)
    node_labels = {n: i for i, n in enumerate(G)}
    
    if verbose:
        print(f"Graph creation complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    for iter_idx in range(max_iter):
        # Create torch tensor with current label state
        current_labels = torch.tensor([node_labels[i] for i in range(n)], dtype=labels.dtype)
        
        # Calculate adaptive similarity
        adaptive_similarity, alpha, avg_alpha, dev_alpha = compute_adaptive_similarity(
            data=data,
            structure_similarity=structure_similarity,
            location_similarity=location_similarity,
            pred_labels=current_labels
        )
        last_similarity = adaptive_similarity
        alpha_info.append(alpha)
        
        # Update graph weights
        for u, v in G.edges():
            if (u, v) in adaptive_similarity:
                G[u][v]['weight'] = adaptive_similarity[(u, v)]
            elif (v, u) in adaptive_similarity:
                G[v][u]['weight'] = adaptive_similarity[(v, u)]
            else:
                G[u][v]['weight'] = 0.0
                G[v][u]['weight'] = 0.0
                
        # Apply asyn_lpa_communities logic (one full node update)
        cont = False
        nodes = list(G)
        if order == 'random':
            random.shuffle(nodes)
        elif order == 'original':
            pass
        elif order == 'reverse':
            nodes.reverse()
        
        label_changes = 0

        all_nodes = list(range(len(G.nodes())))
        
        for node in nodes:
            if not G[node]:  # Skip nodes with no neighbors
                continue

            # Calculate label frequency of neighbors (considering weights)
            label_freq = defaultdict(float)
            for _, v, wt in G.edges(node, data='weight', default=0.0):
                if wt > 0:  # Only consider positive weights
                    label_freq[node_labels[v]] += wt
            
            if not label_freq:  # If no valid neighbors
                continue
                
            # Find labels with maximum frequency
            max_freq = max(label_freq.values())
            best_labels = [
                label for label, freq in label_freq.items() if freq == max_freq
            ]
            
            # Update if current label is not one of the maximum frequency labels
            if node_labels[node] not in best_labels:
                old_label = node_labels[node]
                node_labels[node] = random.choice(best_labels)
                if old_label != node_labels[node]:
                    label_changes += 1
                    cont = True
        
        
        # Collect current state information
        unique_labels = len(set(node_labels.values()))
        label_change_rate = label_changes / n
        
        iter_info.append(f"Iter {iter_idx+1}: average_alpha={avg_alpha:.6f}, dev_alpha={dev_alpha:.6f}, #labels={unique_labels}, changes={label_changes}({label_change_rate:.4f})")
        if verbose:
            print(iter_info[-1])
        
        # Improved convergence conditions
        convergence_conditions = []
        
        # 1. Converge immediately if no label changes
        if not cont or label_changes == 0:
            convergence_conditions.append("no_label_changes")
        
        # 2. Converge if label change rate is very low (less than 1.5%)
        elif label_change_rate < 0.015:
            convergence_conditions.append("low_change_rate")
        
        # Stop if any convergence condition is met
        if convergence_conditions:
            reason = ", ".join(convergence_conditions)
            iter_info.append(f"Converged at iteration {iter_idx+1} (reason: {reason})")
            if verbose:
                print(iter_info[-1])
            break
            
    # Convert final result to torch tensor
    pred_labels = torch.tensor([node_labels[i] for i in range(n)], dtype=labels.dtype)
    
    # Remap labels to start from 0
    unique_labels = torch.unique(pred_labels)
    label_mapping = {old_label.item(): new_label for new_label, old_label in enumerate(unique_labels)}
    for i in range(n):
        pred_labels[i] = label_mapping[pred_labels[i].item()]
    
    return pred_labels, last_similarity, iter_info, alpha_info
