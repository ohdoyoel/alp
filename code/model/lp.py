import torch
import networkx as nx
from model.label_propagation import custom_asyn_lpa_communities
import time

def label_propagation(edge_index, labels, verbose=False):
    """
    Performs Label Propagation algorithm using NetworkX library.
    
    Args:
        edge_index: Edge index tensor of size (2, E)
        labels: Node labels (not used - NetworkX automatically detects communities)
        max_iter: Parameter for compatibility (NetworkX uses built-in convergence conditions)
        verbose: Whether to print detailed output (default: False)
        seed: Random seed (default: None, different results each time)
    
    Returns:
        pred_labels: Predicted labels
        iter_info: Iteration process information
    """
    n = labels.size(0)
    
    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # Convert edge_index to NetworkX edges
    edge_array = edge_index.cpu().numpy() if hasattr(edge_index, 'cpu') else edge_index
    edges = [(int(edge_array[0, i]), int(edge_array[1, i])) for i in range(edge_array.shape[1])]
    G.add_edges_from(edges)
    
    if verbose:
        print(f"Graph creation complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Perform NetworkX label propagation
    try:
        random_seed = int(time.time() * 1000) % 2**32
        communities, it = custom_asyn_lpa_communities(G, seed=random_seed)
        communities_list = list(communities)
        
        # Convert communities to labels
        pred_labels = torch.zeros(n, dtype=labels.dtype)
        for label_idx, comm in enumerate(communities_list):
            for node in comm:
                pred_labels[node] = label_idx
        
        num_communities = len(communities_list)
        
        if verbose:
            print("NetworkX Label Propagation completed")
            print(f"Number of communities found: {num_communities}")
            print(f"Size of each community: {[len(comm) for comm in communities_list]}")
        
    except Exception as e:
        if verbose:
            print(f"Error during NetworkX label propagation execution: {e}")
        # Fallback: assign all nodes to a single community
        pred_labels = torch.zeros(n, dtype=labels.dtype)
        print(f"Assigned to single community due to error: {e}")
    
    return pred_labels, it