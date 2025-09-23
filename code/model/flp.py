import torch
import networkx as nx
from model.label_propagation import custom_asyn_lpa_communities
import time

def fixed_alpha_label_propagation(data, labels, fixed_alpha, structure_similarity=None, location_similarity=None, verbose=False):
    """
    Performs Fixed Alpha Label Propagation algorithm using NetworkX library.
    Combines structural similarity and location similarity with fixed_alpha ratio as weights.
    
    Args:
        data: Graph data object
        labels: Node labels (not used - NetworkX automatically detects communities)
        fixed_alpha: Mixing ratio of structural similarity and location similarity (0~1)
        structure_similarity: Structural similarity dictionary
        location_similarity: Location similarity dictionary
        max_iter: Parameter for compatibility (NetworkX uses built-in convergence conditions)
        verbose: Whether to print detailed output (default: False)
    
    Returns:
        pred_labels: Predicted labels
        mixed_similarity: Mixed similarity dictionary
        iter_info: Iteration process information
    """
    if structure_similarity is None or location_similarity is None:
        raise ValueError("structure_similarity and location_similarity must be provided.")

    n = labels.size(0)
    
    # Mix structural similarity and location similarity with fixed_alpha ratio
    mixed_similarity = {}
    all_edges = set(structure_similarity.keys()) | set(location_similarity.keys())
    
    for edge in all_edges:
        struct_sim = structure_similarity.get(edge, 0.0)
        loc_sim = location_similarity.get(edge, 0.0)
        # fixed_alpha: structural similarity weight, (1-fixed_alpha): location similarity weight
        mixed_sim = fixed_alpha * struct_sim + (1 - fixed_alpha) * loc_sim
        if mixed_sim > 0:  # Use only positive similarities
            mixed_similarity[edge] = mixed_sim

    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(n))
    
    # Add weighted edges
    weighted_edges = []
    for (u, v), weight in mixed_similarity.items():
        weighted_edges.append((u, v, weight))
    
    G.add_weighted_edges_from(weighted_edges)
    
    if verbose:
        print(f"Graph creation complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} weighted edges")
    
    # Perform NetworkX asynchronous label propagation
    try:
        random_seed = int(time.time() * 1000) % 2**32
        communities, it = custom_asyn_lpa_communities(G, weight='weight', seed=random_seed)
        communities_list = list(communities)
        
        # Convert communities to labels
        pred_labels = torch.zeros(n, dtype=labels.dtype)
        for label_idx, comm in enumerate(communities_list):
            for node in comm:
                pred_labels[node] = label_idx
        
        num_communities = len(communities_list)
        
        if verbose:
            print("NetworkX Fixed Alpha Label Propagation completed")
            print(f"Number of communities found: {num_communities}")
            print(f"Fixed Alpha: {fixed_alpha}")
            print(f"Number of weighted edges used: {len(mixed_similarity)}")
            print(f"Size of each community: {[len(comm) for comm in communities_list]}")
        
    except Exception as e:
        if verbose:
            print(f"Error during NetworkX label propagation execution: {e}")
        # Fallback: assign all nodes to a single community
        pred_labels = torch.zeros(n, dtype=labels.dtype)
        print(f"Assigned to single community due to error: {e}")
    
    return pred_labels, mixed_similarity, it
