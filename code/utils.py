import numpy as np
import torch
from scipy.sparse import csr_matrix
from collections import defaultdict

def _to_numpy(x):
    """Convert torch tensor to numpy array if needed."""
    return x.cpu().numpy() if hasattr(x, 'cpu') else np.array(x)

def compute_jaccard_similarity(data, edge_index=None):
    """
    Compute Jaccard similarity for connected edges (or given edge_index pairs).
    Returns: dict with (u, v) tuple as key and Jaccard similarity as value.
    """
    print("Computing Jaccard similarity...")
    if edge_index is None:
        edge_index = data.edge_index
    num_nodes = data.num_nodes

    edge_array = _to_numpy(edge_index)
    adj = csr_matrix((np.ones(edge_array.shape[1]), (edge_array[0], edge_array[1])),
                     shape=(num_nodes, num_nodes))

    pairs = set(map(tuple, edge_array.T))
    jaccard = {}
    for u, v in pairs:
        neighbors_u = set(adj[u].indices)
        neighbors_u.add(u)
        neighbors_v = set(adj[v].indices)
        neighbors_v.add(v)
        intersection = len(neighbors_u & neighbors_v)
        union = len(neighbors_u | neighbors_v)
        if union > 0:
            jaccard[(u, v)] = intersection / union

    print("Jaccard similarity computation completed")

    return jaccard

def compute_location_similarity(data, edge_index=None):
    """
    Compute location similarity for connected edges (or given edge_index pairs) using haversine distance.
    Returns: dict with (u, v) tuple as key and location similarity as value.
    """
    print("Computing location similarity...")
    if edge_index is None:
        edge_index = data.edge_index
    
    edge_array = _to_numpy(edge_index)
    coords = _to_numpy(data.rad_x)
    
    # Haversine distance calculation function
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius (km)
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance = R * c
        
        # # Convert distance to similarity (farther distance means lower similarity)
        # # Use exponential decay function so similarity is 1 when distance is 0, approaches 0 as distance increases
        # similarity = np.exp(-distance/1000)
        return distance
    
    # Calculate similarity for each edge
    distances = {}
    for i, j in zip(edge_array[0], edge_array[1]):
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[j]
        dis = haversine_distance(lat1, lon1, lat2, lon2)
        distances[(i, j)] = dis
        distances[(j, i)] = dis  # Store both directions for undirected graph

    # Calculate similarity (farther distance means lower similarity)
    max_dist = max(distances.values())
    similarities = {k: 1 - v/max_dist for k, v in distances.items()}
    
    print("Location similarity computation completed")

    return similarities

def compute_adaptive_similarity(data, structure_similarity, location_similarity, pred_labels=None):
    """
    Adaptive similarity calculation function
    
    Args:
        data: Data object (requires edge_index and num_nodes)
        structure_similarity: Pre-calculated structural similarity dictionary {(u,v): similarity_score}
        location_similarity: Pre-calculated location-based similarity dictionary {(u,v): similarity_score}
        pred_labels: Currently predicted labels (default: None, uses data.y)
    
    Returns:
        adaptive_sim: Adaptive similarity dictionary {(u,v): similarity_score}
        avg_alpha: Average alpha value
        dev_alpha: Standard deviation of alpha values
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    edge_array = _to_numpy(edge_index)
    
    # Create symmetric matrix for undirected graph (using CSR format)
    adj = csr_matrix((np.ones(edge_array.shape[1]), (edge_array[0], edge_array[1])),
                     shape=(num_nodes, num_nodes))
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)
    
    # Prepare labels
    if pred_labels is None:
        labels = getattr(data, 'y', None)
        if labels is None:
            raise ValueError("pred_labels or data.y must be provided")
        labels = _to_numpy(labels)
    else:
        labels = _to_numpy(pred_labels)
    
    # Select only valid labels
    valid_mask = (labels != -1) & ~np.isnan(labels)
    unique_labels = np.unique(labels[valid_mask])
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    
    # Calculate alpha values for all nodes (to solve convergence issues)
    nodes_to_compute = np.arange(num_nodes)
    
    # Calculate alpha values per node
    alpha = np.full(num_nodes, 0.5)

    for node in nodes_to_compute:
        neighbors = adj[node].indices
        if len(neighbors) == 0:
            alpha[node] = 0
            continue
            
        neigh_labels = labels[neighbors]
        valid_neigh = neigh_labels[neigh_labels != -1]
            
        # Calculate entropy
        label_counts = np.bincount([label_to_idx[l] for l in valid_neigh])
        label_counts = label_counts[label_counts > 0]  # Use only non-zero label counts
        
        # Calculate maximum entropy (when all labels are evenly distributed)
        n_unique_labels = len(label_counts)
        max_entropy = np.log(n_unique_labels) if n_unique_labels > 1 else 1.0
        # max_entropy = np.log(len(unique_labels)) if len(unique_labels) > 1 else 1.0 # Use total number of labels for comparison
        
        # Calculate actual entropy
        probs = label_counts / len(valid_neigh)
        entropy = -np.sum(probs * np.log(probs))
        
        # Normalized entropy (value between 0~1)
        normalized_entropy = entropy / max_entropy if max_entropy != 0 else 0
        normalized_entropy = np.clip(normalized_entropy, 0, 1)
        
        # Calculate alpha value: represents structural reliability of the node
        # - If all neighbors have same label, alpha=1 (structure is very clear)
        # - If neighbors have diverse labels, alpha approaches 0 (structure is ambiguous)
        # This value is used as reliability when the node transmits structural information to other nodes
        alpha[node] = 1 - normalized_entropy
    
    # Calculate adaptive similarity
    adaptive_sim = {}
    edges_to_compute = np.arange(edge_array.shape[1])  # Calculate for all edges
    
    for idx in edges_to_compute:
        i, j = edge_array[0, idx], edge_array[1, idx]
        
        # Use pre-calculated similarities
        sim_structure = structure_similarity[(i, j)]
        sim_location = location_similarity[(i, j)]

        # Apply adaptive weights
        # Use alpha[j]: adjust influence of structural similarity based on structural reliability of j(sender)
        # - When j's neighbors have consistent labels (high alpha[j]) → trust structural similarity more
        # - When j's neighbors have diverse labels (low alpha[j]) → rely more on location similarity
        adaptive_sim[(i, j)] = alpha[j] * sim_structure + (1 - alpha[j]) * sim_location
        
    
    # Calculate statistics and output distribution for debugging
    computed_alphas = alpha  # Use alpha values from all nodes
    avg_alpha = float(np.mean(computed_alphas))
    dev_alpha = float(np.std(computed_alphas))
    
    return adaptive_sim, alpha, avg_alpha, dev_alpha

def compute_fixed_alpha_similarity(data, fixed_alpha, structure_similarity, location_similarity, pred_labels=None):
    """
    Adaptive similarity calculation function
    
    Args:
        data: Data object (requires edge_index and num_nodes)
        structure_similarity: Pre-calculated structural similarity dictionary {(u,v): similarity_score}
        location_similarity: Pre-calculated location-based similarity dictionary {(u,v): similarity_score}
        pred_labels: Currently predicted labels (default: None, uses data.y)
    
    Returns:
        adaptive_sim: Adaptive similarity dictionary {(u,v): similarity_score}
        avg_alpha: Average alpha value
        dev_alpha: Standard deviation of alpha values
    """
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    edge_array = _to_numpy(edge_index)
    
    # Create symmetric matrix for undirected graph (using CSR format)
    adj = csr_matrix((np.ones(edge_array.shape[1]), (edge_array[0], edge_array[1])),
                     shape=(num_nodes, num_nodes))
    adj = adj + adj.T
    adj.data = np.ones_like(adj.data)
    
    # Prepare labels
    if pred_labels is None:
        labels = getattr(data, 'y', None)
        if labels is None:
            raise ValueError("pred_labels or data.y must be provided")
        labels = _to_numpy(labels)
    else:
        labels = _to_numpy(pred_labels)
    
    # Select only valid labels
    valid_mask = (labels != -1) & ~np.isnan(labels)
    unique_labels = np.unique(labels[valid_mask])
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    
    # Identify nodes that need propagation (nodes in edges with different labels)
    nodes_to_compute = set()
    diff_labels = labels[edge_array[0]] != labels[edge_array[1]]
    if diff_labels.any():
        nodes_to_compute.update(edge_array[0][diff_labels])
        nodes_to_compute.update(edge_array[1][diff_labels])
        for node in nodes_to_compute.copy():
            nodes_to_compute.update(adj[node].indices)
    
    nodes_to_compute = np.array(list(nodes_to_compute))
    
    # Calculate alpha values per node
    alpha = np.full(num_nodes, fixed_alpha)
    
    # Calculate adaptive similarity
    adaptive_sim = {}
    edges_to_compute = np.arange(edge_array.shape[1])  # Calculate for all edges
    
    for idx in edges_to_compute:
        i, j = edge_array[0, idx], edge_array[1, idx]
        
        # Use pre-calculated similarities
        sim_structure = structure_similarity[(i, j)]
        sim_location = location_similarity[(i, j)]

        # Apply adaptive weights
        # Use alpha[j]: adjust influence of structural similarity based on structural reliability of j(sender)
        # - When j's neighbors have consistent labels (high alpha[j]) → trust structural similarity more
        # - When j's neighbors have diverse labels (low alpha[j]) → rely more on location similarity
        adaptive_sim[(i, j)] = alpha[j] * sim_structure + (1 - alpha[j]) * sim_location
    
    # Calculate statistics and output distribution for debugging
    computed_alphas = alpha  # Use alpha values from all nodes
    avg_alpha = float(np.mean(computed_alphas))
    dev_alpha = float(np.std(computed_alphas))
    
    return adaptive_sim, avg_alpha, dev_alpha