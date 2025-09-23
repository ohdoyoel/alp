import torch
import numpy as np
import os
import pandas as pd

def load_brightkite(sample_size=None):
    # Load Brightkite_edges.txt
    base_dir = os.path.dirname(os.path.abspath(__file__))
    edge_path = os.path.join(base_dir, '..', 'datasets', 'Brightkite_edges.txt')
    with open(edge_path, 'r') as f:
        edges = [tuple(map(int, line.strip().split())) for line in f]
    edges = np.array(edges).T
    
    # Load checkin data and extract most recent visit locations
    checkin_path = os.path.join(base_dir, '..', 'datasets', 'Brightkite_totalCheckins.txt')
    df = pd.read_csv(checkin_path, sep='\t', header=None, names=['user', 'time', 'lat', 'lon', 'loc'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    
    # Exclude abnormal location data
    # Exclude NaN values
    df = df.dropna(subset=['lat', 'lon'])
    # Latitude: -90° ~ 90°, Longitude: -180° ~ 180°
    df = df[
        (df['lat'] != 0) | (df['lon'] != 0)  # Exclude (0, 0) location
    ]
    # Filter abnormal location data
    df = df[
        (df['lat'] >= -90) & (df['lat'] <= 90) &  # Check latitude range
        (df['lon'] >= -180) & (df['lon'] <= 180)  # Check longitude range
    ]
    
    recent = df.groupby('user').last().reset_index()
    
    # Select only nodes with recent visit locations
    valid_nodes = set(recent['user'].astype(int))
    
    # Apply sampling
    if sample_size is not None and sample_size < len(valid_nodes):
        # Set random seed for reproducibility
        np.random.seed(42)
        # Randomly select sample_size number of nodes
        valid_nodes = set(np.random.choice(list(valid_nodes), size=sample_size, replace=False))
    
    # Filter edges - keep only edges where both nodes are valid
    valid_edges = [(u, v) for u, v in zip(edges[0], edges[1]) 
                   if u in valid_nodes and v in valid_nodes]
    if not valid_edges:
        raise ValueError("No valid edges found after filtering")
    
    # Remap node IDs
    node_map = {old: new for new, old in enumerate(sorted(valid_nodes))}
    edges_remapped = np.array([[node_map[u], node_map[v]] for u, v in valid_edges]).T
    num_nodes = len(node_map)
    
    # Create feature matrix
    features = np.zeros((num_nodes, 2), dtype=np.float32)
    rad_features = np.zeros((num_nodes, 2), dtype=np.float32)
    longitude = np.zeros(num_nodes, dtype=np.float32)
    latitude = np.zeros(num_nodes, dtype=np.float32)
    
    for _, row in recent.iterrows():
        old_id = int(row['user'])
        if old_id in node_map:
            new_id = node_map[old_id]
            features[new_id] = [row['lat'], row['lon']]
            rad_features[new_id] = [np.radians(row['lat']), np.radians(row['lon'])]
            longitude[new_id] = row['lon']
            latitude[new_id] = row['lat']
    
    # Create Data object
    data = type('Data', (), {})()
    data.edge_index = torch.tensor(edges_remapped, dtype=torch.long)
    data.num_nodes = num_nodes
    data.y = torch.zeros(num_nodes, dtype=torch.long)
    data.x = torch.from_numpy(features)
    data.rad_x = torch.from_numpy(rad_features)
    data.longitude = torch.from_numpy(longitude)
    data.latitude = torch.from_numpy(latitude)

    # Also save node ID mapping information
    data.node_map = node_map
    
    # Add average degree
    data.avg_degree = 2 * edges_remapped.shape[1] / num_nodes if num_nodes > 0 else 0
    
    return data, 1

def load_gowalla(sample_size=None):
    # Load Gowalla_edges.txt
    base_dir = os.path.dirname(os.path.abspath(__file__))
    edge_path = os.path.join(base_dir, '..', 'datasets', 'Gowalla_edges.txt')
    with open(edge_path, 'r') as f:
        edges = [tuple(map(int, line.strip().split())) for line in f]
    edges = np.array(edges).T
    
    # Load checkin data and extract most recent visit locations
    checkin_path = os.path.join(base_dir, '..', 'datasets', 'Gowalla_totalCheckins.txt')
    df = pd.read_csv(checkin_path, sep='\t', header=None, names=['user', 'time', 'lat', 'lon', 'loc'])
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    
    # Exclude abnormal location data
    # Exclude NaN values
    df = df.dropna(subset=['lat', 'lon'])
    
    # Latitude: -90° ~ 90°, Longitude: -180° ~ 180°
    df = df[
        (df['lat'] != 0) | (df['lon'] != 0)  # Exclude (0, 0) location
    ]
    # Filter abnormal location data
    df = df[
        (df['lat'] >= -90) & (df['lat'] <= 90) &  # Check latitude range
        (df['lon'] >= -180) & (df['lon'] <= 180)  # Check longitude range
    ]
    
    recent = df.groupby('user').last().reset_index()
    
    # Select only nodes with recent visit locations
    valid_nodes = set(recent['user'].astype(int))
    
    # Apply sampling
    if sample_size is not None and sample_size < len(valid_nodes):
        # Set random seed for reproducibility
        np.random.seed(42)
        # Randomly select sample_size number of nodes
        valid_nodes = set(np.random.choice(list(valid_nodes), size=sample_size, replace=False))
    
    # Filter edges - keep only edges where both nodes are valid
    valid_edges = [(u, v) for u, v in zip(edges[0], edges[1]) 
                   if u in valid_nodes and v in valid_nodes]
    if not valid_edges:
        raise ValueError("No valid edges found after filtering")
    
    # Remap node IDs
    node_map = {old: new for new, old in enumerate(sorted(valid_nodes))}
    edges_remapped = np.array([[node_map[u], node_map[v]] for u, v in valid_edges]).T
    num_nodes = len(node_map)
    
    # Create feature matrix
    features = np.zeros((num_nodes, 2), dtype=np.float32)
    rad_features = np.zeros((num_nodes, 2), dtype=np.float32)
    longitude = np.zeros(num_nodes, dtype=np.float32)
    latitude = np.zeros(num_nodes, dtype=np.float32)
    
    for _, row in recent.iterrows():
        old_id = int(row['user'])
        if old_id in node_map:
            new_id = node_map[old_id]
            features[new_id] = [row['lat'], row['lon']]
            rad_features[new_id] = [np.radians(row['lat']), np.radians(row['lon'])]
            longitude[new_id] = row['lon']
            latitude[new_id] = row['lat']
    
    # Create Data object
    data = type('Data', (), {})()
    data.edge_index = torch.tensor(edges_remapped, dtype=torch.long)
    data.num_nodes = num_nodes
    data.y = torch.zeros(num_nodes, dtype=torch.long)
    data.x = torch.from_numpy(features)
    data.rad_x = torch.from_numpy(rad_features)
    data.longitude = torch.from_numpy(longitude)
    data.latitude = torch.from_numpy(latitude)
    
    # Also save node ID mapping information
    data.node_map = node_map
    
    # Add average degree
    data.avg_degree = 2 * edges_remapped.shape[1] / num_nodes if num_nodes > 0 else 0
    
    return data, 1

def load_custom(sample_size=None):
    """
    Function to load custom dataset
    
    Args:
        sample_size: Number of nodes to sample (use all if None)
    
    Returns:
        data: PyTorch Geometric data object
        num_classes: Number of classes (default 1)
    """
    # Set base directory path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load custom_edges.txt
    edge_path = os.path.join(base_dir, '..', 'datasets', 'custom', 'custom_edges.txt')
    with open(edge_path, 'r') as f:
        edges = [tuple(map(int, line.strip().split())) for line in f]
    edges = np.array(edges).T
    
    # Load custom_totalCheckins.txt and extract checkin information
    checkin_path = os.path.join(base_dir, '..', 'datasets', 'custom', 'custom_totalCheckins.txt')
    checkin_data = []
    with open(checkin_path, 'r') as f:
        for line in f:
            if line.strip():  # Ignore empty lines
                parts = line.strip().split()
                if len(parts) >= 3:  # Must have at least user, x, y coordinates
                    user_id = int(parts[0])
                    x_coord = float(parts[1])
                    y_coord = float(parts[2])
                    checkin_data.append({'user': user_id, 'lat': y_coord, 'lon': x_coord})
    
    # Convert to DataFrame
    df = pd.DataFrame(checkin_data)
    
    # Skip excluding abnormal location data (custom data uses small coordinate system)
    
    # Select only nodes with checkin information
    valid_nodes = set(df['user'].astype(int))
    
    # Apply sampling
    if sample_size is not None and sample_size < len(valid_nodes):
        # Set random seed for reproducibility
        np.random.seed(42)
        # Randomly select sample_size number of nodes
        valid_nodes = set(np.random.choice(list(valid_nodes), size=sample_size, replace=False))
    
    # Filter edges - keep only edges where both nodes are valid
    valid_edges = [(u, v) for u, v in zip(edges[0], edges[1]) 
                   if u in valid_nodes and v in valid_nodes]
    if not valid_edges:
        raise ValueError("No valid edges found after filtering")
    
    # Remap node IDs
    node_map = {old: new for new, old in enumerate(sorted(valid_nodes))}
    
    # Create bidirectional edges (add both u->v and v->u)
    bidirectional_edges = []
    for u, v in valid_edges:
        bidirectional_edges.append([node_map[u], node_map[v]])
        bidirectional_edges.append([node_map[v], node_map[u]])  # Add reverse edge
    edges_remapped = np.array(bidirectional_edges).T
    num_nodes = len(node_map)
    
    # Create feature matrix
    features = np.zeros((num_nodes, 2), dtype=np.float32)
    rad_features = np.zeros((num_nodes, 2), dtype=np.float32)
    longitude = np.zeros(num_nodes, dtype=np.float32)
    latitude = np.zeros(num_nodes, dtype=np.float32)
    
    for _, row in df.iterrows():
        old_id = int(row['user'])
        if old_id in node_map:
            new_id = node_map[old_id]
            features[new_id] = [row['lat'], row['lon']]
            rad_features[new_id] = [np.radians(row['lat']), np.radians(row['lon'])]
            longitude[new_id] = row['lon']
            latitude[new_id] = row['lat']
    
    # Create Data object
    data = type('Data', (), {})()
    data.edge_index = torch.tensor(edges_remapped, dtype=torch.long)
    data.num_nodes = num_nodes
    data.y = torch.zeros(num_nodes, dtype=torch.long)
    data.x = torch.from_numpy(features)
    data.rad_x = torch.from_numpy(rad_features)
    data.longitude = torch.from_numpy(longitude)
    data.latitude = torch.from_numpy(latitude)

    # Also save node ID mapping information
    data.node_map = node_map
    
    # Add average degree
    data.avg_degree = 2 * edges_remapped.shape[1] / num_nodes if num_nodes > 0 else 0
    
    return data, 1

def load_dataset(name='brightkite', sample_size=None):
    if name.lower() == 'brightkite':
        return load_brightkite(sample_size)
    elif name.lower() == 'gowalla':
        return load_gowalla(sample_size)
    elif name.lower() == 'custom':
        return load_custom()