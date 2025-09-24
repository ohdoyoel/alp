import argparse
import os
import time
import torch
import pandas as pd
from dataset import load_dataset
from model.lp import label_propagation
from model.flp import fixed_alpha_label_propagation
from model.alp import adaptive_label_propagation
from utils import compute_jaccard_similarity, compute_location_similarity

def run_clustering(algorithm, dataset, alpha=None, output_path='../result'):
    """
    Run clustering once with specified algorithm and dataset, then save results.
    
    Args:
        algorithm: Algorithm type ('LP', 'FLP', 'ALP')
        dataset: Dataset name ('brightkite', 'gowalla')
        alpha: Alpha value for FLP algorithm (0~1, required for FLP)
        output_path: Directory path where result files will be saved
    
    Returns:
        str: Path of the saved file
    """
    
    print(f"Algorithm: {algorithm}, Dataset: {dataset}")
    if algorithm == 'FLP' and alpha is not None:
        print(f"Alpha value: {alpha}")
    print(f"Output path: {output_path}")
    
    # Load dataset
    print(f"\nLoading dataset...")
    data, _ = load_dataset(dataset)
    num_nodes = data.num_nodes
    print(f"Number of nodes: {num_nodes}, Number of edges: {data.edge_index.shape[1]}")
    print(f"Average degree: {data.avg_degree}")
    
    # Set initial labels (unique label for each node)
    labels = torch.arange(num_nodes)
    
    start_time = time.time()
    
    if algorithm == 'LP':
        # Pure Label Propagation
        print("\nRunning LP algorithm...")
        pred_labels, iterations = label_propagation(data.edge_index, labels)
        
    elif algorithm == 'FLP':
        # Fixed Alpha Label Propagation
        if alpha is None:
            raise ValueError("FLP algorithm requires alpha value.")
        
        print(f"\nRunning FLP algorithm (alpha={alpha})...")
        
        # Calculate similarities (required for FLP)
        structure_similarity = compute_jaccard_similarity(data)
        location_similarity = compute_location_similarity(data)
        
        pred_labels, _, iterations = fixed_alpha_label_propagation(
            data, labels, alpha, 
            structure_similarity=structure_similarity,
            location_similarity=location_similarity
        )
        
    elif algorithm == 'ALP':
        # Adaptive Label Propagation
        print("\nRunning ALP algorithm...")
        
        # Calculate similarities (required for ALP)
        structure_similarity = compute_jaccard_similarity(data)
        location_similarity = compute_location_similarity(data)
        
        pred_labels, _, iter_info, alpha_info = adaptive_label_propagation(
            data, labels,
            structure_similarity=structure_similarity,
            location_similarity=location_similarity
        )
        iterations = len(alpha_info)
        
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Please choose one of 'LP', 'FLP', 'ALP'.")
    
    elapsed_time = time.time() - start_time
    print(f"\nExecution completed: {elapsed_time:.2f} seconds")
    print(f"Number of iterations: {iterations}")
    print(f"Number of clusters found: {len(torch.unique(pred_labels))}")
    
    # Create result folder
    result_dir = output_path
    os.makedirs(result_dir, exist_ok=True)
    
    # Generate filename
    if algorithm == 'FLP' and alpha is not None:
        filename = f"{algorithm.lower()}_{dataset}_alpha{alpha}.csv"
    else:
        filename = f"{algorithm.lower()}_{dataset}.csv"
    
    filepath = os.path.join(result_dir, filename)
    
    # Convert results to DataFrame
    result_df = pd.DataFrame({
        'node_id': range(num_nodes),
        'label': pred_labels.cpu().numpy()
    })
    
    # Save as CSV file
    result_df.to_csv(filepath, index=False)
    
    return filepath

def main():
    """
    Parse command line arguments and run clustering.
    """
    parser = argparse.ArgumentParser(description='Run clustering algorithms')
    parser.add_argument('--algorithm', type=str,
                       help='Algorithm to use (LP: Label Propagation, FLP: Fixed Alpha LP, ALP: Adaptive LP)')
    parser.add_argument('--dataset', type=str,
                       help='Dataset to use (brightkite, gowalla, custom)')
    parser.add_argument('--alpha', type=float, 
                       help='Alpha value to use in FLP algorithm (0~1)')
    parser.add_argument('--output', default='../result', 
                       help='Directory path where result files will be saved (default: ../result)')
    
    args = parser.parse_args()
    
    # Convert input values to handle case-insensitive input
    if args.algorithm:
        args.algorithm = args.algorithm.upper()
    if args.dataset:
        args.dataset = args.dataset.lower()
    
    # Check if values are valid
    valid_algorithms = ['LP', 'FLP', 'ALP']
    valid_datasets = ['brightkite', 'gowalla', 'custom']
    
    if args.algorithm not in valid_algorithms:
        print(f"Error: Unsupported algorithm. Available algorithms: {', '.join(valid_algorithms)}")
        return
    
    if args.dataset not in valid_datasets:
        print(f"Error: Unsupported dataset. Available datasets: {', '.join(valid_datasets)}")
        return
    
    # Check required alpha value for FLP algorithm
    if args.algorithm == 'FLP':
        if args.alpha is None:
            print("Error: FLP algorithm requires --alpha value.")
            print("Example: python main.py --algorithm FLP --dataset brightkite --alpha 0.5")
            return
        if not (0 <= args.alpha <= 1):
            print("Error: alpha value must be between 0 and 1.")
            return
    
    # Run clustering
    try:
        result_file = run_clustering(args.algorithm, args.dataset, args.alpha, args.output)
        print(f"\n✅ Successfully completed!")
        print(f"Result file: {result_file}")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
