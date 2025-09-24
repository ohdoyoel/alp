# Entropy-Guided Adaptive Label Propagation for Location-Aware Graph Clustering

This repository provides a complete implementation of _Entropy-Guided Adaptive Label Propagation for Location-Aware Graph Clustering_.

- **Adaptive Label Propagation (ALP)**: A label propagation algorithm that uses entropy-guided weighting to adjust the weights of structural and location information.
- **Fixed Label Propagation (FLP)**: A label propagation algorithm that uses a fixed α value to combine structural and location information.
- **Label Propagation (LP)**: A classic label propagation algorithm that uses majority voting.

The proposed ALP algorithm shows better performance than both FLP and LP algorithms in location-aware graph clustering tasks.

## Code Structure

```
├── code/
│   ├── model/
│   │   ├── lp.py                   # Label Propagation (Majority Vote)
│   │   ├── flp.py                  # Fixed Label Propagation (Fixed α)
│   │   ├── alp.py                  # Adaptive Label Propagation (Adaptive α)
│   │   └── label_propagation.py    # Label Propagation implementation from NetworkX
│   ├── dataset.py                  # Dataset loading
│   ├── utils.py                    # Utility functions
│   └── main.py                     # Entry point
├── datasets/                       # Datasets (Brightkite, Gowalla, Custom)
│   │   ├── custom/            
│   │   │   ├── custom_edges.txt                
│   │   │   ├── custom_totalCheckins.txt
│   │   │  # Download from the link at the bottom
│   │   ├── Brightkite_edges.txt            
│   │   ├── Brightkite_totalCheckins.txt               
│   │   ├── Gowalla_edges.txt          
│   │   └── Gowalla_totalCheckins.txt  
└── README.md
```

## How to Run

### Requirements

- Python 3.8+

```
pip install -r requirements.txt
```

### Example Usage

```
python main.py [-h] [--algorithm ALGORITHM] [--alpha ALPHA] [--dataset DATASET] [--output OUTPUT]
```

```
python main.py --algorithm LP --dataset brightkite
python main.py --algorithm FLP --dataset brightkite --alpha 0.5
python main.py --algorithm ALP --dataset brightkite
```

### Arguments

| Parameter     | Description                                                            |
| ------------- | ---------------------------------------------------------------------- |
| `--algorithm` | Algorithm to use (`LP`, `FLP`, `ALP`)                                  |
| `--alpha`     | α value to use in FLP algorithm (0~1)                                  |
| `--dataset`   | Dataset to use (`brightkite`, `gowalla`, `custom`)                     |
| `--output`    | Directory path where result files will be saved (default: `../result`) |

### Datasets

- [Brightkite](https://snap.stanford.edu/data/loc-brightkite.html)
- [Gowalla](https://snap.stanford.edu/data/loc-gowalla.html)
- Custom


