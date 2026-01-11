# AdaTrip: Adaptive Graph-based Reservoir Inflow Prediction

A deep learning framework for predicting reservoir inflow using Graph Neural Networks (GNN) with adaptive graph refinement and attention mechanisms.

This project implements a spatiotemporal prediction model for reservoir inflow forecasting using:
- **Graph Attention Networks (GAT)** for capturing spatial dependencies between reservoirs
- **LSTM** for temporal sequence modeling
- **Adaptive graph refinement** that dynamically prunes edges based on attention weights
- **Weather prediction integration** for enhanced forecasting accuracy

The model predicts 7-day ahead reservoir inflow based on 30 days of historical data from multiple interconnected reservoirs.

## Project Structure

```
AdaTrip/
├── main.py                          # Main training script with adaptive graph refinement
├── main_adapt.py                    # Enhanced training with adaptive threshold scheduling
├── main_w_future_weather.py         # Training with future weather predictions
├── baselines.py                     # Baseline model evaluation and comparison
├── utils.py                         # Utility functions (datasets, scalers, logging)
├── _preprocess.py                   # Data preprocessing pipeline
├── _pretrain.py                     # Encoder pretraining script
├── models/
│   ├── gnn.py                       # GNN model architectures
│   └── lstm.py                      # LSTM and Transformer models
├── _build_graph.ipynb               # Graph construction from reservoir data
├── _aligning_time.ipynb             # Time series alignment notebook
├── batch_script.sh                  # Batch job submission script
└── data/                            # Data directory (see Data Organization)
```

## Setup

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- PyTorch Geometric
- NumPy, Pandas, scikit-learn
- tqdm

### Directory Creation

Before running any scripts, create the required data directories:

```bash
mkdir -p data
mkdir -p logs
```

## Data Organization

### Required Directory Structure

The `data/` directory should be organized as follows:

```
data/
├── align/                           # Time-aligned reservoir CSV files
│   ├── reservoir1.csv               # Format: temperature, precipitation, inflow_x, inflow_y, date
│   ├── reservoir2.csv
│   └── ...
├── parsed/                          # Preprocessed data (auto-generated)
│   ├── all_rsr_data_global.pkl      # Global scaler preprocessed data
│   ├── all_rsr_data_local.pkl       # Local scaler preprocessed data
│   ├── _GNN_supervise_global.pt     # Global scaler supervised learning data
│   └── _GNN_supervise_local.pt      # Local scaler supervised learning data
├── graph/                           # Graph structures (auto-generated)
│   ├── config3.pkl                  # Base graph configuration
│   ├── k2_config3.pkl               # k=2 nearest neighbor graph
│   └── k5_config3.pkl               # k=5 nearest neighbor graph
└── weather/                         # Weather prediction data
    └── weather_data_2009_2011.pkl   # Future weather predictions (temp, precip)
```

### Data File Requirements

#### Input CSV Files (`data/align/`)
Each reservoir CSV file should contain:
- **Column 0**: Temperature (Celsius)
- **Column 1**: Precipitation (mm)
- **Column 2**: Inflow X coordinate
- **Column 3**: Inflow Y coordinate (target variable)
- **Column 4**: Date (YYYY-MM-DD format)

**Time Periods**:
- Pretrain: Before 1999-01-01
- Train: 1999-01-01 to 2008-12-31
- Test: 2009-01-01 to 2011-12-31

#### Weather Prediction Data (`data/weather/`)
Required for `main_w_future_weather.py`:
- Format: Pickle file containing weather predictions
- Columns: reservoir name, date, temperature (Kelvin), precipitation (meters)
- Covers test period: 2009-2011

### Logs Directory

The `logs/` directory stores training outputs:

```
logs/
├── ReservoirAttentionNet/
│   ├── global/
│   │   ├── checkpoint_YYYYMMDDHHMM.pth      # Best model checkpoint
│   │   ├── edge_tracking_YYYYMMDDHHMM.pkl   # Edge refinement history
│   │   └── results_YYYYMMDDHHMM.txt         # Training results
│   └── local/
│       └── ...
├── ReservoirNetSeq2Seq/
│   └── ...
└── PretrainModel/
    ├── global/
    │   └── pretrain_best_model.pth          # Pretrained encoder weights
    └── local/
        └── pretrain_best_model.pth
```


## Basic Workflow

1. **Prepare Data**:
   ```bash
   # Create required directories
   mkdir -p data/align data/parsed data/graph data/weather logs

   # Place reservoir CSV files in data/align/
   # Run graph construction notebook
   jupyter notebook _build_graph.ipynb
   ```

2. **Preprocess Data**:
   ```bash
   python _preprocess.py
   ```
   Generates preprocessed data for both global and local scalers.

3. **Optional: Pretrain Encoder**:
   ```bash
   python _pretrain.py
   ```

4. **Train Model**:
   ```bash
   # Standard training with graph refinement
   python main.py

   # Or adaptive threshold training
   python main_adapt.py

   # Or weather-enhanced training
   python main_w_future_weather.py
   ```

5. **Evaluate Baselines**:
   ```bash
   python baselines.py
   ```


To load and evaluate a trained model:

```python
# In main.py
load_checkpoint = True
checkpoint_time = "202507101845_p"  # Timestamp from checkpoint filename
```

The script will automatically find the closest matching checkpoint if exact timestamp doesn't exist.




- **Architecture**: Transformer encoder
- **Key Feature**: Self-attention baseline
- **File**: `baselines.py`

## Training Pipeline

### Data Flow

```
Raw CSV → _preprocess.py → Preprocessed Data → Main Script → Trained Model
                                ↓
                          Graph Structure
                          (from _build_graph.ipynb)
```

### Hyperparameters

**Model Architecture**:
- Input dimension: 3 features (temperature, precipitation, inflow)
- Hidden dimension: 128
- GNN dimension: 64
- LSTM dimension: 64
- Prediction days: 7

**Training**:
- Optimizer: Adam (lr=1e-3)
- Scheduler: StepLR (step_size=1, gamma=0.5)
- Loss: MSE
- Epochs: 10
- Batch size: 4

**Graph Refinement** (main.py):
- Threshold: 0.3
- Frequency: Every 4 epochs
- Self-loops: Always preserved



- **Scaler Types**:
  - `global`: Better for cross-reservoir generalization
  - `local`: Better for reservoir-specific patterns
- **Graph Refinement**: Improves efficiency by pruning irrelevant edges while maintaining performance
- **Weather Integration**: Requires separate weather prediction pipeline (not included)
- **GPU Support**: Automatically uses CUDA if available

## Citation

If you use this code, please cite the corresponding paper.
