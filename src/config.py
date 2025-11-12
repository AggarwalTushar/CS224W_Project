DATA_FILE = "data/synthetic_data.xlsx"
DIST_THRESHOLD_KM = 50.0
LOOKBACK_DAYS = 180  # 6 months of history
PREDICTION_HORIZONS = [30, 90, 180]  # Predict 1, 3, 6 months ahead
EPOCHS = 300
LR = 2e-3
WEIGHT_DECAY = 5e-4
HIDDEN_DIM = 256
OUT_DIM = 256
DROPOUT = 0.3
BATCH_SIZE = 64  # Batch size for graphs (each graph = one time window)
OUT_DIR = "./model/"