DATA_FILE = "data/synthetic_data_loading_rates.xlsx"
DIST_THRESHOLD_KM = 50.0
LOOKBACK_DAYS = 540  # 6 months of history
PREDICTION_HORIZONS = [30, 90, 180]  # Predict 1, 3, 6 months ahead
EPOCHS = 200
LR = 2e-3
WEIGHT_DECAY = 5e-4
HIDDEN_DIM = 64
OUT_DIM = 64
DROPOUT = 0.3
BATCH_SIZE = 64  # Batch size for graphs (each graph = one time window)
OUT_DIR = "./model/"
USE_LOADING_RATE = False
USE_RECURRENCE_TIME_TASK = True

TRAIN_SPLIT = .7
VAL_SPLIT = .2
TEST_SPLIT = .1