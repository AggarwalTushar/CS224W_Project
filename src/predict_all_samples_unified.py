import os
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler

from config import DATA_FILE, OUT_DIR, PREDICTION_HORIZONS, DIST_THRESHOLD_KM, LOOKBACK_DAYS
from data_utils import load_and_prepare_data, build_edge_index, build_temporal_graphs, build_spatiotemporal_dataset
from model import GraphSAGE
from model_transformer import HeterogeneousTransformer


def predict_all_samples_unified(checkpoint_path, model_type=None, out_dir=OUT_DIR, device=None, data_file=DATA_FILE):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    os.makedirs(out_dir, exist_ok=True)

    print(f"Using device: {device}")
    print(f"Model type: {model_type}")
    print(f"Data file: {data_file}")
    # Load checkpoint
    print(f"Loading checkpoint from: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if not isinstance(ckpt, dict) or 'model_state' not in ckpt:
        raise RuntimeError("Checkpoint does not contain 'model_state' key.")

    saved_state = ckpt['model_state']
    saved_config = ckpt.get('config', {})

    # Load data
    print("Loading data...")
    df = load_and_prepare_data(data_file)

    # Build dataset based on model type
    if model_type == 'graphsage':
        print("Building spatial graph (edge_index)")
        nodes, node_to_idx, edge_index = build_edge_index(df, DIST_THRESHOLD_KM)

        print("Building temporal graph snapshots...")
        data_list = build_temporal_graphs(df, nodes, node_to_idx, edge_index, LOOKBACK_DAYS)

        feat_dim = data_list[0].x.shape[1]
        scaler = ckpt.get('scaler')
        if scaler is None:
            raise RuntimeError("Checkpoint does not contain 'scaler' key.")

        # Instantiate model
        if model_type == 'graphsage':
            model = GraphSAGE(
                feat_dim, 
                hidden_dim=saved_config.get('hidden_dim', 256), 
                out_dim=saved_config.get('out_dim', 256), 
                n_horizons=len(PREDICTION_HORIZONS), 
                dropout=saved_config.get('dropout', 0.3)
            )
        
        model.load_state_dict(saved_state)
        model = model.to(device)
        model.eval()

        # Run predictions
        records = []
        with torch.no_grad():
            for t_idx, data in enumerate(data_list):
                sample_date = getattr(data, 'sample_date', None)

                # Apply scaler
                x_np = data.x.numpy()
                x_scaled = scaler.transform(x_np)
                x_tensor = torch.tensor(x_scaled, dtype=torch.float32).to(device)

                # Create data container
                data_for_model = type('D', (), {})()
                data_for_model.x = x_tensor
                data_for_model.edge_index = data.edge_index.to(device)

                logits = model(data_for_model)
                probs = torch.sigmoid(logits).cpu().numpy()

                targets = data.y.numpy()
                node_mask = data.node_mask.numpy() if hasattr(data, 'node_mask') else np.ones(targets.shape[0], dtype=bool)

                n_nodes, n_h = probs.shape
                for node_idx in range(n_nodes):
                    if not node_mask[node_idx]:
                        continue
                    for h_idx, horizon in enumerate(PREDICTION_HORIZONS):
                        p = float(probs[node_idx, h_idx])
                        pred = int(p > 0.5)
                        true = int(targets[node_idx, h_idx])

                        records.append({
                            'time_index': t_idx,
                            'sample_date': pd.Timestamp(sample_date) if sample_date is not None else None,
                            'node_idx': int(node_idx),
                            'horizon_days': int(horizon),
                            'true_label': true,
                            'predicted_label': pred,
                            'prob_no_slip': 1.0 - p,
                            'prob_slip': p
                        })

    elif model_type == 'transformer':
        print("Building transformer dataset...")
        lookback_months = saved_config.get('lookback_months', int(LOOKBACK_DAYS / 30))
        data_list, dist_tensor = build_spatiotemporal_dataset(df, lookback_months=lookback_months)
        
        # Get scaler and apply to data
        t_scaler = ckpt.get('t_scaler')
        if t_scaler is None:
            print("Warning: No temporal scaler found in checkpoint. Using RobustScaler fit on all data.")
            all_temporal = []
            for d in data_list:
                all_temporal.append(d.x_temporal.view(-1, 3).numpy())
            all_temporal = np.concatenate(all_temporal, axis=0)
            t_scaler = RobustScaler()
            t_scaler.fit(all_temporal)
        
        # Pre-scale data
        for d in data_list:
            N, L, C = d.x_temporal.shape
            t_flat = d.x_temporal.view(-1, C).numpy()
            t_scaled = t_scaler.transform(t_flat)
            d.x_temporal = torch.tensor(t_scaled, dtype=torch.float32).view(N, L, C)
            d.x_spatial = d.x_spatial.view(-1).long()

        num_nodes = data_list[0].x_spatial.shape[0]

        # Instantiate model
        model = HeterogeneousTransformer(
            num_nodes=num_nodes,
            temporal_feat_dim=saved_config.get('temporal_dim', 3),
            hidden_dim=saved_config.get('hidden_dim', 64),
            num_heads=saved_config.get('num_heads', 4),
            num_layers=saved_config.get('num_layers', 2),
            num_horizons=saved_config.get('num_horizons', len(PREDICTION_HORIZONS)),
            dropout=saved_config.get('dropout', 0.5),
            tau_km=25.0
        )
        
        model.load_state_dict(saved_state)
        model = model.to(device)
        model.eval()

        # Run predictions
        records = []
        with torch.no_grad():
            for t_idx, data in enumerate(data_list):
                x_s = data.x_spatial.unsqueeze(0).to(device)  # (1, N)
                x_t = data.x_temporal.unsqueeze(0).to(device)  # (1, N, L, 3)
                dist = data.dist_matrix.unsqueeze(0).to(device)  # (1, N, N)
                
                logits = model(x_s, x_t, dist).squeeze(0)  # (N, num_horizons)
                probs = torch.sigmoid(logits).cpu().numpy()

                targets = data.y.numpy()
                n_nodes, n_h = probs.shape
                
                for node_idx in range(n_nodes):
                    for h_idx, horizon in enumerate(PREDICTION_HORIZONS):
                        p = float(probs[node_idx, h_idx])
                        pred = int(p > 0.5)
                        true = int(targets[node_idx, h_idx])

                        records.append({
                            'time_index': t_idx,
                            'sample_date': None,  # transformer doesn't track sample_date currently
                            'node_idx': int(node_idx),
                            'horizon_days': int(horizon),
                            'true_label': true,
                            'predicted_label': pred,
                            'prob_no_slip': 1.0 - p,
                            'prob_slip': p
                        })
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Create DataFrame
    pred_df = pd.DataFrame.from_records(records)

    # Save predictions summary with model-specific naming
    out_preds_dir = os.path.join(out_dir, 'predictions')
    os.makedirs(out_preds_dir, exist_ok=True)
    preds_filename = os.path.join(out_preds_dir, f'predictions_summary_{model_type}.csv')
    pred_df.to_csv(preds_filename, index=False)
    print(f"Saved predictions summary to: {preds_filename}")

    # Compute overall accuracy and per-horizon accuracy
    if not pred_df.empty:
        overall_acc = (pred_df['predicted_label'] == pred_df['true_label']).mean()
        print(f"Overall accuracy (all horizons): {overall_acc:.4f}")

        for horizon in sorted(pred_df['horizon_days'].unique()):
            df_h = pred_df[pred_df['horizon_days'] == horizon]
            acc_h = (df_h['predicted_label'] == df_h['true_label']).mean()
            print(f"Horizon {horizon}d accuracy: {acc_h:.4f} (samples={len(df_h)})")

    # Per-node statistics
    node_stats = []
    for node_idx, g in pred_df.groupby('node_idx'):
        total = len(g)
        correct = (g['predicted_label'] == g['true_label']).sum()
        acc = correct / total if total > 0 else np.nan
        mean_conf = np.mean([row['prob_slip'] if row['predicted_label'] == 1 else row['prob_no_slip'] for _, row in g.iterrows()])
        node_stats.append({
            'node_idx': int(node_idx),
            'Total_Samples': int(total),
            'Correct': int(correct),
            'Accuracy': float(acc),
            'Mean_Confidence': float(mean_conf)
        })

    node_stats_df = pd.DataFrame(node_stats).set_index('node_idx')
    node_stats_filename = os.path.join(out_preds_dir, f'node_statistics_{model_type}.csv')
    node_stats_df.to_csv(node_stats_filename)
    print(f"Saved node statistics to: {node_stats_filename}")

    return pred_df, node_stats_df


if __name__ == '__main__':
    r"""python .\src\predict_all_samples_unified.py --checkpoint model\spatiotemporal_unified_repeaters.pth --data_file data/repeaters.csv --model_type transformer"""
    parser = argparse.ArgumentParser(description='Run predictions on all samples for any model type')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint (.pth file)')
    parser.add_argument('--model_type', type=str, default=None, 
                        choices=['graphsage', 'transformer'],
                        help='Type of model: graphsage or transformer')
    parser.add_argument('--out_dir', type=str, default=OUT_DIR, help='Output directory for predictions')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu, auto-detect if not specified)')
    parser.add_argument('--data_file', type=str, default=DATA_FILE, help='Path to input data file (CSV)')
    args = parser.parse_args()
    
    device = None
    if args.device:
        device = torch.device(args.device)
    
    predict_all_samples_unified(
        checkpoint_path=args.checkpoint,
        model_type=args.model_type,
        out_dir=args.out_dir,
        device=device,
        data_file=args.data_file
    )
