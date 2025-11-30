import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import argparse
from tqdm import trange
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score

from data_utils import load_and_prepare_data, build_spatiotemporal_dataset
from model_spatiotemporal import HeterogeneousSpatioTemporalTransformer
from model import FocalLoss 
from plot_utils import (
    plot_training_curves, plot_roc_curves, plot_precision_recall_curves, 
    plot_confusion_matrices, plot_performance_metrics, plot_comprehensive_summary
)
from config import HIDDEN_DIM, OUT_DIR, EPOCHS, PREDICTION_HORIZONS, LOOKBACK_DAYS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONFIG = {
    'data_file': "data/synthetic_data.xlsx",
    'epochs': EPOCHS,
    'batch_size': 16,
    'out_dir': OUT_DIR,
    'lookback_months': int(LOOKBACK_DAYS/30),
    'temporal_dim': 3,
    'hidden_dim': 16,
    'num_layers': 2,
    'num_heads': 4,
    'num_horizons': len(PREDICTION_HORIZONS),
    'dropout': 0.4
}
file_name = os.path.splitext(os.path.basename(CONFIG['data_file']))[0]


def collate_fn(batch):
    return {
        'x_spatial': torch.stack([b.x_spatial for b in batch]),
        'x_temporal': torch.stack([b.x_temporal for b in batch]),
        'y': torch.stack([b.y for b in batch]),
        'dist_matrix': torch.stack([b.dist_matrix for b in batch]),
    }


def prescale_data(data_list, scaler=None, fit=False):
    """
    Pre-scale temporal features and convert spatial to LongTensor.
    """
    if fit:
        print("Fitting scaler on training data")
        all_temporal = []
        for d in data_list:
            all_temporal.append(d.x_temporal.view(-1, 3).numpy())
        all_temporal = np.concatenate(all_temporal, axis=0)
        scaler = RobustScaler()
        scaler.fit(all_temporal)
    
    for d in data_list:
        # Scale temporal
        N, L, C = d.x_temporal.shape
        t_flat = d.x_temporal.view(-1, C).numpy()
        t_scaled = scaler.transform(t_flat)
        d.x_temporal = torch.tensor(t_scaled, dtype=torch.float32).view(N, L, C)
        # Convert spatial to Long
        d.x_spatial = d.x_spatial.view(-1).long()
    
    return scaler


def train_model(epochs=EPOCHS, hidden_dim=HIDDEN_DIM, num_layers=2, num_heads=4, dropout=0.2, batch_size=16):
    os.makedirs(OUT_DIR, exist_ok=True)
    
    print(f"Loading data from {CONFIG['data_file']}: (Device: {DEVICE})")
    df = load_and_prepare_data(CONFIG['data_file'])
    
    print("Building dataset:")
    # Lookback lookback_months months
    data_list, _ = build_spatiotemporal_dataset(df, lookback_months=CONFIG['lookback_months'])
    
    # Split chronologically into train/val/test (70/15/15)
    n_total = len(data_list)
    n_train = int(0.7 * n_total)
    n_val = int(0.85 * n_total)
    
    train_raw = data_list[:n_train]
    val_raw = data_list[n_train:n_val]
    test_raw = data_list[n_val:]
    
    print(f"Train snapshots: {len(train_raw)}, Val snapshots: {len(val_raw)}, Test snapshots: {len(test_raw)}")
    
    # Pre-scale data (fit on train, apply to all)
    t_scaler = prescale_data(train_raw, fit=True)
    prescale_data(val_raw, scaler=t_scaler, fit=False)
    prescale_data(test_raw, scaler=t_scaler, fit=False)
    
    train_loader = DataLoader(train_raw, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_raw, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_raw, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    # Init Model
    num_nodes = train_raw[0].x_spatial.shape[0]
    
    model = HeterogeneousSpatioTemporalTransformer(
        num_nodes=num_nodes,
        temporal_feat_dim=CONFIG['temporal_dim'],
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_horizons=len(PREDICTION_HORIZONS),
        dropout=dropout,
        tau_km=25.0
    ).to(DEVICE)
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)
    criterion = FocalLoss(alpha=0.70, gamma=2.0)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("Starting Training:")
    best_val_auc = -1.0
    best_state = None
    patience = 10  # number of validations to wait
    bad = 0
    
    history = {
        'epochs': [],
        'train_loss': [],
        'val_epochs': [],
        'val_loss': [],
        'val_auc': []
    }
    for horizon in PREDICTION_HORIZONS:
        history[f'val_auc_{horizon}d'] = []
    
    for epoch in trange(1, epochs + 1):
        model.train()
        train_losses = []
        
        for batch in train_loader:
            x_s = batch['x_spatial'].to(DEVICE) 
            x_t = batch['x_temporal'].to(DEVICE)
            y = batch['y'].to(DEVICE)
            dist = batch['dist_matrix'].to(DEVICE)
            
            optimizer.zero_grad()
            logits = model(x_s, x_t, dist)
            loss = criterion(logits, y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        history['epochs'].append(epoch)
        history['train_loss'].append(np.mean(train_losses) if train_losses else 0.0)
        
        # Validation every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            val_logits = []
            val_targets = []
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    x_s = batch['x_spatial'].to(DEVICE)
                    x_t = batch['x_temporal'].to(DEVICE)
                    y = batch['y'].to(DEVICE)
                    dist = batch['dist_matrix'].to(DEVICE)
                    
                    logits = model(x_s, x_t, dist)
                    val_loss = criterion(logits, y)
                    
                    val_logits.append(logits.cpu())
                    val_targets.append(y.cpu())
                    val_losses.append(val_loss.item())
            
            if len(val_logits) > 0:
                val_logits = torch.cat(val_logits, dim=0).view(-1, len(PREDICTION_HORIZONS)).numpy()
                val_targets = torch.cat(val_targets, dim=0).view(-1, len(PREDICTION_HORIZONS)).numpy()
                val_probs = 1 / (1 + np.exp(-val_logits))
                
                # Calculate AUC for each horizon
                aucs = []
                for i in range(len(PREDICTION_HORIZONS)):
                    if len(np.unique(val_targets[:, i])) > 1:
                        auc = roc_auc_score(val_targets[:, i], val_probs[:, i])
                        aucs.append(auc)
                        history[f'val_auc_{PREDICTION_HORIZONS[i]}d'].append(auc)
                    else:
                        history[f'val_auc_{PREDICTION_HORIZONS[i]}d'].append(0)
                
                avg_auc = np.mean(aucs) if aucs else 0
                history['val_epochs'].append(epoch)
                history['val_loss'].append(np.mean(val_losses))
                history['val_auc'].append(avg_auc)
                
                if avg_auc > best_val_auc + 1e-4:
                    best_val_auc = avg_auc
                    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1

                if bad >= patience:
                    print(f"Early stopping at epoch {epoch}, best val AUC={best_val_auc:.4f}")
                    break
                
                print(f"Epoch {epoch:03d} | Loss {np.mean(train_losses):.4f} | Val Loss {np.mean(val_losses):.4f} | Val AUC {avg_auc:.4f} | " +
                      " | ".join([f"{PREDICTION_HORIZONS[i]}d: {aucs[i]:.3f}" for i in range(len(aucs))]))
            else:
                print(f"Epoch {epoch:03d} | Loss {np.mean(train_losses):.4f} | Val set empty")
    
    # Plot training curves
    plot_training_curves(history, os.path.join(OUT_DIR, f"training_curves_spatiotemporal_{file_name}.png"))
    
    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)
    
    # Test evaluation
    print("=" * 60)
    print("TEST EVALUATION")
    print("=" * 60)
    
    model.eval()
    test_logits = []
    test_targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            x_s = batch['x_spatial'].to(DEVICE)
            x_t = batch['x_temporal'].to(DEVICE)
            y = batch['y'].to(DEVICE)
            dist = batch['dist_matrix'].to(DEVICE)
            
            logits = model(x_s, x_t, dist)
            test_logits.append(logits.cpu())
            test_targets.append(y.cpu())
    
    if len(test_logits) > 0:
        test_logits = torch.cat(test_logits, dim=0).view(-1, len(PREDICTION_HORIZONS)).numpy()
        test_targets = torch.cat(test_targets, dim=0).view(-1, len(PREDICTION_HORIZONS)).numpy()
        test_probs = 1 / (1 + np.exp(-test_logits))
        
        metrics_dict = {}
        best_thresholds = []
        
        for i, horizon in enumerate(PREDICTION_HORIZONS):
            print(f"\n--- Horizon: {horizon} days ---")
            metrics_dict[horizon] = {}
            
            if len(np.unique(test_targets[:, i])) > 1:
                auc = roc_auc_score(test_targets[:, i], test_probs[:, i])
                
                # Find optimal threshold
                best_f1 = 0
                best_thresh = 0.5
                for thresh in np.linspace(0.1, 0.9, 17):
                    preds = (test_probs[:, i] > thresh).astype(int)
                    _, _, f1, _ = precision_recall_fscore_support(
                        test_targets[:, i], preds, average="binary", zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = thresh
                
                best_thresholds.append(best_thresh)
                preds = (test_probs[:, i] > best_thresh).astype(int)
                acc = accuracy_score(test_targets[:, i], preds)
                prec, rec, f1, _ = precision_recall_fscore_support(
                    test_targets[:, i], preds, average="binary", zero_division=0)
                
                metrics_dict[horizon] = {
                    'auc': auc,
                    'accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'f1': f1,
                    'threshold': best_thresh
                }
                
                print(f"AUC: {auc:.4f} | Threshold: {best_thresh:.3f}")
                print(f"ACC: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
            else:
                best_thresholds.append(0.5)
                print("Insufficient class diversity in test set.")
        
        # Generate all plots
        print("=" * 60)
        print("GENERATING PLOTS")
        print("=" * 60)
        
        plot_roc_curves(test_targets, test_probs, os.path.join(OUT_DIR, f"roc_curves_spatiotemporal_{file_name}.png"))
        plot_precision_recall_curves(test_targets, test_probs, os.path.join(OUT_DIR, f"pr_curves_spatiotemporal_{file_name}.png"))
        plot_confusion_matrices(test_targets, test_probs, best_thresholds, os.path.join(OUT_DIR, f"confusion_matrices_spatiotemporal_{file_name}.png"))
        plot_performance_metrics(metrics_dict, os.path.join(OUT_DIR, f"performance_metrics_spatiotemporal_{file_name}.png"))
        plot_comprehensive_summary(metrics_dict, os.path.join(OUT_DIR, f"comprehensive_summary_spatiotemporal_{file_name}.png"))
        
        print("ALL PLOTS SAVED SUCCESSFULLY!")
        
        # Save model
        torch.save({
            'model_state': model.state_dict(),
            't_scaler': t_scaler,
            # No spatial scaler needed
            'config': {
                    'num_nodes': num_nodes,
                    'lookback_months': CONFIG['lookback_months'],
                    'temporal_dim': CONFIG['temporal_dim'],
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'num_heads': num_heads,
                    'num_horizons': len(PREDICTION_HORIZONS),
                    'dropout': dropout
            },
            'metrics': metrics_dict
        }, os.path.join(OUT_DIR, f"spatiotemporal_unified_{file_name}.pth"))
        
    print("Training Complete.")
    return model, metrics_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train SpatioTemporal Transformer for Earthquake Prediction')
    parser.add_argument('--epochs', type=int, default=CONFIG['epochs'], help='Number of training epochs')
    parser.add_argument('--hidden_dim', type=int, default=CONFIG['hidden_dim'], help='Hidden dimension')
    parser.add_argument('--num_layers', type=int, default=CONFIG['num_layers'], help='Number of transformer layers')
    parser.add_argument('--num_heads', type=int, default=CONFIG['num_heads'], help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=CONFIG['dropout'], help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=CONFIG['batch_size'], help='Batch size')
    args = parser.parse_args()
    
    train_model(
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        batch_size=args.batch_size
    )