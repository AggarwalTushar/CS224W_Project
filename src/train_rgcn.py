from config import LOOKBACK_DAYS, PREDICTION_HORIZONS, DATA_FILE, DIST_THRESHOLD_KM, OUT_DIR, EPOCHS, LR, WEIGHT_DECAY, HIDDEN_DIM, OUT_DIM, DROPOUT, BATCH_SIZE, TRAIN_SPLIT, VAL_SPLIT, USE_RECURRENCE_TIME_TASK
from data_utils import load_and_prepare_data, build_edge_index, build_temporal_graphs, build_temporal_snapshot_graph, process_repeaters_csv
from plot_utils import plot_training_curves, plot_roc_curves, plot_precision_recall_curves, plot_confusion_matrices, plot_performance_metrics, plot_comprehensive_summary
from model import RGCN, GraphSAGE, FocalLoss
import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score
import numpy as np
import pandas as pd
import os
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(model, opt, scheduler, loss_fn, train_loader, val_loader, test_loader, epochs=EPOCHS):
    """
    Trains the model
    """
    
    # Initialize model
    best_val_auc = 0
    best_val_loss = float("inf")
    best_state = None
    
    history = {
        'epochs': [],
        'train_loss': [],
        'val_epochs': [],
        'val_loss': []
    }
    
    if not USE_RECURRENCE_TIME_TASK:
        history['val_auc']: []
        for horizon in PREDICTION_HORIZONS:
            history[f'val_auc_{horizon}d'] = []
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        
        for batch in train_loader:
            batch = batch.to(DEVICE)
            logits = model(batch)
            # targets = batch.y[batch.node_mask]
            targets = batch.y
            loss = loss_fn(logits, targets)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            train_losses.append(loss.item())
        
        scheduler.step()
        history['epochs'].append(epoch)
        history['train_loss'].append(np.mean(train_losses))
        
        # Validation
        if epoch % 5 == 0:
            model.eval()
            val_logits = []
            val_targets = []
            val_losses = []
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(DEVICE)
                    logits = model(batch)
                    # targets = batch.y[batch.node_mask]
                    targets = batch.y
                    val_loss = loss_fn(logits, targets)
                    val_logits.append(logits.cpu())
                    val_targets.append(targets.cpu())
                    val_losses.append(val_loss.item())
            
            val_logits = torch.cat(val_logits, dim = 0).numpy()
            val_targets = torch.cat(val_targets, dim = 0).numpy()
            val_probs = 1 / (1 + np.exp(-val_logits))
            
            # Calculate AUC for each horizon
            if USE_RECURRENCE_TIME_TASK:
                history['val_epochs'].append(epoch)
                mean_val_loss = np.mean(val_losses)
                history['val_loss'].append(mean_val_loss)
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    best_state = model.state_dict()

                print(f"Epoch {epoch:03d} | Loss {np.mean(train_losses):.4f} | Val Loss {np.mean(val_losses):.4f}")
            else:
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
                
                if avg_auc > best_val_auc:
                    best_val_auc = avg_auc
                    best_state = model.state_dict()
                
                print(f"Epoch {epoch:03d} | Loss {np.mean(train_losses):.4f} | Val Loss {np.mean(val_losses):.4f} | Val AUC {avg_auc:.4f} | " + " | ".join([f"{PREDICTION_HORIZONS[i]}d: {aucs[i]:.3f}" for i in range(len(aucs))]))
    
    # Plot training curves
    plot_training_curves(history, os.path.join(OUT_DIR, "training_curves.png"))
    
    # Load best model
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
            batch = batch.to(DEVICE)
            logits = model(batch)
            # targets = batch.y[batch.node_mask]
            targets = batch.y
            test_logits.append(logits.cpu())
            test_targets.append(targets.cpu())
    
    test_logits = torch.cat(test_logits, dim = 0).numpy()
    test_targets = torch.cat(test_targets, dim = 0).numpy()
    test_probs = 1 / (1 + np.exp(-test_logits))
    
    # Store metrics for plotting
    metrics_dict = {}
    best_thresholds = []
    
    if USE_RECURRENCE_TIME_TASK:
        print("TODO")
    else:
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
                    _, _, f1, _ = precision_recall_fscore_support(test_targets[:, i], preds, average = "binary", zero_division=0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_thresh = thresh
                
                best_thresholds.append(best_thresh)
                preds = (test_probs[:, i] > best_thresh).astype(int)
                acc = accuracy_score(test_targets[:, i], preds)
                prec, rec, f1, _ = precision_recall_fscore_support(test_targets[:, i], preds, average = "binary", zero_division=0)
                
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
    
    # Generate all plots
    print("=" * 60)
    print("GENERATING PLOTS")
    print("=" * 60)
    
    if USE_RECURRENCE_TIME_TASK:
        print("TODO")
    else:
        plot_roc_curves(test_targets, test_probs, os.path.join(OUT_DIR, "roc_curves.png"))
        plot_precision_recall_curves(test_targets, test_probs, os.path.join(OUT_DIR, "pr_curves.png"))
        plot_confusion_matrices(test_targets, test_probs, best_thresholds, os.path.join(OUT_DIR, "confusion_matrices.png"))
        plot_performance_metrics(metrics_dict, os.path.join(OUT_DIR, "performance_metrics.png"))
        plot_comprehensive_summary(metrics_dict, os.path.join(OUT_DIR, "comprehensive_summary.png"))
    
    print("=" * 60)
    print("ALL PLOTS SAVED SUCCESSFULLY!")
    print("=" * 60)
    
    return model#, scaler

def data_as_feature_block(data_list):
    # Split data
    data_list = sorted(data_list, key = lambda d: d.sample_date)
    n_total = len(data_list)
    n_train = int(0.7 * n_total) # TODO use splits defined in config
    n_val = int(0.85 * n_total)
    
    train_data = data_list[:n_train]
    val_data = data_list[n_train:n_val]
    test_data = data_list[n_val:]
    
    print(f"Split: Train = {len(train_data)}, Val = {len(val_data)}, Test = {len(test_data)}")
    
    # Count total samples
    train_samples = sum(d.num_active_nodes for d in train_data)
    val_samples = sum(d.num_active_nodes for d in val_data)
    test_samples = sum(d.num_active_nodes for d in test_data)
    print(f"Total samples: Train = {train_samples}, Val = {val_samples}, Test = {test_samples}")
    
    # Normalize features
    train_feats = torch.cat([d.x for d in train_data], dim = 0).numpy()
    scaler = RobustScaler()
    scaler.fit(train_feats)
    
    for d in train_data:
        d.x = torch.tensor(scaler.transform(d.x.numpy()), dtype = torch.float32)
    for d in val_data:
        d.x = torch.tensor(scaler.transform(d.x.numpy()), dtype = torch.float32)
    for d in test_data:
        d.x = torch.tensor(scaler.transform(d.x.numpy()), dtype = torch.float32)
    
    # Create PyG DataLoaders
    train_loader = DataLoader(train_data, batch_size = BATCH_SIZE, shuffle = True)
    val_loader = DataLoader(val_data, batch_size = BATCH_SIZE, shuffle = False)
    test_loader = DataLoader(test_data, batch_size = BATCH_SIZE, shuffle = False)

    return train_loader, val_loader, test_loader

def train_graphSAGE_feature_block(feat_dim, train_loader, val_loader, test_loader):
    # Initialize model
    feat_dim = next(iter(train_loader))[0].x.shape[1] #train_data[0].x.shape[1]
    model = GraphSAGE(feat_dim, HIDDEN_DIM, OUT_DIM, len(PREDICTION_HORIZONS), DROPOUT).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr = LR, weight_decay = WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0 = 20, T_mult = 2)
    loss_fn = FocalLoss(alpha = 0.70, gamma = 2.0)
    return train_model(model, opt, scheduler, loss_fn, train_loader, val_loader, test_loader)

def train_rGCN_temporal_snapshot(train_loader, val_loader, test_loader):
    # feat_dim = next(iter(train_loader))[0]["earthquake_source"].x
    # print(feat_dim)
    sample = next(iter(train_loader))[0]
    feat_dim = sample.num_node_features["earthquake_source"]
    model = RGCN(feat_dim, 4, HIDDEN_DIM, OUT_DIM)
    opt = optim.AdamW(model.parameters(), lr = LR, weight_decay = WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0 = 20, T_mult = 2)
    if USE_RECURRENCE_TIME_TASK:
        loss_fn = torch.nn.MSELoss()
    else:
        loss_fn = FocalLoss(alpha = 0.70, gamma = 2.0)
    return train_model(model, opt, scheduler, loss_fn, train_loader, val_loader, test_loader)

def main():
    print("Loading data...")
    if "repeaters" in DATA_FILE:
        df = process_repeaters_csv(DATA_FILE)
    else:
        df = load_and_prepare_data(DATA_FILE)


    print("Building spatial graphs")
    # nodes, node_to_idx, edge_index = build_edge_index(df, DIST_THRESHOLD_KM)
    all_samples = build_temporal_snapshot_graph(df, int(LOOKBACK_DAYS / 30))
    
    TRAIN_INDEX_END = int(len(all_samples) * TRAIN_SPLIT)
    VAL_INDEX_END = TRAIN_INDEX_END + int(len(all_samples) * VAL_SPLIT)

    train_loader = DataLoader(all_samples[:TRAIN_INDEX_END], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(all_samples[TRAIN_INDEX_END:VAL_INDEX_END], batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(all_samples[VAL_INDEX_END:], batch_size=BATCH_SIZE, shuffle=True)

    # print("Building graphs data")
    # data_list = build_temporal_graphs(df, nodes, node_to_idx, edge_index, LOOKBACK_DAYS)
    
    print("Training model")
    # model, scaler = train_model(all_samples)
    model = train_rGCN_temporal_snapshot(train_loader, val_loader, test_loader)

    print("Training complete")
    torch.save({
        'model_state': model.state_dict(),
        # 'scaler': scaler, #TODO add this back in
        # 'nodes': nodes,
        # 'edge_index': edge_index
    }, os.path.join(OUT_DIR, "rgcn_unified.pth"))

    records = []
    for t, sample in enumerate(all_samples):
        out = model(sample)
        for node_idx, node_pred in enumerate(out):
            records.append({
                'time_index': t,
                # 'sample_date': pd.Timestamp(sample_date) if sample_date is not None else None,
                'node_idx': int(node_idx),
                # 'horizon_days': int(horizon),
                'true_label': float(sample.y[node_idx, 0]),
                'predicted_label': float(node_pred),
                # 'prob_no_slip': 1.0 - p,
                # 'prob_slip': p
            })
    # Create DataFrame
    pred_df = pd.DataFrame.from_records(records)

    # Save predictions summary with model-specific naming
    out_preds_dir = os.path.join(OUT_DIR, 'predictions')
    os.makedirs(out_preds_dir, exist_ok=True)
    preds_filename = os.path.join(out_preds_dir, f'predictions_summary_rgcn.csv')
    pred_df.to_csv(preds_filename, index=False)
    print(f"Saved predictions summary to: {preds_filename}")


if __name__ == "__main__":
    main()