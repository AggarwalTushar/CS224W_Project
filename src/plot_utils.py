import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
import numpy as np
from config import PREDICTION_HORIZONS

plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13


def plot_training_curves(history, filename):
    """
    Plot training and validation loss/AUC curves
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4))
    
    # Loss curves
    ax1.plot(history['epochs'], history['train_loss'], label = 'Train Loss', linewidth = 2, marker = 'o', markersize = 3)
    ax1.plot(history['val_epochs'], history['val_loss'], label = 'Val Loss', linewidth = 2, marker = 's', markersize = 3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha = 0.3)
    
    # AUC curves for each horizon
    ax2.plot(history['val_epochs'], history['val_auc'], label = 'Average AUC', linewidth = 2.5, marker = 'o', markersize = 4, color = 'black')
    for i, horizon in enumerate(PREDICTION_HORIZONS):
        if f'val_auc_{horizon}d' in history:
            ax2.plot(history['val_epochs'], history[f'val_auc_{horizon}d'], label = f'{horizon}-day', linewidth = 1.5, marker = 's', markersize = 3, alpha = 0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC-ROC')
    ax2.set_title('Validation AUC by Prediction Horizon')
    ax2.legend()
    ax2.grid(True, alpha = 0.3)
    ax2.set_ylim([0.5, 1.0])
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches = 'tight')
    plt.close()


def plot_roc_curves(test_targets, test_probs, filename):
    """Plot ROC curves for all prediction horizons"""
    fig, ax = plt.subplots(figsize = (8, 6))
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(PREDICTION_HORIZONS)))
    
    for i, horizon in enumerate(PREDICTION_HORIZONS):
        if len(np.unique(test_targets[:, i])) > 1:
            fpr, tpr, _ = roc_curve(test_targets[:, i], test_probs[:, i])
            auc = roc_auc_score(test_targets[:, i], test_probs[:, i])
            ax.plot(fpr, tpr, label = f'{horizon}-day (AUC = {auc:.3f})', linewidth = 2.5, color = colors[i])
    
    ax.plot([0, 1], [0, 1], 'k--', linewidth = 1.5, alpha = 0.5, label = 'Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves for Different Prediction Horizons')
    ax.legend(loc = 'lower right')
    ax.grid(True, alpha = 0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches = 'tight')
    plt.close()


def plot_precision_recall_curves(test_targets, test_probs, filename):
    """
    Plot Precision-Recall curves for all prediction horizons
    """
    fig, ax = plt.subplots(figsize = (8, 6))
    
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(PREDICTION_HORIZONS)))
    
    for i, horizon in enumerate(PREDICTION_HORIZONS):
        if len(np.unique(test_targets[:, i])) > 1:
            precision, recall, _ = precision_recall_curve(test_targets[:, i], test_probs[:, i])
            ax.plot(recall, precision, label=f'{horizon}-day', 
                   linewidth = 2.5, color = colors[i])
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves for Different Prediction Horizons')
    ax.legend(loc = 'best')
    ax.grid(True, alpha = 0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches = 'tight')
    plt.close()


def plot_confusion_matrices(test_targets, test_probs, thresholds, filename):
    """
    Plot confusion matrices for all prediction horizons
    """
    n_horizons = len(PREDICTION_HORIZONS)
    fig, axes = plt.subplots(1, n_horizons, figsize = (4*n_horizons, 3.5))
    
    if n_horizons == 1:
        axes = [axes]
    
    for i, (horizon, ax) in enumerate(zip(PREDICTION_HORIZONS, axes)):
        if len(np.unique(test_targets[:, i])) > 1:
            preds = (test_probs[:, i] > thresholds[i]).astype(int)
            cm = confusion_matrix(test_targets[:, i], preds)
            
            sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', ax = ax, cbar = True, square = True, linewidths = 1, linecolor = 'gray')
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title(f'{horizon}-day Prediction')
            ax.set_xticklabels(['Negative', 'Positive'])
            ax.set_yticklabels(['Negative', 'Positive'])
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches = 'tight')
    plt.close()


def plot_performance_metrics(metrics_dict, filename):
    """
    Plot bar chart of performance metrics across horizons
    """
    fig, axes = plt.subplots(2, 2, figsize = (12, 10))
    axes = axes.flatten()
    
    metric_names = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score']
    metric_keys = ['auc', 'accuracy', 'precision', 'recall', 'f1']
    
    horizons = [str(h) for h in PREDICTION_HORIZONS]
    x = np.arange(len(horizons))
    
    # Plot first 4 metrics in subplots
    for idx, (metric_name, metric_key) in enumerate(zip(metric_names[:4], metric_keys[:4])):
        values = [metrics_dict[h][metric_key] for h in PREDICTION_HORIZONS]
        axes[idx].bar(x, values, width = 0.6, alpha = 0.8, color = plt.cm.Set2(idx))
        axes[idx].set_ylabel(metric_name)
        axes[idx].set_title(f'{metric_name} by Prediction Horizon')
        axes[idx].set_xticks(x)
        axes[idx].set_xticklabels([f'{h}d' for h in horizons])
        axes[idx].set_ylim([0, 1.0])
        axes[idx].grid(True, alpha = 0.3, axis = 'y')
        
        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v + 0.02, f'{v:.3f}', ha = 'center', va = 'bottom', fontsize = 8)
    
    plt.tight_layout()
    plt.savefig(filename, bbox_inches = 'tight')
    plt.close()


def plot_comprehensive_summary(metrics_dict, filename):
    """
    Create a comprehensive summary plot with all key metrics
    """
    fig = plt.figure(figsize = (14, 8))
    gs = GridSpec(2, 3, figure = fig, hspace = 0.3, wspace = 0.3)
    
    horizons = [str(h) for h in PREDICTION_HORIZONS]
    x = np.arange(len(horizons))
    
    # 1. AUC comparison
    ax1 = fig.add_subplot(gs[0, 0])
    aucs = [metrics_dict[h]['auc'] for h in PREDICTION_HORIZONS]
    bars1 = ax1.bar(x, aucs, alpha = 0.8, color = 'steelblue', edgecolor = 'black', linewidth = 1)
    ax1.set_ylabel('AUC-ROC')
    ax1.set_title('AUC-ROC Score', fontweight = 'bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{h}d' for h in horizons])
    ax1.set_ylim([0, 1.0])
    ax1.grid(True, alpha = 0.3, axis = 'y')
    for i, v in enumerate(aucs):
        ax1.text(i, v + 0.02, f'{v:.3f}', ha = 'center', fontsize = 8)
    
    # 2. Precision vs Recall
    ax2 = fig.add_subplot(gs[0, 1])
    precisions = [metrics_dict[h]['precision'] for h in PREDICTION_HORIZONS]
    recalls = [metrics_dict[h]['recall'] for h in PREDICTION_HORIZONS]
    width = 0.35
    ax2.bar(x - width/2, precisions, width, label = 'Precision', alpha = 0.8, color = 'coral')
    ax2.bar(x + width/2, recalls, width, label = 'Recall', alpha = 0.8, color = 'lightgreen')
    ax2.set_ylabel('Score')
    ax2.set_title('Precision vs Recall', fontweight = 'bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'{h}d' for h in horizons])
    ax2.set_ylim([0, 1.0])
    ax2.legend()
    ax2.grid(True, alpha = 0.3, axis = 'y')
    
    # 3. F1-Score
    ax3 = fig.add_subplot(gs[0, 2])
    f1s = [metrics_dict[h]['f1'] for h in PREDICTION_HORIZONS]
    bars3 = ax3.bar(x, f1s, alpha = 0.8, color = 'mediumpurple', edgecolor = 'black', linewidth = 1)
    ax3.set_ylabel('F1-Score')
    ax3.set_title('F1-Score', fontweight = 'bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{h}d' for h in horizons])
    ax3.set_ylim([0, 1.0])
    ax3.grid(True, alpha = 0.3, axis = 'y')
    for i, v in enumerate(f1s):
        ax3.text(i, v + 0.02, f'{v:.3f}', ha = 'center', fontsize = 8)
    
    # 4. Accuracy
    ax4 = fig.add_subplot(gs[1, 0])
    accs = [metrics_dict[h]['accuracy'] for h in PREDICTION_HORIZONS]
    bars4 = ax4.bar(x, accs, alpha = 0.8, color = 'gold', edgecolor = 'black', linewidth = 1)
    ax4.set_ylabel('Accuracy')
    ax4.set_title('Accuracy', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{h}d' for h in horizons])
    ax4.set_ylim([0, 1.0])
    ax4.grid(True, alpha = 0.3, axis = 'y')
    for i, v in enumerate(accs):
        ax4.text(i, v + 0.02, f'{v:.3f}', ha = 'center', fontsize = 8)
    
    # 5. All metrics radar chart (if multiple horizons)
    ax5 = fig.add_subplot(gs[1, 1:], projection = 'polar')
    categories = ['AUC', 'Accuracy', 'Precision', 'Recall', 'F1']
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint = False).tolist()
    angles += angles[:1]
    
    colors_radar = plt.cm.Set3(np.linspace(0, 1, len(PREDICTION_HORIZONS)))
    for i, horizon in enumerate(PREDICTION_HORIZONS):
        values = [
            metrics_dict[horizon]['auc'],
            metrics_dict[horizon]['accuracy'],
            metrics_dict[horizon]['precision'],
            metrics_dict[horizon]['recall'],
            metrics_dict[horizon]['f1']
        ]
        values += values[:1]
        ax5.plot(angles, values, 'o-', linewidth = 2, label = f'{horizon}-day', color = colors_radar[i])
        ax5.fill(angles, values, alpha = 0.15, color = colors_radar[i])
    
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(categories)
    ax5.set_ylim(0, 1)
    ax5.set_title('Performance Metrics Overview', fontweight = 'bold', pad = 20)
    ax5.legend(loc = 'upper right', bbox_to_anchor = (1.3, 1.0))
    ax5.grid(True)
    
    plt.suptitle('GraphSAGE Model Performance Summary', fontsize = 14, fontweight = 'bold', y = 0.98)
    plt.savefig(filename, bbox_inches = 'tight')
    plt.close()