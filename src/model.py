from torch_geometric.nn import SAGEConv, BatchNorm
from torch_geometric.nn import RGCNConv, to_hetero
import torch, torch.nn as nn

class GraphSAGE(nn.Module):
    """
    Graph neural network for predicting multiple time horizons
    """
    def __init__(self, in_channels, hidden_dim = 256, out_dim = 128, n_horizons = 3, dropout = 0.4):
        super().__init__()
        
        self.conv1 = SAGEConv(in_channels, hidden_dim, normalize = True)
        self.bn1 = BatchNorm(hidden_dim)
        
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, normalize = True)
        self.bn2 = BatchNorm(hidden_dim)
        
        self.conv3 = SAGEConv(hidden_dim, hidden_dim, normalize = True)
        self.bn3 = BatchNorm(hidden_dim)
        
        self.conv4 = SAGEConv(hidden_dim, out_dim, normalize = True)
        self.bn4 = BatchNorm(out_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        # Multi-task prediction heads (one per horizon)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_dim, out_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim * 4, 1)
            ) for _ in range(n_horizons)
        ])
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.conv4(x, edge_index)
        x = self.bn4(x)
        
        
        # Multi-task predictions
        outputs = [head(x).squeeze(-1) for head in self.heads]
        outputs = torch.stack(outputs, dim = 1) 
        
        return outputs


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    def __init__(self, alpha = 0.75, gamma = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction = 'none')
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_weight * focal_weight * bce_loss
        return loss.mean()
    

class RGCN(nn.Module):
    """
    Graph neural network for predicting multiple time horizons
    """
    def __init__(self, in_channels, hidden_dim = 256, out_dim = 128, n_horizons = 3, dropout = 0.4):
        super().__init__()
        
        self.conv1 = RGCNConv(in_channels, hidden_dim, normalize = True)
        self.bn1 = BatchNorm(hidden_dim)
        
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, normalize = True)
        self.bn2 = BatchNorm(hidden_dim)
        
        self.conv3 = RGCNConv(hidden_dim, hidden_dim, normalize = True)
        self.bn3 = BatchNorm(hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
        # Multi-task prediction heads (one per horizon)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(out_dim, out_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim * 4, 1)
            ) for _ in range(n_horizons)
        ])
    
    def forward(self, hetero_data):
        
        homogeneous_data = hetero_data.to_homogenous()


        edge_index = homogeneous_data["edge_index"]

        edge_type = homogeneous_data["edge_Type"]

        x = hetero_data["earthquake_source"].x
        # edge_index_temporal = hetero_data[("earthquake_source", "temporal", "earthquake_source")]
        # edge_index_spatial = hetero_data[("earthquake_source", "spatial", "earthquake_source")]
        x = self.conv1(x, edge_index, edge_type)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_type)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index, edge_type)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.dropout(x)
        
        # Multi-task predictions
        outputs = [head(x).squeeze(-1) for head in self.heads]
        outputs = torch.stack(outputs, dim = 1) 
        
        return outputs


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    def __init__(self, alpha = 0.75, gamma = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction = 'none')
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_weight * focal_weight * bce_loss
        return loss.mean()