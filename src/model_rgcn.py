from torch_geometric.nn import BatchNorm, RGCNConv
import torch, torch.nn as nn
import math

class DistanceBias(nn.Module):
    """Computes distance bias for spatial attention."""
    def __init__(self, tau_km: float = 25.0):
        super().__init__()
        self.tau = float(tau_km)

    def forward(self, dist_matrix: torch.Tensor):
        # dist_matrix: (B,N,N) or (N,N)
        if dist_matrix.dim() == 2:
            dist_matrix = dist_matrix.unsqueeze(0)
        bias = -torch.clamp(dist_matrix, min=0.0) / self.tau
        return bias.unsqueeze(1)  # (B,1,N,N)


class SpatialAttention(nn.Module):
    """Multi-head attention between spatial nodes with distance bias."""
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.beta = nn.Parameter(torch.tensor(0.5))  # balance content vs distance

    def forward(self, x: torch.Tensor, bias: torch.Tensor):
        # x: (B, N, C), bias: (B, 1, N, N)
        B, N, C = x.shape
        h = self.norm(x)
        
        # Fused QKV projection
        qkv = self.qkv_proj(h)  # (B, N, 3*C)
        q, k, v = qkv.chunk(3, dim=-1)  # (B, N, C) each
        
        # Reshape for multi-head: (B, N, C) -> (B, num_heads, N, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores: (B, num_heads, N, N)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        
        # Add distance bias (broadcast across heads)
        scores = scores + self.beta * bias  # bias: (B, 1, N, N)
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention: (B, num_heads, N, head_dim)
        out = torch.matmul(attn, v)
        
        # Reshape back: (B, num_heads, N, head_dim) -> (B, N, C)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.out_proj(out)
        
        return x + self.dropout(out)

    def forward(self, x, bias, spatial_graph_size):
        B, N, C = x.shape
        assert N % spatial_graph_size == 0, "N must be divisible by block size"

        num_blocks = N // spatial_graph_size

        # reshape into blocks
        x = x.view(B, num_blocks, spatial_graph_size, C)

        # normalize
        h = self.norm(x)

        # QKV
        qkv = self.qkv_proj(h)                      # (B, num_blocks, M, 3C)
        q, k, v = qkv.chunk(3, dim=-1)

        # multi-head: (B, num_blocks, M, head_dim)
        q = q.view(B, num_blocks, spatial_graph_size, self.num_heads, self.head_dim).transpose(2, 3)
        k = k.view(B, num_blocks, spatial_graph_size, self.num_heads, self.head_dim).transpose(2, 3)
        v = v.view(B, num_blocks, spatial_graph_size, self.num_heads, self.head_dim).transpose(2, 3)
        # shapes now: (B, num_blocks, num_heads, M, head_dim)

        # attention scores inside each block only
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        # scores: (B, num_blocks, num_heads, M, M)
        # bias: (B, 1, M, M)
        bias_blocks = bias.expand(B, num_blocks, spatial_graph_size, spatial_graph_size)      # (B, num_blocks, M, M)
        bias_blocks = bias_blocks.unsqueeze(2)              # (B, num_blocks, 1, M, M)
        bias_blocks = bias_blocks.expand(B, num_blocks, self.num_heads, spatial_graph_size, spatial_graph_size)  # (B, num_blocks, num_heads, M, M)
        scores = scores + self.beta * bias_blocks

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # apply attention: (B, num_blocks, num_heads, M, head_dim)
        out = torch.matmul(attn, v)

        # reshape back to (B, N, C)
        out = out.transpose(2, 3).contiguous().view(B, N, C)

        out = self.out_proj(out)

        return x.view(B, N, C) + self.dropout(out)


class RGCN(nn.Module):
    """
    RGCNConv based HeteroGNN model applied to temporal snapshot graph.
    """
    def __init__(self, in_channels, num_layers = 7, hidden_dim = 256, out_dim = 128, n_horizons = 3, dropout = 0.4, num_spatial_att_heads = 2, distance_matrix = None, use_regression_task = False, use_loading_rate = False, use_spatial_edges = False, use_spatial_attention = True):
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        self.use_regression_task = use_regression_task

        self.use_loading_rate = use_loading_rate
        self.use_spatial_edges = use_spatial_edges
        self.use_spatial_attention = use_spatial_attention

        num_relations = 1 + self.use_loading_rate + self.use_spatial_edges

        if self.use_spatial_edges and self.use_spatial_attention:
            print("WARNING: using USE_SPATIAL_EDGES and USE_SPATIAL ATTENTION are both true. You probably don't want this.")

        self.conv1 = RGCNConv(in_channels, hidden_dim, num_relations)
        self.bn1 = BatchNorm(hidden_dim)
        self.conv_layers = [RGCNConv(hidden_dim, hidden_dim, num_relations) for _ in range(num_layers - 1)]
        self.bn_layers = [BatchNorm(hidden_dim) for _ in range(num_layers - 1)]
        
        if self.use_spatial_attention:
            if distance_matrix == None:
                raise RuntimeError("no distance matrix provided for spatial attention")
            self.spatial_attention = SpatialAttention(hidden_dim, num_spatial_att_heads)
            self.distance_bias = DistanceBias()
            self.distance_matrix = distance_matrix

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

        # only if using recurrence time regression task
        self.regression_head = nn.Sequential(
                nn.Linear(out_dim, out_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(out_dim * 4, 1)
        )
    
    def forward(self, hetero_data):
        homogeneous_data = hetero_data.to_homogeneous()

        edge_index = homogeneous_data["edge_index"]

        edge_type = homogeneous_data["edge_type"]

        x = homogeneous_data.x

        x = self.conv1(x, edge_index, edge_type)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout(x)

        if self.use_spatial_attention:
            dist_bias = self.distance_bias(self.distance_matrix)

        for i in range(self.num_layers - 1):
            if self.use_spatial_attention:
                num_graphs = 1 if not hasattr(hetero_data, "num_graphs") else hetero_data.num_graphs
                graph_size = int(x.shape[0] / num_graphs)
                # hacky way to deal with batches
                context_length = hetero_data.context_length if isinstance(hetero_data.context_length, int) else int(hetero_data.context_length[0])
                spatial_graph_size = graph_size // context_length
                x = x.view(num_graphs, graph_size, self.hidden_dim)
                x = self.spatial_attention(x, dist_bias, spatial_graph_size)
                x = x.view(num_graphs * graph_size, self.hidden_dim)

            conv = self.conv_layers[i]
            bn = self.bn_layers[i]
            x = x + conv(x, edge_index, edge_type) # skip connection
            x = bn(x)
            x = self.activation(x)
            x = self.dropout(x)
    
        prediction_nodes = x[homogeneous_data.node_predict, :]

        # Multi-task predictions
        if self.use_regression_task:
            outputs = self.regression_head(prediction_nodes)
        else:
            outputs = [head(prediction_nodes).squeeze(-1) for head in self.heads]
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