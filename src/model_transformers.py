import torch
import torch.nn as nn
import torch.nn.functional as F
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


class SpatioTemporalBlock(nn.Module):
    """One block of spatio-temporal message passing
    temporal->spatial aggregation (only for first block)
    spatial self-attention + feed-forward fusion
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1, use_temporal: bool = False):
        super().__init__()
        self.use_temporal = use_temporal
        if use_temporal:
            self.temporal_to_spatial = TemporalToSpatialAttention(hidden_dim, num_heads, dropout)
            # projection to map concatenated [spatial, temporal_msg] (2*C) -> C
            self.proj_ln = nn.LayerNorm(hidden_dim * 2)
            self.proj_lin = nn.Linear(hidden_dim * 2, hidden_dim)

        self.spatial_attn = SpatialAttention(hidden_dim, num_heads, dropout)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, h_spatial: torch.Tensor, h_temporal: torch.Tensor, dist_bias: torch.Tensor):

        if self.use_temporal:
            t_msg = self.temporal_to_spatial(h_spatial, h_temporal)
            # fuse spatial identity with temporal message
            h = torch.cat([h_spatial, t_msg], dim=-1)
            # project back to hidden dim using registered layers
            h = self.proj_ln(h)
            h = self.proj_lin(h)
        else:
            h = h_spatial

        # spatial attention
        h = self.spatial_attn(h, dist_bias)

        h = h + self.ffn(h)
        return h


class TemporalToSpatialAttention(nn.Module):
    """
    Multi-head message passing from temporal nodes to their corresponding spatial node
    Each spatial node attends over its L temporal nodes to aggregate history
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.kv_proj = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, spatial_nodes: torch.Tensor, temporal_nodes: torch.Tensor):
        B, N, C = spatial_nodes.shape
        L = temporal_nodes.shape[2]
        
        # Query from spatial nodes
        q = self.q_proj(spatial_nodes)  # (B, N, C)
        
        # Keys and values from temporal nodes (fused)
        kv = self.kv_proj(temporal_nodes)  # (B, N, L, 2*C)
        k, v = kv.chunk(2, dim=-1)  # (B, N, L, C) each
        
        # Reshape for multi-head
        # q: (B, N, C) -> (B, N, num_heads, head_dim) -> (B, num_heads, N, 1, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2).unsqueeze(3)
        # k, v: (B, N, L, C) -> (B, N, L, num_heads, head_dim) -> (B, num_heads, N, L, head_dim)
        k = k.view(B, N, L, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        v = v.view(B, N, L, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        
        # Attention: (B, num_heads, N, 1, head_dim) @ (B, num_heads, N, head_dim, L) -> (B, num_heads, N, 1, L)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Weighted sum: (B, num_heads, N, 1, L) @ (B, num_heads, N, L, head_dim) -> (B, num_heads, N, 1, head_dim)
        out = torch.matmul(attn, v).squeeze(3)  # (B, num_heads, N, head_dim)
        
        # Reshape back: (B, num_heads, N, head_dim) -> (B, N, C)
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.out_proj(out)
        
        return out


class HeterogeneousSpatioTemporalTransformer(nn.Module):
    """
    Heterogeneous graph with two node types:
    1. Spatial nodes: One per fault, feature = fault embedding
    2. Temporal nodes: L per fault (one per time step), features = [count, max_mag, months_ago]
    
    Message passing:
    1. Temporal nodes -> Spatial nodes each fault aggregates its history
    2. Spatial nodes <-> Spatial nodes faults share info with distance-biased attention
    3. Predict from spatial nodes
    """
    def __init__(self, num_nodes: int, temporal_feat_dim: int = 3, hidden_dim: int = 64,
                 num_heads: int = 4, num_layers: int = 2, num_horizons: int = 3, dropout: float = 0.1, tau_km: float = 25.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.num_layers = num_layers

        # Spatial node embedding (learnable identity per fault)
        self.spatial_embedding = nn.Embedding(num_nodes, hidden_dim)

        # Temporal node encoder (project temporal features to hidden dim)
        self.temporal_encoder = nn.Sequential(
            nn.Linear(temporal_feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Temporal position encoding (which month in the sequence)
        self.temporal_pos_encoding = nn.Embedding(100, hidden_dim)  # up to 100 months

        # Distance bias for spatial attention
        self.distance_bias = DistanceBias(tau_km=tau_km)

        # Build stacked spatiotemporal blocks
        layers = []
        for i in range(num_layers):
            use_temporal = (i == 0)
            layers.append(SpatioTemporalBlock(hidden_dim, num_heads, dropout, use_temporal=use_temporal))
        self.layers = nn.ModuleList(layers)

        # Final norm
        self.final_norm = nn.LayerNorm(hidden_dim)

        # Prediction heads (one per horizon)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_horizons)
        ])

    def forward(self, x_spatial: torch.Tensor, x_temporal: torch.Tensor, dist_matrix: torch.Tensor):
        if x_spatial.dim() == 3:
            x_spatial = x_spatial.squeeze(-1)
        
        B, N, L, C = x_temporal.shape
        
        # Get spatial node embeddings (B, N, hidden)
        h_spatial = self.spatial_embedding(x_spatial.long())
        
        # Encode temporal features (B, N, L, hidden)
        h_temporal = self.temporal_encoder(x_temporal)
        
        # Add positional encoding for temporal order
        positions = torch.arange(L, device=x_temporal.device)
        pos_enc = self.temporal_pos_encoding(positions)  # (L, hidden)
        h_temporal = h_temporal + pos_enc.unsqueeze(0).unsqueeze(0)  # broadcast to (B, N, L, hidden)
        
        # distance bias
        dist_bias = self.distance_bias(dist_matrix)  # (B, 1, N, N)

        h = h_spatial
        # pass through stacked layers
        for layer in self.layers:
            h = layer(h, h_temporal, dist_bias)

        # final norm and heads
        h = self.final_norm(h)
        out = torch.stack([head(h).squeeze(-1) for head in self.heads], dim=-1)
        return out
