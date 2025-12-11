from torch_geometric.nn import HeteroConv, TransformerConv
import torch.nn as nn
import torch.nn.functional as F
import torch

# Graph Transformer Model for Heterogeneous Graphs
class GraphTransformer(torch.nn.Module):
    def __init__(self, spatial_feat_dim=4, temporal_feat_dim=3, 
                 hidden_dim=64, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        
        # 1. Encoders
        self.spatial_enc = nn.Linear(spatial_feat_dim, hidden_dim)
        self.temporal_enc = nn.Linear(temporal_feat_dim, hidden_dim)
        
        # 2. Transformer Body
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {
                # Spatial-Spatial: Dense connection, weighted by Haversine distance
                # edge_dim=1 tells the layer to look for the edge weight
                ('spatial', 'nearby', 'spatial'): TransformerConv(
                    hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                    dropout=dropout, edge_dim=1
                ),
                
                # Time-Spatial: Sparse connection, weighted by Magnitude
                ('time', 'event', 'spatial'): TransformerConv(
                    hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                    dropout=dropout, edge_dim=1
                ),
                
                # Time-Time: Sparse connection, weight is 1.0
                ('time', 'past', 'time'): TransformerConv(
                    hidden_dim, hidden_dim // num_heads, heads=num_heads, 
                    dropout=dropout, edge_dim=1
                )
            }
            # 'sum' aggregation: Sums up the attention results from different edge types
            self.layers.append(HeteroConv(conv_dict, aggr='sum'))

        # 3. Prediction Head
        self.predict_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), 
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, node_feature, edge_index, target_edges, edge_attr_dict=None):
        # 1. Embed Features
        x = {
            'spatial': self.spatial_enc(node_feature['spatial']),
            'time': self.temporal_enc(node_feature['time'])
        }
        
        # 2. Transformer Layers
        for conv in self.layers:
            x = conv(x, edge_index, edge_attr_dict=edge_attr_dict)
            # Non-linearity and Residuals
            x = {key: F.relu(val) for key, val in x.items()}

        # 3. Predict on Target Edges
        time_idx = target_edges[0]
        space_idx = target_edges[1]
        
        feat_t = x['time'][time_idx]
        feat_s = x['spatial'][space_idx]
        
        # Concatenate and pass through prediction head
        combined = torch.cat([feat_t, feat_s], dim=1)
        return self.predict_head(combined)

