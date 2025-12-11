import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import deepsnap.hetero_gnn
from torch_geometric.utils import matmul
from deepsnap.hetero_gnn import forward_op


class HeteroGNNConv(pyg_nn.MessagePassing):
    """
    Heterogeneous GNN convolution layer
    """
    def __init__(self, in_channels_src, in_channels_dst, out_channels):
        super(HeteroGNNConv, self).__init__(aggr="sum")

        self.in_channels_src = in_channels_src
        self.in_channels_dst = in_channels_dst
        self.out_channels = out_channels

        self.lin_dst = None
        self.lin_src = None
        self.lin_update = None

        self.lin_dst = nn.Linear(in_channels_dst, out_channels)
        self.lin_src = nn.Linear(in_channels_src, out_channels)
        self.lin_update = nn.Linear(2 * out_channels, out_channels)


    def forward(
        self,
        node_feature_src,
        node_feature_dst,
        edge_index,
        size=None
    ):
        return self.propagate(edge_index, node_feature_src=node_feature_src, node_feature_dst=node_feature_dst)


    def message_and_aggregate(self, edge_index, node_feature_src):

        out = None
        out = matmul(edge_index, node_feature_src, reduce = self.aggr)
        out = self.lin_src(out)

        return out

    def update(self, aggr_out, node_feature_dst):

        node_feature_dst = self.lin_dst(node_feature_dst)
        aggr_out = torch.concat((node_feature_dst, aggr_out), dim = 1)
        aggr_out = self.lin_update(aggr_out)

        return aggr_out


class HeteroGNNWrapperConv(deepsnap.hetero_gnn.HeteroConv):
    """
    Heterogeneous GNN wrapper convolution layer
    """
    def __init__(self, convs, args, aggr="sum"):
        super(HeteroGNNWrapperConv, self).__init__(convs, None)
        self.aggr = aggr

        # Map the index and message type
        self.mapping = {}

        # A numpy array that stores the final attention probability
        self.alpha = None

        self.attn_proj = None

        if self.aggr == "attn":

            self.attn_proj = nn.Sequential(
                nn.Linear(args['hidden_size'], args['attn_size']),
                nn.Tanh(),
                nn.Linear(args['attn_size'], 1, bias = False)
            )

    def reset_parameters(self):
        super(HeteroConvWrapper, self).reset_parameters()
        if self.aggr == "attn":
            for layer in self.attn_proj.children():
                layer.reset_parameters()

    def forward(self, node_features, edge_indices):
        message_type_emb = {}
        for message_key, message_type in edge_indices.items():
            src_type, edge_type, dst_type = message_key
            node_feature_src = node_features[src_type]
            node_feature_dst = node_features[dst_type]
            edge_index = edge_indices[message_key]
            message_type_emb[message_key] = (
                self.convs[message_key](
                    node_feature_src,
                    node_feature_dst,
                    edge_index,
                )
            )
        node_emb = {dst: [] for _, _, dst in message_type_emb.keys()}
        mapping = {}
        for (src, edge_type, dst), item in message_type_emb.items():
            mapping[len(node_emb[dst])] = (src, edge_type, dst)
            node_emb[dst].append(item)
        self.mapping = mapping
        for node_type, embs in node_emb.items():
            if len(embs) == 1:
                node_emb[node_type] = embs[0]
            else:
                node_emb[node_type] = self.aggregate(embs)
        return node_emb

    def aggregate(self, xs):

        if self.aggr == "sum":
            return torch.sum(torch.stack(xs, dim = 0), dim = 0)

        elif self.aggr == "attn":
            N = xs[0].shape[0] # Number of nodes for that node type
            M = len(xs) # Number of message types for that node type

            x = torch.cat(xs, dim=0).view(M, N, -1) # M * N * D
            z = self.attn_proj(x).view(M, N) # M * N * 1
            z = z.mean(1) # M * 1
            alpha = torch.softmax(z, dim=0) # M * 1

            # Store the attention result to self.alpha as np array
            self.alpha = alpha.view(-1).data.cpu().numpy()

            alpha = alpha.view(M, 1, 1)
            x = x * alpha
            return x.sum(dim=0)


def generate_convs(hetero_graph, conv, hidden_size, first_layer=False):
    """
    Returns a dictionary of `HeteroGNNConv`
    layers where the keys are message types. `hetero_graph` is deepsnap `HeteroGraph`
    object and the `conv` is the `HeteroGNNConv`.
    """
    convs = {}

    for message_type in hetero_graph.message_types:
        in_channels_src = hidden_size
        in_channels_dst = hidden_size
        convs[message_type] = conv(in_channels_src, in_channels_dst, hidden_size)

    return convs


class HeteroGNN(torch.nn.Module):
    """
    Heterogeneous GNN model
    """
    def __init__(self, hetero_graph, args, aggr="sum"):
        super(HeteroGNN, self).__init__()

        self.aggr = aggr
        self.hidden_size = args['hidden_size']
        self.pre_mlp_spatial = nn.Sequential(
            nn.Linear(hetero_graph.num_node_features('spatial'), self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        self.pre_mlp_time = nn.Sequential(
            nn.Linear(hetero_graph.num_node_features('time'), self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

        self.convs1 = None
        self.convs2 = None

        self.bns1 = nn.ModuleDict()
        self.bns2 = nn.ModuleDict()
        self.bns3 = nn.ModuleDict()
        self.bns4 = nn.ModuleDict()
        self.relus1 = nn.ModuleDict()
        self.relus2 = nn.ModuleDict()
        self.relus3 = nn.ModuleDict()
        self.relus4 = nn.ModuleDict()
        
        self.w1 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.w2 = torch.nn.Linear(self.hidden_size, self.hidden_size)

        convs1_dict = generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size, True)
        self.convs1 = HeteroGNNWrapperConv(convs1_dict, args, self.aggr)
        convs2_dict = generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size, False)
        self.convs2 = HeteroGNNWrapperConv(convs2_dict, args, self.aggr)
        convs3_dict = generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size, False)
        self.convs3 = HeteroGNNWrapperConv(convs3_dict, args, self.aggr)
        convs4_dict = generate_convs(hetero_graph, HeteroGNNConv, self.hidden_size, False)
        self.convs4 = HeteroGNNWrapperConv(convs4_dict, args, self.aggr)
        for node_type in hetero_graph.node_types:
            self.bns1[node_type] = nn.BatchNorm1d(self.hidden_size, eps = 1)
            self.bns2[node_type] = nn.BatchNorm1d(self.hidden_size, eps = 1)
            self.bns3[node_type] = nn.BatchNorm1d(self.hidden_size, eps = 1)
            self.bns4[node_type] = nn.BatchNorm1d(self.hidden_size, eps = 1)
            self.relus1[node_type] = nn.PReLU()
            self.relus2[node_type] = nn.PReLU()
            self.relus3[node_type] = nn.PReLU()
            self.relus4[node_type] = nn.PReLU()
        
        self.post_mlp = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.PReLU(),
            nn.Linear(self.hidden_size, self.hidden_size)
        )

    def forward(self, node_feature, edge_index, target_edges):
        x = node_feature.copy()

        # Pre MLPs
        x['spatial'] = self.pre_mlp_spatial(x['spatial'])
        x['time'] = self.pre_mlp_time(x['time'])
        
        # Store initial features for skip connections
        x_input = {key: val.clone() for key, val in x.items()}
        
        ############# Layer 1 #############
        x = self.convs1(edge_indices=edge_index, node_features=x)
        x = forward_op(x=x, module_dict=self.bns1)  # or layer norms
        
        for key in x.keys():
            x[key] = x[key] + x_input[key]

        x = forward_op(x=x, module_dict=self.relus1)
    
        
        # Store for next skip connection
        x_layer1 = {key: val.clone() for key, val in x.items()}
        
        ############# Layer 2 #############
        x = self.convs2(edge_indices=edge_index, node_features=x)
        x = forward_op(x=x, module_dict=self.bns2)
        
        for key in x.keys():
            x[key] = x[key] + x_layer1[key] + x_input[key]
        
        x = forward_op(x=x, module_dict=self.relus2)
        
        # Store for next skip connection
        x_layer2 = {key: val.clone() for key, val in x.items()}
        
        ############# Layer 3 #############
        x = self.convs3(edge_indices=edge_index, node_features=x)
        x = forward_op(x=x, module_dict=self.bns3)
        
        for key in x.keys():
            x[key] = x[key] + x_layer2[key] + x_layer1[key] + x_input[key]
        x = forward_op(x=x, module_dict=self.relus3)

        # Store for next skip connection
        x_layer3 = {key: val.clone() for key, val in x.items()}
        
        ############# Layer 4 #############
        x = self.convs4(edge_indices=edge_index, node_features=x)
        x = forward_op(x=x, module_dict=self.bns4)
        
        for key in x.keys():
            x[key] = x[key] + x_layer3[key] + x_layer2[key] + x_layer1[key] + x_input[key]
        x = forward_op(x=x, module_dict=self.relus4)

        # post MLP
        for key in x.keys():
            x[key] = self.post_mlp(x[key])
        
        
        ############# Edge Prediction #############
        time_indices = target_edges[0]
        spatial_indices = target_edges[1]
        
        x_time = x['time'][time_indices]
        x_spatial = x['spatial'][spatial_indices]
        
        return (F.cosine_similarity(self.w1(x_time), self.w2(x_spatial), dim=1).view(-1, 1))