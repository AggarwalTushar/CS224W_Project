import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from collections import defaultdict
from config import DIST_THRESHOLD_KM, LOOKBACK_DAYS, PREDICTION_HORIZONS, USE_LOADING_RATE, USE_RECURRENCE_TIME_TASK

def haversine(lon1, lat1, lon2, lat2):
    "Calculates distance between two coordinates"
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c


def load_and_prepare_data(path):
    """Load and convert to datetime"""
    df = pd.read_excel(path)
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Convert time to datetime so that it would be easy to do operations
    if np.issubdtype(df["event_time"].dtype, np.number):
        # Assume it's years since 2000
        ref = pd.Timestamp("2000-01-01")
        df["datetime"] = [ref + pd.DateOffset(months=int(y*12)) for y in df["event_time"].values]
    else:
        df["datetime"] = pd.to_datetime(df["event_time"], errors="coerce")
    
    df = df.sort_values("datetime").reset_index(drop=True)
    print(f"Loaded {len(df)} events from {df['datetime'].min()} to {df['datetime'].max()}")
    return df


def build_edge_index(df, dist_thresh_km = DIST_THRESHOLD_KM):
    """Build edge_index in COO format for PyG"""
    nodes = sorted(df["fault_radius"].unique())
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    N = len(nodes)
    
    # Get node locations
    node_coords = df.groupby("fault_radius").agg({
        "latitude": "mean",
        "longitude": "mean"
    })
    
    # Build edge list
    edge_list = []
    adj_list = defaultdict(list)
    distances = np.zeros((N, N))
    
    for i, ni in enumerate(nodes):
        for j, nj in enumerate(nodes):
            if i != j:
                dist = haversine(
                    node_coords.loc[ni, "longitude"], node_coords.loc[ni, "latitude"],
                    node_coords.loc[nj, "longitude"], node_coords.loc[nj, "latitude"]
                )
                distances[i, j] = dist
                if dist <= dist_thresh_km:
                    edge_list.append([i, j])
                    adj_list[i].append(j)
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    
    print(f"Built graph: {N} nodes, {edge_index.shape[1]} edges (avg degree: {edge_index.shape[1]/N:.1f})")
    return nodes, node_to_idx, edge_index


def extract_node_features(hist_events, lookback_days):
    """Extract features from historical events at a single node"""
    features = []
    
    # 1. Basic counts and recency (4 features)
    features.append(len(hist_events))
    features.append(hist_events["days_ago"].min() if len(hist_events) > 0 else lookback_days)
    features.append(hist_events["days_ago"].max() if len(hist_events) > 0 else lookback_days)
    features.append(hist_events["days_ago"].mean() if len(hist_events) > 0 else lookback_days)
    
    # 2. Magnitude statistics (5 features)
    mags = hist_events["magnitude"].values
    mags = mags[~np.isnan(mags)]  # Remove NaN magnitudes
    if len(mags) > 0:
        features.extend([
            np.nanmean(mags),
            np.nanstd(mags) if len(mags) > 1 else 0,
            np.nanmax(mags),
            np.nanmin(mags),
            np.nanpercentile(mags, 75)
        ])
    else:
        features.extend([0, 0, 0, 0, 0])
    
    # 3. Temporal patterns - divide into time bins (3 features)
    bins = [0, 30, 90, lookback_days]
    for i in range(len(bins)-1):
        count_in_bin = ((hist_events["days_ago"] >= bins[i]) & 
                       (hist_events["days_ago"] < bins[i+1])).sum()
        features.append(count_in_bin)
    
    # 5. Gutenberg-Richter b-value (1 feature)
    if len(mags) > 3:
        mag_range = np.nanmax(mags) - np.nanmin(mags)
        if mag_range > 0.1:
            b_value = np.log10(np.e) / (np.nanmean(mags) - np.nanmin(mags) + 0.1)
            features.append(min(b_value, 5.0))  # Cap at reasonable value
        else:
            features.append(1.0)  # Default b-value
    else:
        features.append(0)
    
    # 6. Activity rate (events per month) (1 feature)
    features.append(len(hist_events) / (lookback_days / 30))
    
    # Convert to array and replace any remaining NaN with 0
    features = np.array(features, dtype = np.float32)
    features = np.nan_to_num(features, nan = 0.0, posinf = 0.0, neginf = 0.0)
    
    return features


def build_temporal_graphs(df, nodes, node_to_idx, edge_index, lookback_days = LOOKBACK_DAYS):
    """
    Build ONE graph per time window, with features for ALL nodes
    Each graph contains predictions for all active nodes at that time
    """
    df = df.copy()
    df["node_idx"] = df["fault_radius"].map(node_to_idx)
    n_nodes = len(nodes)
    
    # Group events by node
    node_events = {i: df[df["node_idx"] == i].copy() for i in range(n_nodes)}
    
    
    # Determine time range
    all_dates = df["datetime"].values
    min_date = pd.Timestamp(all_dates.min()) + pd.Timedelta(days = lookback_days)
    max_date = pd.Timestamp(all_dates.max()) - pd.Timedelta(days = max(PREDICTION_HORIZONS))
    
    # Create time windows (every 30 days)
    sample_dates = pd.date_range(min_date, max_date, freq = "30D")
    print(f"Creating {len(sample_dates)} time windows")
    
    data_list = []
    
    for sample_date in sample_dates:
        hist_start = sample_date - pd.Timedelta(days = lookback_days)
        
        # Build features for ALL nodes at this time window
        x = np.zeros((n_nodes, 14), dtype = np.float32) - 1 
        y = np.zeros((n_nodes, len(PREDICTION_HORIZONS)), dtype=np.float32)
        node_mask = np.ones(n_nodes, dtype = bool)
        
        for node_idx in range(n_nodes):
            if node_idx not in node_events:
                continue
                
            events = node_events[node_idx]
            
            # Get historical events
            hist_events = events[(events["datetime"] >= hist_start) & (events["datetime"] < sample_date)]
            
            # Only compute features if there's historical data
            if len(hist_events) > 0:
                hist_events = hist_events.copy()
                hist_events["days_ago"] = (sample_date - hist_events["datetime"]).dt.total_seconds() / 86400
                
                # Extract features
                x[node_idx] = extract_node_features(hist_events, lookback_days)
                
                # Compute targets for multiple horizons
                for h_idx, horizon in enumerate(PREDICTION_HORIZONS):
                    future_start = sample_date
                    future_end = sample_date + pd.Timedelta(days = horizon)
                    future_events = events[(events["datetime"] >= future_start) & (events["datetime"] < future_end)]
                    y[node_idx, h_idx] = float(len(future_events) > 0)
        
        data = Data(
            x = torch.tensor(x, dtype=torch.float32),
            edge_index = edge_index.clone(),
            y = torch.tensor(y, dtype=torch.float32),
            node_mask = torch.tensor(node_mask, dtype=torch.bool),
            sample_date = sample_date,
            num_active_nodes = int(node_mask.sum())
        )
        data_list.append(data)
    
    print(f"Generated {len(data_list)} temporal graphs")
    print(f"Average active nodes per graph: {np.mean([d.num_active_nodes for d in data_list]):.1f}")
    
    return data_list

from torch_geometric.utils import subgraph
from torch_geometric.data import HeteroData
def get_time_window_subgraph(hetero_data, start_time, context_length):
    mask = (hetero_data["earthquake_source"].t >= start_time) & (hetero_data["earthquake_source"].t < (start_time + context_length))
    node_idx = torch.nonzero(mask).view(-1)
    subset_dict = {
        "earthquake_source": node_idx,
    }
    subgraph_hetero_sample = hetero_data.subgraph(subset_dict)
    # subgraph_sample = HeteroData()
    # subgraph_sample['earthquake_source'].x = hetero_data['earthquake_source'].x[node_idx]
    # subgraph_sample['earthquake_source'].t = hetero_data['earthquake_source'].t[node_idx]
    predict_nodes_mask = subgraph_hetero_sample['earthquake_source'].t == start_time + context_length - 1
    # predict_nodes = np.nonzero(predict_nodes_mask)
    # nonzero = np.count_nonzero(predict_nodes_mask)
    subgraph_hetero_sample['earthquake_source'].node_predict = predict_nodes_mask
    subgraph_hetero_sample['earthquake_source'].y = torch.tensor(hetero_data['earthquake_source'].y[node_idx][predict_nodes_mask])
    subgraph_hetero_sample.y = subgraph_hetero_sample['earthquake_source'].y
    # subgraph_sample.y = subgraph_sample['earthquake_source'].y

    # node_idx_lr = np.arange(hetero_data["loading_rate"].num_nodes)

    # for edge_type in hetero_data.edge_types:
    #     if edge_type[0] == edge_type[2]:
    #         edge_index, edge_mask = subgraph(node_idx, hetero_data[edge_type].edge_index, relabel_nodes=True)
    #         subgraph_sample[edge_type].edge_index = edge_index
    return subgraph_hetero_sample


def build_temporal_snapshot_graph(df, CONTEXT_LENGTH = 6, ):
    unique_nodes = df["fault_radius"].unique()
    N = len(unique_nodes)

    import math
    import numpy as np
    group_by_nodes = df.groupby("fault_radius")
    num_nodes = len(group_by_nodes)
    latest_time = math.ceil(max(group_by_nodes.aggregate("max")["event_time"])) * 12
    fault_radii = list(group_by_nodes.count().index)

    if USE_RECURRENCE_TIME_TASK:
        node_to_event_labels = np.zeros((num_nodes, latest_time), dtype=float)
        label_dim = 1
    else:
        node_to_event_labels = np.empty((num_nodes, latest_time, len(PREDICTION_HORIZONS)))
        label_dim = len(PREDICTION_HORIZONS)
    node_to_time_since_last = np.zeros((num_nodes, latest_time))
    node_to_time_to_next = np.zeros((num_nodes, latest_time), dtype=float) - 1
    node_to_events_per_month = np.zeros((num_nodes, latest_time))
    node_id = 0

    loading_rate_nodes = np.zeros((num_nodes, 1)) # all constant lr for now
    for node, group_df in group_by_nodes:
        event_labels = np.zeros((latest_time, len(PREDICTION_HORIZONS)))
        event_times_months = group_df["event_time"] * 12
        loading_rate = group_df["loading_rate"].iloc[-1] # all constant for now
        loading_rate_nodes[node_id][0] = loading_rate
        prev_event_time = 0
        for event_time in event_times_months:
            # set time since last
            start_id = math.floor(prev_event_time) + 1
            end_id = math.floor(event_time) + 1
            node_to_time_since_last[node_id, start_id:end_id] = np.arange(0, end_id - start_id)

            if USE_RECURRENCE_TIME_TASK:
                # set remaining recurrence time
                start_id_label = math.floor(prev_event_time)
                end_id_label = math.floor(event_time)
                node_to_time_to_next[node_id, start_id_label:end_id_label] = np.arange(event_time - start_id_label, event_time - end_id_label, -1, dtype=float)
            else:
                # set horizon labels
                for horizon_idx, pred_horizon in enumerate(PREDICTION_HORIZONS):
                    max_offset = -(int(pred_horizon / 30) - 1)
                    for offset in range(0, max_offset - 1, -1):
                        curr_time_idx = math.floor(event_time)
                        if curr_time_idx + offset >= 0:
                            event_labels[curr_time_idx + offset, horizon_idx] = 1

            prev_event_time = event_time
        # set monthly events
        for j in range(CONTEXT_LENGTH, event_labels.shape[0]):
            events_stream_for_node = event_labels[j-CONTEXT_LENGTH:j, 0]
            node_to_events_per_month[node_id, j] = np.mean(events_stream_for_node)

        if USE_RECURRENCE_TIME_TASK:
            node_to_event_labels[node_id] = node_to_time_to_next[node_id]
        else:
            node_to_event_labels[node_id] = event_labels
        node_id += 1

    hetero_data = HeteroData()
    hetero_data["earthquake_source"].y = torch.tensor(node_to_event_labels.reshape((num_nodes * latest_time, label_dim), order="F"), dtype=torch.float32)
    fault_radii_feat = torch.tensor(fault_radii * latest_time).unsqueeze(-1)
    # fault_radii_feat /= fault_radii_feat.norm()
    # labels_as_feat = hetero_data["earthquake_source"].y[:, 0].unsqueeze(-1)
    time_since_last_feat = torch.tensor(node_to_time_since_last.flatten(order="F")).unsqueeze(-1)
    events_per_month_feat = torch.tensor(node_to_events_per_month.flatten(order="F")).unsqueeze(-1)
    features = torch.hstack([fault_radii_feat, time_since_last_feat, events_per_month_feat]).float()
    hetero_data["earthquake_source"].x = features

    # hetero_data["earthquake_source"].x = hetero_data["earthquake_source"].x / hetero_data["earthquake_source"].x.norm()
    hetero_data["earthquake_source"].t = torch.arange(latest_time).repeat_interleave(N)

    if USE_LOADING_RATE:
        hetero_data["loading_rate"].x = torch.hstack((torch.tensor(loading_rate_nodes), torch.zeros((loading_rate_nodes.shape[0], 2)))).float()

    edge_index_spatial = []
    for t in range(latest_time):
        src = torch.arange(num_nodes - 1) + t * num_nodes
        dst = src + 1
        edge_index_spatial.append(torch.stack([src, dst]))

    edge_index_spatial = torch.cat(edge_index_spatial, dim=1)
    hetero_data['earthquake_source', 'spatial', 'earthquake_source'].edge_index = edge_index_spatial

    edge_index_temporal = []
    for t in range(latest_time - 1):
        src = torch.arange(num_nodes) + t * num_nodes
        dst = src + num_nodes
        edge_index_temporal.append(torch.stack([src, dst]))

    edge_index_temporal = torch.cat(edge_index_temporal, dim=1)
    hetero_data['earthquake_source', 'temporal', 'earthquake_source'].edge_index = edge_index_temporal

    if USE_LOADING_RATE:
        edge_index_loading_rate = torch.stack((torch.arange(num_nodes).repeat(latest_time), torch.arange(num_nodes * latest_time)))
        hetero_data['loading_rate', 'lr', 'earthquake_source'].edge_index = edge_index_loading_rate


    from torch_geometric.loader import DataLoader
    all_samples = [get_time_window_subgraph(hetero_data, start_time, CONTEXT_LENGTH) for start_time in range(latest_time - CONTEXT_LENGTH + 1)]
    return all_samples
    # TRAIN_INDEX_END = int(len(all_samples) * TRAIN_SPLIT)
    # VAL_INDEX_END = TRAIN_INDEX_END + int(len(all_samples) * VAL_SPLIT)

    # train_loader = DataLoader(all_samples[:TRAIN_INDEX_END], batch_size=BATCH_SIZE, shuffle=True)
    # val_loader = DataLoader(all_samples[TRAIN_INDEX_END:VAL_INDEX_END], batch_size=BATCH_SIZE, shuffle=True)
    # test_loader = DataLoader(all_samples[VAL_INDEX_END:], batch_size=BATCH_SIZE, shuffle=True)