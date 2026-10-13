"""Node-count-independent forecasting models, implemented in native PyTorch."""

import torch
from torch import nn
from torch.nn import functional as F


MODEL_NAMES = ("persistence", "seasonal_naive", "gru", "gru_attention",
               "gcn_gru", "gat_gru", "stfednet", "stfednet_residual")


def adjacency(edge_index, num_nodes, device):
    """Incoming-neighbour adjacency, including exactly one self-loop per node."""
    edges = edge_index.to(device=device, dtype=torch.long)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if edges.numel() and (edges.min() < 0 or edges.max() >= num_nodes):
        raise ValueError("Graph endpoint outside input node range")
    graph = torch.eye(num_nodes, dtype=torch.bool, device=device)
    graph[edges[1], edges[0]] = True
    return graph


class GraphAttention(nn.Module):
    """Two-head GAT with shared parameters at every time step.

    Dense masking is intentional for these small graphs (39--80 nodes).
    Head features are concatenated to the requested hidden dimension.
    No learnable parameter is indexed by physical bus ID or node count.
    """

    def __init__(self, in_features=1, hidden_dim=32, heads=2, dropout=0.1):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by GAT heads")
        self.heads, self.head_dim = heads, hidden_dim // heads
        self.projection = nn.Linear(in_features, hidden_dim, bias=False)
        self.source_attention = nn.Parameter(torch.empty(heads, self.head_dim))
        self.target_attention = nn.Parameter(torch.empty(heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.source_attention)
        nn.init.xavier_uniform_(self.target_attention)

    def forward(self, x, edge_index):
        batch, time, nodes, _ = x.shape
        z = self.projection(x).reshape(batch * time, nodes, self.heads, self.head_dim)
        z = z.permute(0, 2, 1, 3)  # [B*T, H, node, head_feature]
        source = (z * self.source_attention[None, :, None, :]).sum(-1)
        target = (z * self.target_attention[None, :, None, :]).sum(-1)
        scores = F.leaky_relu(target.unsqueeze(-1) + source.unsqueeze(-2), 0.2)
        graph = adjacency(edge_index, nodes, x.device)
        weights = scores.masked_fill(~graph, float("-inf")).softmax(dim=-1)
        output = self.dropout(weights) @ z
        output = output.permute(0, 2, 1, 3).reshape(batch, time, nodes, -1)
        return F.elu(output + self.bias)


class GraphConvolution(nn.Module):
    """GCN using D^-1/2 (A+I) D^-1/2 with the physical adjacency."""

    def __init__(self, in_features=1, hidden_dim=32):
        super().__init__()
        self.projection = nn.Linear(in_features, hidden_dim)

    def forward(self, x, edge_index):
        graph = adjacency(edge_index, x.shape[2], x.device).to(x.dtype)
        degree = graph.sum(-1).clamp_min(1).rsqrt()
        normalised = degree[:, None] * graph * degree[None, :]
        return F.relu(torch.einsum("ij,btjf->btif", normalised, self.projection(x)))


class TemporalEncoder(nn.Module):
    """One GRU shared across all nodes; returns every historical state."""

    def __init__(self, in_features, hidden_dim=32):
        super().__init__()
        self.gru = nn.GRU(in_features, hidden_dim, num_layers=1, batch_first=True)

    def forward(self, x):
        batch, time, nodes, features = x.shape
        sequences = x.permute(0, 2, 1, 3).reshape(batch * nodes, time, features)
        states, _ = self.gru(sequences)
        return states.reshape(batch, nodes, time, -1)


class AdditiveTemporalAttention(nn.Module):
    """Single-head v^T tanh(W h_t + b), normalised over historical time."""

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.hidden = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, states, return_weights=False):
        weights = self.score(torch.tanh(self.hidden(states))).squeeze(-1).softmax(dim=-1)
        context = (states * weights.unsqueeze(-1)).sum(dim=-2)
        return (context, weights) if return_weights else context


class NaiveForecast(nn.Module):
    def __init__(self, seasonal=False, horizon=1):
        super().__init__()
        if horizon != 1:
            raise ValueError("G1 naive baselines require horizon=1")
        self.seasonal = seasonal

    def forward(self, x, edge_index=None):
        if self.seasonal and x.shape[1] < 24:
            raise ValueError("Seasonal naive requires at least 24 history hours")
        position = -24 if self.seasonal else -1
        return x[:, position, :, 0].unsqueeze(1)


class NeuralForecast(nn.Module):
    def __init__(self, spatial=None, temporal_attention=False,
                 hidden_dim=32, dropout=0.1, horizon=1):
        super().__init__()
        self.spatial = spatial
        self.temporal = TemporalEncoder(1 if spatial is None else hidden_dim, hidden_dim)
        self.attention = AdditiveTemporalAttention(hidden_dim) if temporal_attention else None
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, horizon)

    def forward(self, x, edge_index):
        if self.spatial is not None:
            x = self.spatial(x, edge_index)
        states = self.temporal(x)
        context = self.attention(states) if self.attention is not None else states[:, :, -1, :]
        return self.head(self.dropout(context)).transpose(1, 2).contiguous()


class SpatialResidualFusion(nn.Module):
    """Map GAT features to one channel and add a small learned spatial residual."""

    def __init__(self, hidden_dim=32, dropout=0.1, initial_scale=0.05):
        super().__init__()
        self.gat = GraphAttention(hidden_dim=hidden_dim, heads=2, dropout=dropout)
        self.to_load_channel = nn.Linear(hidden_dim, 1)
        self.space_scale = nn.Parameter(torch.tensor(float(initial_scale)))

    def forward(self, x, edge_index, return_spatial=False):
        spatial_features = self.to_load_channel(self.gat(x, edge_index))
        fused = x + self.space_scale * spatial_features
        if return_spatial:
            return fused, spatial_features
        return fused


class ResidualSTFedNet(nn.Module):
    """GAT -> learned scalar spatial residual -> GRU -> temporal attention."""

    def __init__(self, hidden_dim=32, dropout=0.1, horizon=1, initial_scale=0.05):
        super().__init__()
        self.spatial = SpatialResidualFusion(hidden_dim, dropout, initial_scale)
        self.temporal = TemporalEncoder(1, hidden_dim)
        self.attention = AdditiveTemporalAttention(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_dim, horizon)

    @property
    def space_scale(self):
        """Learnable scalar coefficient, exposed for logs and experiment analysis."""
        return self.spatial.space_scale

    def forward(self, x, edge_index):
        x = self.spatial(x, edge_index)
        states = self.temporal(x)
        context = self.attention(states)
        return self.head(self.dropout(context)).transpose(1, 2).contiguous()


def build_model(name, hidden_dim=32, dropout=0.1, horizon=1):
    """Common factory used by local training and future federated clients."""
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; choose from {MODEL_NAMES}")
    if name in {"persistence", "seasonal_naive"}:
        return NaiveForecast(name == "seasonal_naive", horizon)
    if name == "stfednet_residual":
        return ResidualSTFedNet(hidden_dim, dropout, horizon, initial_scale=0.05)
    spatial = None
    if name in {"gat_gru", "stfednet"}:
        spatial = GraphAttention(hidden_dim=hidden_dim, heads=2, dropout=dropout)
    elif name == "gcn_gru":
        spatial = GraphConvolution(hidden_dim=hidden_dim)
    return NeuralForecast(spatial, name in {"gru_attention", "stfednet"},
                          hidden_dim, dropout, horizon)
