import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix
from torch.nn import init
from enum import Enum
import time
import numpy as np
from utils import *
import scipy.sparse as sp
import math
from model import TransformerModel, EncoderLayer, Time2Vec, FuseEmbeddings, FuseEmbeddings1, Decoder

from learnable_graph import LearnableGraphBuilder, CandidateSelector, GraphCache
from interest_fusion import InterestFusionModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import math


def gaussian_distance_weight(haversine_distance, sigma=1):
    return torch.exp(-haversine_distance ** 2 / (2 * sigma ** 2))


def migration_weight(migration_count, delta=1):
    return torch.log(1 + migration_count) / (1 + delta)


def combined_weight(haversine_distance, migration_count, alpha=1, delta=1, method='add'):
    w_dist = gaussian_distance_weight(haversine_distance, alpha)
    w_migr = migration_weight(migration_count, delta)

    if method == 'add':
        return w_dist + w_migr
    elif method == 'mult':
        return w_dist * w_migr
    else:
        raise ValueError("Method must be 'add' or 'multiply'")


def haversine(s1, s2):
    """Calculate Haversine distance between two batches of geographic locations."""
    s1 = s1 * math.pi / 180
    s2 = s2 * math.pi / 180

    lat1, lon1 = s1[:, 0], s1[:, 1]
    lat2, lon2 = s2[:, 0], s2[:, 1]

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.asin(torch.sqrt(a))

    r = 6371
    return c * r


def haversines(s1, s2):
    """Calculate Haversine distance between two batches of geographic locations."""
    s1 = s1 * math.pi / 180
    s2 = s2 * math.pi / 180

    s1_expanded = s1.unsqueeze(2)
    s2_expanded = s2.unsqueeze(1)

    dlat = s2_expanded[..., 0] - s1_expanded[..., 0]
    dlon = s2_expanded[..., 1] - s1_expanded[..., 1]

    a = torch.sin(dlat / 2) ** 2 + torch.cos(s1_expanded[..., 0]) * torch.cos(s2_expanded[..., 0]) * torch.sin(
        dlon / 2) ** 2
    c = 2 * torch.asin(torch.sqrt(a))

    r = 6371
    return c * r


class Rnn(Enum):
    """The available RNN units"""

    RNN = 0
    GRU = 1
    LSTM = 2

    @staticmethod
    def from_string(name):
        if name == 'rnn':
            return Rnn.RNN
        if name == 'gru':
            return Rnn.GRU
        if name == 'lstm':
            return Rnn.LSTM
        raise ValueError('{} not supported in --rnn'.format(name))


class RnnFactory():
    """Creates the desired RNN unit."""

    def __init__(self, rnn_type_str):
        self.rnn_type = Rnn.from_string(rnn_type_str)

    def __str__(self):
        if self.rnn_type == Rnn.RNN:
            return 'Use pytorch RNN implementation.'
        if self.rnn_type == Rnn.GRU:
            return 'Use pytorch GRU implementation.'
        if self.rnn_type == Rnn.LSTM:
            return 'Use pytorch LSTM implementation.'

    def is_lstm(self):
        return self.rnn_type in [Rnn.LSTM]

    def create(self, hidden_size):
        if self.rnn_type == Rnn.RNN:
            return nn.RNN(hidden_size, hidden_size)
        if self.rnn_type == Rnn.GRU:
            return nn.GRU(hidden_size, hidden_size)
        if self.rnn_type == Rnn.LSTM:
            return nn.LSTM(hidden_size, hidden_size)


class AttentionLayer(nn.Module):
    def __init__(self, user_dim, item_dim):
        super(AttentionLayer, self).__init__()
        self.attention_fc = nn.Sequential(
            nn.Linear(user_dim + item_dim, 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32, 1)
        )
        for layer in self.attention_fc:
            if isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                init.zeros_(layer.bias)

    def forward(self, user_embeddings, item_embeddings, edge_index):
        user_indices = edge_index[0]
        item_indices = edge_index[1]
        user_feats = user_embeddings[user_indices]
        item_feats = item_embeddings[item_indices]

        edge_feats = torch.cat([user_feats, item_feats], dim=1)
        edge_weights = torch.sigmoid(self.attention_fc(edge_feats)).squeeze()

        return edge_weights


class DenoisingLayer(nn.Module):
    def __init__(self):
        super(DenoisingLayer, self).__init__()

    def forward(self, edge_weights, edge_index, threshold=0.8):
        mask = edge_weights > threshold
        if mask.sum() == 0:
            mask[edge_weights.argmax()] = True
        denoised_edge_index = edge_index[:, mask]
        denoised_edge_weights = edge_weights[mask]

        return denoised_edge_index, denoised_edge_weights


class GCNLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNLayer, self).__init__()
        self.conv1 = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x


class DenoisingGCNNet(nn.Module):
    def __init__(self, user_dim, item_dim, out_channels):
        super(DenoisingGCNNet, self).__init__()
        self.attention_layer = AttentionLayer(user_dim, item_dim)
        self.denoising_layer = DenoisingLayer()
        self.gcn_layer = GCNLayer(user_dim, out_channels)

    def forward(self, user_embeddings, item_embeddings, edge_index):
        edge_weights = self.attention_layer(user_embeddings, item_embeddings, edge_index)
        denoised_edge_index, denoised_edge_weights = self.denoising_layer(edge_weights, edge_index)

        gcn_input = torch.cat([user_embeddings, item_embeddings], dim=0)
        gcn_output = self.gcn_layer(gcn_input, denoised_edge_index)

        return gcn_output, denoised_edge_index, denoised_edge_weights


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class Flashback(nn.Module):
    """Enhanced Flashback RNN with learnable graph and interest fusion capabilities"""

    def __init__(self, input_size, user_count, hidden_size, f_t, f_s, rnn_factory, lambda_loc, lambda_user, use_weight,
                 setting,
                 use_learnable_graph=False, candidate_graph=None, use_mlp_edge=False, top_k_neighbors=10,
                 use_interest_fusion=False, fusion_type='linear', init_alpha=0.7):
        super().__init__()
        self.input_size = input_size
        self.user_count = user_count
        self.hidden_size = hidden_size
        self.f_t = f_t
        self.f_s = f_s

        self.lambda_loc = lambda_loc
        self.lambda_user = lambda_user
        self.use_weight = use_weight

        self.use_learnable_graph = use_learnable_graph
        self.use_interest_fusion = use_interest_fusion

        self.setting = setting

        # learnable graph setup
        if use_learnable_graph:
            self.learnable_graph_builder = LearnableGraphBuilder(
                emb_dim=hidden_size,
                candidate_graph=candidate_graph,
                use_mlp=use_mlp_edge,
                top_k=top_k_neighbors
            )
            self.graph_cache = GraphCache()
            print(f"Learnable graph enabled with {'MLP' if use_mlp_edge else 'dot-product'} edge weights")
        else:
            print("Learnable graph disabled")

        self.encoder = nn.Embedding(input_size, hidden_size)
        self.user_encoder = nn.Embedding(user_count, hidden_size)
        self.rnn = rnn_factory.create(hidden_size)

        # LoTNext components
        self.seq_model = EncoderLayer(
            setting.hidden_dim + 6,
            setting.transformer_nhid,
            setting.transformer_dropout,
            setting.attention_dropout_rate,
            setting.transformer_nhead)
        self.time_embed_model = Time2Vec('sin', setting.batch_size, setting.sequence_length, out_dim=6)
        self.embed_fuse_model = FuseEmbeddings(hidden_size, 6)

        self.decoder = nn.Linear(setting.hidden_dim + 6, setting.hidden_dim)
        self.fc = nn.Linear(2 * hidden_size, input_size)
        self.time_decoder = nn.Linear(setting.hidden_dim + 6, 1)

        # interest fusion module
        if self.use_interest_fusion:
            self.interest_fusion = InterestFusionModule(
                hidden_dim=hidden_size,
                num_users=user_count,
                fusion_type=fusion_type,
                init_alpha=init_alpha
            )
            print(f"Interest Fusion Module initialized: {fusion_type} fusion with alpha={init_alpha}")
        else:
            self.interest_fusion = None
            print("Interest Fusion disabled")

    def update_graph_cache(self, edge_index, edge_weight):
        """Update graph cache"""
        if hasattr(self, 'graph_cache'):
            self.graph_cache.update_cache(edge_index, edge_weight, 0)

    def get_interest_fusion_info(self):
        """Get interest fusion module information"""
        if self.interest_fusion is not None:
            return {
                'enabled': True,
                'fusion_type': self.interest_fusion.fusion_type,
                'fusion_weight': self.interest_fusion.get_fusion_weight(),
                'num_users': self.interest_fusion.num_users
            }
        else:
            return {'enabled': False}

    def apply_learnable_gcn(self, loc_emb, edge_index, edge_weight):
        """Apply learnable graph for GCN operation"""
        if edge_index.size(1) > 0:
            sparse_adj = torch.sparse_coo_tensor(
                edge_index,
                edge_weight,
                (self.input_size, self.input_size),
                device=loc_emb.device
            ).coalesce()
            encoder_weight = torch.sparse.mm(sparse_adj, loc_emb)
        else:
            encoder_weight = loc_emb
        return encoder_weight

    def forward(self, x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_user, f, y_f, dataset):
        seq_len, user_len = x.size()
        x_emb = self.encoder(x)

        p_u = self.user_encoder(active_user)
        p_u = p_u.view(user_len, self.hidden_size)

        # POI embedding processing with learnable graph
        loc_emb = self.encoder(torch.LongTensor(list(range(self.input_size))).to(x.device))

        if self.use_learnable_graph:
            if hasattr(self, 'graph_cache') and self.graph_cache.cached_edge_index is not None:
                edge_index = self.graph_cache.cached_edge_index.to(x.device)
                edge_weight = self.graph_cache.cached_edge_weight.to(x.device)
            else:
                edge_index, edge_weight = self.learnable_graph_builder(loc_emb)
                edge_index = edge_index.to(x.device)
                edge_weight = edge_weight.to(x.device)

            encoder_weight = self.apply_learnable_gcn(loc_emb, edge_index, edge_weight)
        else:
            encoder_weight = loc_emb

        # build new POI embedding sequence
        new_x_emb = []
        for i in range(seq_len):
            temp_x = torch.index_select(encoder_weight, 0, x[i])
            new_x_emb.append(temp_x)
        x_emb = torch.stack(new_x_emb, dim=0)

        # default user location similarity
        user_loc_similarity = torch.ones(user_len, seq_len, device=x.device)

        # LoTNext Transformer processing
        t_emb = self.time_embed_model(t_slot.transpose(0, 1) / 168).to(x.device)
        x_emb = self.embed_fuse_model(x_emb.transpose(0, 1), t_emb).to(x.device)
        out = self.seq_model(x_emb).to(x.device)
        out_time = self.time_decoder(out).to(x.device).transpose(0, 1)
        out = self.decoder(out).to(x.device).transpose(0, 1)

        # spatial-temporal weight calculation
        out_w = torch.zeros(seq_len, user_len, self.hidden_size, device=x.device)

        for i in range(seq_len):
            sum_w = torch.zeros(user_len, 1, device=x.device)
            for j in range(i + 1):
                dist_s = haversine(s[i], s[j])
                b_j = self.f_s(dist_s, user_len)
                b_j = b_j.unsqueeze(1)
                w_j = b_j + 1e-10
                w_j = w_j * user_loc_similarity[:, j].unsqueeze(1)
                sum_w += w_j
                out_w[i] += w_j * out[j]
            out_w[i] /= sum_w

        # interest fusion logic
        out_pu = torch.zeros(seq_len, user_len, 2 * self.hidden_size, device=x.device)

        for i in range(seq_len):
            short_term_interest = out_w[i]

            if self.use_interest_fusion and self.interest_fusion is not None:
                final_user_interest = self.interest_fusion(
                    short_term_interest=short_term_interest,
                    user_ids=active_user.squeeze()
                )
            else:
                final_user_interest = p_u

            out_pu[i] = torch.cat([short_term_interest, final_user_interest], dim=1)

        # compute cosine similarity
        cosine_similarity = F.linear(F.normalize(out_pu), F.normalize(self.fc.weight))
        y_linear = self.fc(out_pu)

        return y_linear, h, cosine_similarity, out_time


def create_h0_strategy(hidden_size, is_lstm):
    if is_lstm:
        return LstmStrategy(hidden_size, FixNoiseStrategy(hidden_size), FixNoiseStrategy(hidden_size))
    else:
        return FixNoiseStrategy(hidden_size)


class H0Strategy():

    def __init__(self, hidden_size):
        self.hidden_size = hidden_size

    def on_init(self, user_len, device):
        pass

    def on_reset(self, user):
        pass

    def on_reset_test(self, user, device):
        return self.on_reset(user)


class FixNoiseStrategy(H0Strategy):
    """use fixed normal noise as initialization"""

    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        mu = 0
        sd = 1 / self.hidden_size
        self.h0 = torch.randn(self.hidden_size, requires_grad=False) * sd + mu

    def on_init(self, user_len, device):
        hs = []
        for i in range(user_len):
            hs.append(self.h0)
        return torch.stack(hs, dim=0).view(1, user_len, self.hidden_size).to(device)

    def on_reset(self, user):
        return self.h0


class LstmStrategy(H0Strategy):
    """creates h0 and c0 using the inner strategy"""

    def __init__(self, hidden_size, h_strategy, c_strategy):
        super(LstmStrategy, self).__init__(hidden_size)
        self.h_strategy = h_strategy
        self.c_strategy = c_strategy

    def on_init(self, user_len, device):
        h = self.h_strategy.on_init(user_len, device)
        c = self.c_strategy.on_init(user_len, device)
        return h, c

    def on_reset(self, user):
        h = self.h_strategy.on_reset(user)
        c = self.c_strategy.on_reset(user)
        return h, c