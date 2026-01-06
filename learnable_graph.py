#learnable_graph
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.neighbors import KDTree
from collections import defaultdict, Counter
from tqdm import tqdm
import pickle
import os


class CandidateSelector:
    """离线候选边筛选模块：地理+行为双重筛选"""

    def __init__(self, poi_coords, trajectories, times):
        self.poi_coords = poi_coords
        self.trajectories = trajectories
        self.times = times

        # 构建KD-Tree用于地理邻居搜索
        self.poi_ids = list(poi_coords.keys())
        self.coords_array = np.array([poi_coords[pid] for pid in self.poi_ids])
        self.geo_tree = KDTree(self.coords_array)

        # 构建转移频率矩阵
        self.transition_freq = self.build_transition_matrix()

    def build_transition_matrix(self):
        freq = defaultdict(Counter)

        print("Building transition frequency matrix...")
        for traj, time_seq in tqdm(zip(self.trajectories, self.times), desc="Processing trajectories"):
            for i in range(len(traj) - 1):
                poi_curr = traj[i]
                poi_next = traj[i + 1]
                time_curr = time_seq[i]
                time_next = time_seq[i + 1]

                if abs(time_next - time_curr) < 86400:
                    freq[poi_curr][poi_next] += 1

        print(f"Built transition matrix for {len(freq)} POIs")
        return freq

    def get_candidates(self, poi_id, k_geo=50, k_freq=20):
        if poi_id not in self.poi_coords:
            return []

        poi_idx = self.poi_ids.index(poi_id)
        distances, indices = self.geo_tree.query([self.coords_array[poi_idx]], k=min(k_geo, len(self.poi_ids)))
        geo_neighbors = [self.poi_ids[idx] for idx in indices[0] if self.poi_ids[idx] != poi_id]

        scored = []
        for neighbor_id in geo_neighbors:
            freq_score = self.transition_freq[poi_id].get(neighbor_id, 0)
            scored.append((neighbor_id, freq_score))

        sorted_neighbors = sorted(scored, key=lambda x: -x[1])
        candidates = [neighbor_id for neighbor_id, _ in sorted_neighbors[:k_freq]]

        if not candidates or all(score == 0 for _, score in scored[:k_freq]):
            candidates = geo_neighbors[:k_freq]

        return candidates

    def build_candidate_graph(self, k_geo=50, k_freq=20):
        candidate_graph = {}

        print("Building candidate graph...")
        for poi_id in tqdm(self.poi_ids, desc="Processing POIs"):
            candidates = self.get_candidates(poi_id, k_geo, k_freq)
            candidate_graph[poi_id] = candidates

        return candidate_graph

    def save_candidates(self, filepath, k_geo=50, k_freq=20):
        candidate_graph = self.build_candidate_graph(k_geo, k_freq)
        with open(filepath, 'wb') as f:
            pickle.dump(candidate_graph, f)
        print(f"Candidate graph saved to {filepath}")

    @staticmethod
    def load_candidates(filepath):
        with open(filepath, 'rb') as f:
            candidate_graph = pickle.load(f)
        print(f"Candidate graph loaded from {filepath}")
        return candidate_graph


class LearnableGraphBuilder(nn.Module):
    """在线可训练图构建模块"""

    def __init__(self, emb_dim, candidate_graph, top_k=20, use_mlp=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.candidate_graph = candidate_graph
        self.top_k = top_k
        self.use_mlp = use_mlp

        # 【新增】MLP边权重学习模块
        if use_mlp:
            self.score_mlp = nn.Sequential(
                nn.Linear(emb_dim * 2, emb_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(emb_dim, 1)
            )
            print(f"Initialized MLP-based edge weight learning with emb_dim={emb_dim}")
        else:
            print(f"Using dot-product edge weight calculation with emb_dim={emb_dim}")

    def forward(self, poi_embeddings):
        src_list, dst_list, weight_list = [], [], []

        for src_poi, neighbor_pois in self.candidate_graph.items():
            if not neighbor_pois:
                continue

            neighbor_pois = neighbor_pois[:self.top_k]
            src_emb = poi_embeddings[src_poi]
            neighbor_embs = poi_embeddings[neighbor_pois]

            # 【核心改进】MLP vs 点积边权重计算
            if self.use_mlp:
                # MLP方式：学习非线性边权重关系
                src_expanded = src_emb.unsqueeze(0).repeat(len(neighbor_pois), 1)  # (k, emb_dim)
                edge_features = torch.cat([src_expanded, neighbor_embs], dim=-1)    # (k, 2*emb_dim)
                weights = self.score_mlp(edge_features).squeeze(-1)                # (k,)
            else:
                # 原始点积方式：线性相似度计算
                weights = torch.matmul(neighbor_embs, src_emb)                     # (k,)

            # 保持softmax归一化（无论哪种方式）
            weights = F.softmax(weights, dim=0)

            src_list.extend([src_poi] * len(neighbor_pois))
            dst_list.extend(neighbor_pois)
            weight_list.append(weights)

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_weight = torch.cat(weight_list, dim=0) if weight_list else torch.empty(0)

        return edge_index, edge_weight


def prepare_candidate_data(poi_loader):
    poi_coords = poi_loader.poi2gps
    trajectories = poi_loader.locs
    times = poi_loader.times

    return poi_coords, trajectories, times


def build_and_save_candidates(poi_loader, save_path, k_geo=50, k_freq=20):
    poi_coords = poi_loader.poi2gps
    trajectories = poi_loader.locs
    times = poi_loader.times

    print(f"Using {len(poi_coords)} POI coordinates")
    print(f"Using {len(trajectories)} user trajectories")

    selector = CandidateSelector(poi_coords, trajectories, times)
    selector.save_candidates(save_path, k_geo, k_freq)

    return selector


class GraphCache:
    def __init__(self, update_freq=5):
        self.update_freq = update_freq
        self.cached_edge_index = None
        self.cached_edge_weight = None
        self.last_update_epoch = -1

    def should_update(self, epoch):
        return epoch % self.update_freq == 0 or self.cached_edge_index is None

    def update_cache(self, edge_index, edge_weight, epoch):
        self.cached_edge_index = edge_index.detach()
        self.cached_edge_weight = edge_weight.detach()
        self.last_update_epoch = epoch

    def get_cached_graph(self):
        return self.cached_edge_index, self.cached_edge_weight