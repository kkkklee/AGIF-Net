# interest_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class InterestFusionModule(nn.Module):
    """
    兴趣漂移建模模块：融合短期兴趣和长期偏好

    功能：
    - 管理用户长期偏好embedding
    - 融合短期RNN输出和长期偏好
    - 支持线性融合和Gate融合两种方式
    """

    def __init__(self, hidden_dim, num_users, fusion_type='linear', init_alpha=0.7):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_users = num_users
        self.fusion_type = fusion_type

        # 长期偏好embedding：每个用户一个可学习的长期兴趣向量
        self.long_term_emb = nn.Embedding(num_users, hidden_dim)

        # 初始化长期偏好embedding
        nn.init.normal_(self.long_term_emb.weight, mean=0.0, std=0.1)

        if fusion_type == 'linear':
            # 线性融合：可学习的权重参数
            self.alpha = nn.Parameter(torch.tensor(init_alpha))
            print(f"Interest Fusion: Linear fusion with learnable alpha (init={init_alpha})")

        elif fusion_type == 'gate':
            # Gate融合：MLP决定融合权重
            self.gate_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            )
            print(f"Interest Fusion: Gate fusion with MLP")

        else:
            raise ValueError(f"Unsupported fusion_type: {fusion_type}")

    def forward(self, short_term_interest, user_ids):
        """
        前向传播：融合短期兴趣和长期偏好

        Args:
            short_term_interest: 短期兴趣表示 (batch_size, hidden_dim)
            user_ids: 用户ID (batch_size,)

        Returns:
            final_interest: 融合后的兴趣表示 (batch_size, hidden_dim)
        """
        # 获取长期偏好
        long_term_pref = self.long_term_emb(user_ids)  # (batch_size, hidden_dim)

        if self.fusion_type == 'linear':
            # 线性融合：α * short + (1-α) * long
            alpha = torch.sigmoid(self.alpha)  # 确保α在(0,1)范围内
            final_interest = alpha * short_term_interest + (1 - alpha) * long_term_pref

        elif self.fusion_type == 'gate':
            # Gate融合：MLP计算动态权重
            gate_input = torch.cat([short_term_interest, long_term_pref], dim=-1)
            gate = self.gate_mlp(gate_input)  # (batch_size, 1)
            final_interest = gate * short_term_interest + (1 - gate) * long_term_pref

        return final_interest

    def get_fusion_weight(self):
        """
        获取当前融合权重（用于日志记录和分析）
        """
        if self.fusion_type == 'linear':
            return torch.sigmoid(self.alpha).item()
        else:
            return "Dynamic (Gate-based)"

    def get_long_term_preference(self, user_id):
        """
        获取特定用户的长期偏好向量（用于分析）
        """
        return self.long_term_emb(torch.tensor([user_id])).squeeze(0)


class InterestAnalyzer:
    """
    兴趣分析工具：用于可视化和分析兴趣融合效果
    """

    @staticmethod
    def compute_interest_similarity(fusion_module, user_ids, short_term_interests):
        """
        计算短期兴趣和长期偏好的相似度
        """
        with torch.no_grad():
            long_term_prefs = fusion_module.long_term_emb(user_ids)
            similarities = F.cosine_similarity(short_term_interests, long_term_prefs, dim=-1)
            return similarities.cpu().numpy()

    @staticmethod
    def analyze_fusion_dynamics(fusion_module, user_ids, short_term_interests):
        """
        分析融合动态：短期vs长期的权重分布
        """
        results = {}

        if fusion_module.fusion_type == 'linear':
            alpha = torch.sigmoid(fusion_module.alpha).item()
            results['short_weight'] = alpha
            results['long_weight'] = 1 - alpha
            results['fusion_type'] = 'linear'

        elif fusion_module.fusion_type == 'gate':
            with torch.no_grad():
                long_term_prefs = fusion_module.long_term_emb(user_ids)
                gate_input = torch.cat([short_term_interests, long_term_prefs], dim=-1)
                gates = fusion_module.gate_mlp(gate_input)
                results['short_weights'] = gates.squeeze(-1).cpu().numpy()
                results['long_weights'] = (1 - gates.squeeze(-1)).cpu().numpy()
                results['fusion_type'] = 'gate'

        return results


# 辅助函数：计算用户历史POI访问均值（可选的长期偏好初始化方式）
def compute_historical_preference(poi_loader, user_id, poi_embeddings):
    """
    基于历史访问计算用户长期偏好（备选方案）

    Args:
        poi_loader: POI数据加载器
        user_id: 用户ID
        poi_embeddings: POI embedding矩阵

    Returns:
        historical_pref: 历史访问的POI embedding均值
    """
    # 获取用户历史访问的POI列表
    if hasattr(poi_loader, 'locs') and user_id < len(poi_loader.locs):
        user_pois = poi_loader.locs[user_id]
        if len(user_pois) > 0:
            # 计算历史POI embedding的均值
            poi_embs = poi_embeddings[user_pois]
            historical_pref = torch.mean(poi_embs, dim=0)
            return historical_pref

    # 如果没有历史数据，返回零向量
    return torch.zeros(poi_embeddings.size(1))


def init_long_term_from_history(fusion_module, poi_loader, poi_embeddings):
    """
    用历史数据初始化长期偏好embedding（可选）
    """
    print("Initializing long-term preferences from historical data...")

    with torch.no_grad():
        for user_id in range(fusion_module.num_users):
            hist_pref = compute_historical_preference(poi_loader, user_id, poi_embeddings)
            if hist_pref.sum() != 0:  # 如果有历史数据
                fusion_module.long_term_emb.weight[user_id] = hist_pref

    print("Long-term preference initialization completed.")