import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from network import Flashback


def maksed_mse_loss(input, target, mask_value=-1):
    mask = target == mask_value
    out = (input[~mask] - target[~mask]) ** 2
    loss = out.mean()
    return loss


def trajectory_forecasting_loss(pred, true):
    return F.mse_loss(pred, true, reduction='mean')


def consistency_loss(pred_aux, pred_main):
    return F.mse_loss(pred_aux, pred_main, reduction='mean')


class FlashbackTrainer():
    """Enhanced Flashback trainer with learnable graph and interest fusion support."""

    def __init__(self, lambda_t, lambda_s, lambda_loc, lambda_user, use_weight,
                 use_learnable_graph=False, candidate_graph=None, use_mlp_edge=False, top_k_neighbors=10,
                 use_interest_fusion=False, fusion_type='linear', init_alpha=0.7):
        self.lambda_t = lambda_t
        self.lambda_s = lambda_s
        self.lambda_loc = lambda_loc
        self.lambda_user = lambda_user
        self.use_weight = use_weight

        self.use_learnable_graph = use_learnable_graph
        self.candidate_graph = candidate_graph
        self.use_mlp_edge = use_mlp_edge
        self.top_k_neighbors = top_k_neighbors

        self.use_interest_fusion = use_interest_fusion
        self.fusion_type = fusion_type
        self.init_alpha = init_alpha

        self.loss_weight = nn.Parameter(torch.ones(3))
        self.setting = None  # 添加这行，初始化setting属性

    def __str__(self):
        base_str = ""
        if self.use_learnable_graph:
            edge_method = "MLP-based" if self.use_mlp_edge else "dot-product"
            base_str = f'Use flashback training with learnable graph ({edge_method} edge weights)'
        else:
            base_str = 'Use flashback training without graph'

        if self.use_interest_fusion:
            base_str += f' + Interest Fusion ({self.fusion_type})'

        return base_str + '.'

    def count_parameters(self):
        param_count = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param_count += param.numel()
        return param_count

    def parameters(self):
        return self.model.parameters()

    def prepare(self, loc_count, user_count, hidden_size, gru_factory, device, setting):
        def f_t(delta_t, user_len):
            return ((torch.cos(delta_t * 2 * np.pi / 86400) + 1) / 2) * torch.exp(-(delta_t / 86400 * self.lambda_t))

        def f_s(delta_s, user_len):
            return torch.exp(-(delta_s * self.lambda_s))

        self.loc_count = loc_count
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.setting = setting  # 添加这行，保存setting

        self.model = Flashback(
            input_size=loc_count,
            user_count=user_count,
            hidden_size=hidden_size,
            f_t=f_t,
            f_s=f_s,
            rnn_factory=gru_factory,
            lambda_loc=self.lambda_loc,
            lambda_user=self.lambda_user,
            use_weight=self.use_weight,
            setting=setting,
            use_learnable_graph=self.use_learnable_graph,
            candidate_graph=self.candidate_graph,
            use_mlp_edge=self.use_mlp_edge,
            top_k_neighbors=self.top_k_neighbors,
            use_interest_fusion=self.use_interest_fusion,
            fusion_type=self.fusion_type,
            init_alpha=self.init_alpha
        ).to(device)

    def update_learnable_graph(self, epoch):
        if self.use_learnable_graph and hasattr(self.model, 'learnable_graph_builder'):
            with torch.no_grad():
                poi_emb = self.model.encoder.weight.detach()
                edge_index, edge_weight = self.model.learnable_graph_builder(poi_emb)
                self.model.update_graph_cache(edge_index, edge_weight)
                if hasattr(self.model, 'graph_cache'):
                    self.model.graph_cache.last_update_epoch = epoch
            return edge_index.size(1)
        return 0

    def should_update_graph(self, epoch, update_freq=5):
        if not self.use_learnable_graph:
            return False

        if hasattr(self.model, 'graph_cache'):
            return self.model.graph_cache.should_update(epoch)

        return epoch % update_freq == 0

    def get_model_info(self):
        info = {
            'use_learnable_graph': self.use_learnable_graph,
            'use_mlp_edge': self.use_mlp_edge,
            'use_interest_fusion': self.use_interest_fusion
        }

        if self.use_interest_fusion and hasattr(self.model, 'interest_fusion'):
            info.update(self.model.get_interest_fusion_info())

        return info

    def evaluate(self, x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_users, f, y_f, dataset):
        self.model.eval()
        out, h, _, _ = self.model(x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_users, f, y_f, dataset)
        out_t = out.transpose(0, 1)
        return out_t, h

    def loss(self, x, t, t_slot, s, y, y_t, y_t_slot, y_s, h, active_users, f, y_f, logits, dataset):
        self.model.train()
        out, h, cos, out_time = self.model(x, t, t_slot, s, y_t, y_t_slot, y_s, h, active_users, f, y_f, dataset)

        out = out.view(-1, self.loc_count)
        y = y.view(-1)

        # 【修改这里】改为使用self.setting判断
        if self.setting and self.setting.use_enhanced_loss:
            # 使用增强损失函数（处理长尾POI）
            cos = cos.view(-1, self.loc_count)

            target_cosine = cos.gather(1, y.unsqueeze(1)).view(-1)
            vector_lengths = torch.where(target_cosine > 0, torch.ones_like(target_cosine), 1 - target_cosine)

            log_geom_mean_length = torch.log(vector_lengths + 1e-9).mean()
            geom_mean_length = torch.exp(log_geom_mean_length)

            length_diff = vector_lengths - geom_mean_length

            weights = torch.ones_like(vector_lengths)
            weights[length_diff > 0] = 1 + length_diff[length_diff > 0]

            l1 = (self.cross_entropy_loss(out + logits, y) * weights).mean()

            loss_weights = F.softmax(self.loss_weight, dim=0)

            l2 = self.cross_entropy_loss(out, y)
            l3 = maksed_mse_loss(out_time.squeeze(-1), y_t_slot.squeeze(0) / 168)

            l = loss_weights[0] * l1 + loss_weights[1] * l2 + loss_weights[2] * l3
        else:
            # 使用简单交叉熵损失
            l = self.cross_entropy_loss(out, y)

        return l