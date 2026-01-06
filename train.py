import torch
from torch.utils.data import DataLoader
import numpy as np
import time, os
import pickle
import copy
from setting import Setting
from trainer import FlashbackTrainer
from dataloader import PoiDataloader
from dataset import Split
from utils import *
from network import create_h0_strategy
from evaluation import Evaluation
from tqdm import tqdm
from collections import defaultdict, Counter

from learnable_graph import build_and_save_candidates, CandidateSelector, GraphCache
from interest_fusion import InterestFusionModule, InterestAnalyzer, init_long_term_from_history


def compute_adjustment(label_freq, setting):
    """compute the base probabilities"""
    label_freq_array = np.array(list(label_freq.values()))
    max_freq = label_freq_array.max()
    tau = 1.2
    adjustments = tau * (1 - (np.log(label_freq_array + 1e-4) / np.log(max_freq + 1e-4)))
    adjustments = torch.from_numpy(adjustments)
    adjustments = adjustments.to(setting.device)
    return adjustments


def check_for_nan(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                print(f"NaN detected in gradients of parameter: {name}")


seed = 3407
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# parse settings
setting = Setting()
setting.parse()
dir_name = os.path.dirname(setting.log_file)
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
timestring = time.strftime('%Y%m%d%H%M%S', time.localtime())
setting.log_file = setting.log_file + '_' + timestring
setting.model_files = setting.model_file + '_' + timestring + '.pth'
log = open(setting.log_file, 'w')

message = ''.join([f'{k}: {v}\n' for k, v in vars(setting).items()])
log_string(log, message)

# log configuration
if setting.use_learnable_graph:
    edge_method = "MLP-based" if setting.use_mlp_edge else "dot-product"
    log_string(log, f'Learnable Graph Configuration:')
    log_string(log, f'   - Learnable graph enabled: {setting.use_learnable_graph}')
    log_string(log, f'   - Edge weight method: {edge_method}')
    log_string(log, f'   - Graph update frequency: {setting.graph_update_freq} epochs')
    log_string(log, f'   - Top-k neighbors: {setting.top_k_neighbors}')
    log_string(log, f'   - Candidate parameters: k_geo={setting.k_geo}, k_freq={setting.k_freq}')
else:
    log_string(log, 'Learnable Graph disabled')

if setting.use_interest_fusion:
    log_string(log, f'Interest Fusion Configuration:')
    log_string(log, f'   - Interest fusion enabled: {setting.use_interest_fusion}')
    log_string(log, f'   - Fusion type: {setting.fusion_type}')
    log_string(log, f'   - Initial alpha: {setting.init_alpha}')
else:
    log_string(log, 'Interest Fusion disabled')

# load dataset
if os.path.exists(setting.loader_file):
    # 如果预处理的loader存在，直接加载
    log_string(log, f'Loading preprocessed data from {setting.loader_file}')
    with open(setting.loader_file, 'rb') as f:
        poi_loader = pickle.load(f)
else:
    # 否则从原始数据创建新的loader
    log_string(log, f'Creating new dataloader from raw data: {setting.dataset_file}')
    poi_loader = PoiDataloader(setting.max_users, setting.min_checkins)
    poi_loader.read(setting.dataset_file)

    # 可选：保存处理好的loader以便下次快速加载
    log_string(log, f'Saving processed data to {setting.loader_file}')
    with open(setting.loader_file, 'wb') as f:
        pickle.dump(poi_loader, f)

log_string(log, 'Active POI number:{}'.format(poi_loader.locations()))
log_string(log, 'Active User number:{}'.format(poi_loader.user_count()))
log_string(log, 'Total Checkins number:{}'.format(poi_loader.checkins_count()))

dataset = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TRAIN)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
dataset_test = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TEST)
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)
assert setting.batch_size < poi_loader.user_count(), 'batch size must be lower than the amount of available users'

# handle graph structure
candidate_graph = None

if setting.use_learnable_graph:
    log_string(log, 'Setting up learnable graph...')

    # build or load candidate graph
    if not os.path.exists(setting.candidate_file):
        log_string(log, f'Candidate graph not found at {setting.candidate_file}, building...')
        selector = build_and_save_candidates(
            poi_loader, setting.candidate_file,
            k_geo=setting.k_geo, k_freq=setting.k_freq
        )
        log_string(log, f'Candidate graph built and saved to {setting.candidate_file}')
    else:
        log_string(log, f'Loading existing candidate graph from {setting.candidate_file}')

    candidate_graph = CandidateSelector.load_candidates(setting.candidate_file)

log_string(log, 'Successfully loaded graph structures')

# create enhanced flashback trainer
trainer = FlashbackTrainer(
    setting.lambda_t, setting.lambda_s, setting.lambda_loc, setting.lambda_user,
    setting.use_weight,
    setting.use_learnable_graph, candidate_graph,
    setting.use_mlp_edge, setting.top_k_neighbors,
    setting.use_interest_fusion, setting.fusion_type, setting.init_alpha
)

h0_strategy = create_h0_strategy(setting.hidden_dim, setting.is_lstm)
trainer.prepare(poi_loader.locations(), poi_loader.user_count(), setting.hidden_dim, setting.rnn_factory,
                setting.device, setting)

evaluation_test = Evaluation(dataset_test, dataloader_test,
                             poi_loader.user_count(), h0_strategy, trainer, setting, log)

# model configuration info
model_info = trainer.get_model_info()
graph_type = "Learnable Graph" if model_info['use_learnable_graph'] else "No Graph"
edge_method = ""
if model_info['use_learnable_graph']:
    edge_method = f" ({('MLP-based' if model_info['use_mlp_edge'] else 'dot-product')} edges)"

fusion_info = ""
if model_info['use_interest_fusion']:
    fusion_info = f" + Interest Fusion ({setting.fusion_type})"

print(f'{trainer} {setting.rnn_factory} with {graph_type}{edge_method}{fusion_info}')
log_string(log, f'Model configuration: {graph_type}{edge_method}{fusion_info}')

# interest fusion details
if setting.use_interest_fusion and hasattr(trainer.model, 'interest_fusion') and trainer.model.interest_fusion is not None:
    fusion_info = trainer.model.get_interest_fusion_info()
    log_string(log, f'   - Current fusion weight: {fusion_info["fusion_weight"]}')
    log_string(log, f'   - Users with long-term preferences: {fusion_info["num_users"]}')

logits = compute_adjustment(dataset.freq, setting)

# initialize graph cache
graph_cache = None
if setting.use_learnable_graph:
    graph_cache = GraphCache(update_freq=setting.graph_update_freq)
    log_string(log, f'Graph cache initialized with update frequency: {setting.graph_update_freq} epochs')

# training loop
optimizer = torch.optim.AdamW(trainer.parameters(), lr=setting.learning_rate, weight_decay=setting.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.2)

param_count = trainer.count_parameters()
log_string(log, f'In total: {param_count} trainable parameters')

# log parameter counts
if setting.use_interest_fusion and hasattr(trainer.model, 'interest_fusion') and trainer.model.interest_fusion is not None:
    fusion_params = sum(p.numel() for p in trainer.model.interest_fusion.parameters())
    log_string(log, f'Interest Fusion module parameters: {fusion_params}')

    if setting.fusion_type == 'linear':
        log_string(log, f'   - Long-term embeddings: {trainer.model.interest_fusion.num_users * setting.hidden_dim}')
        log_string(log, f'   - Fusion weight (alpha): 1')
    elif setting.fusion_type == 'gate':
        gate_params = sum(p.numel() for p in trainer.model.interest_fusion.gate_mlp.parameters())
        log_string(log, f'   - Gate MLP parameters: {gate_params}')
        log_string(log, f'   - Long-term embeddings: {trainer.model.interest_fusion.num_users * setting.hidden_dim}')

if setting.use_learnable_graph and setting.use_mlp_edge and hasattr(trainer.model, 'learnable_graph_builder'):
    mlp_params = sum(p.numel() for p in trainer.model.learnable_graph_builder.score_mlp.parameters())
    log_string(log, f'MLP edge weight module parameters: {mlp_params}')

bar = tqdm(total=setting.epochs)
bar.set_description('Training')

best_acc = 0

for e in range(setting.epochs):
    h = h0_strategy.on_init(setting.batch_size, setting.device)
    dataset.shuffle_users()

    # graph cache management
    if setting.use_learnable_graph and trainer.should_update_graph(e, setting.graph_update_freq):
        edge_method = "MLP-based" if setting.use_mlp_edge else "dot-product"
        log_string(log, f'Updating learnable graph ({edge_method}) at epoch {e}...')

        num_edges = trainer.update_learnable_graph(e)
        log_string(log, f'Graph updated with {num_edges} edges at epoch {e} using {edge_method} edge weights')

    losses = []
    epoch_start = time.time()

    for i, (x, t, t_slot, s, y, y_t, y_t_slot, y_s, reset_h, active_users, f, y_f) in enumerate(dataloader):
        # reset hidden states for newly added users
        for j, reset in enumerate(reset_h):
            if reset:
                if setting.is_lstm:
                    hc = h0_strategy.on_reset(active_users[0][j])
                    h[0][0, j] = hc[0]
                    h[1][0, j] = hc[1]
                else:
                    h[0, j] = h0_strategy.on_reset(active_users[0][j])

        x = x.squeeze().to(setting.device)
        t = t.squeeze().to(setting.device)
        t_slot = t_slot.squeeze().to(setting.device)
        s = s.squeeze().to(setting.device)

        y = y.squeeze().to(setting.device)
        y_t = y_t.squeeze().to(setting.device)
        y_t_slot = y_t_slot.squeeze().to(setting.device)
        y_s = y_s.squeeze().to(setting.device)
        active_users = active_users.to(setting.device)

        f = f.squeeze().to(setting.device)
        y_f = y_f.squeeze().to(setting.device)

        optimizer.zero_grad()
        loss = trainer.loss(x, t, t_slot, s, y, y_t, y_t_slot, y_s, h, active_users, f, y_f, logits, dataset)

        loss.backward(retain_graph=True)
        check_for_nan(trainer.model)

        losses.append(loss.item())
        optimizer.step()

    # schedule learning rate
    scheduler.step()
    bar.update(1)
    epoch_end = time.time()
    log_string(log, 'One training epoch needs {:.2f}s'.format(epoch_end - epoch_start))

    # statistics
    if (e + 1) % 1 == 0:
        epoch_loss = np.mean(losses)
        log_string(log, f'Epoch: {e + 1}/{setting.epochs}')
        log_string(log, f'Used learning rate: {scheduler.get_last_lr()[0]}')
        log_string(log, f'Avg Loss: {epoch_loss}')

        # interest fusion weight logging
        if setting.use_interest_fusion and trainer.model.interest_fusion is not None:
            current_weight = trainer.model.interest_fusion.get_fusion_weight()
            if isinstance(current_weight, float):
                log_string(log, f'Current fusion weight (α): {current_weight:.4f}')

    # interest analysis
    if setting.use_interest_fusion and (e + 1) % 10 == 0:
        log_string(log, f'Interest Analysis (Epoch {e + 1}):')
        fusion_info = trainer.model.get_interest_fusion_info()
        if fusion_info['enabled']:
            log_string(log, f'   - Fusion type: {fusion_info["fusion_type"]}')
            log_string(log, f'   - Current fusion weight: {fusion_info["fusion_weight"]}')

    if (e + 1) % setting.validate_epoch == 0:
        log_string(log, f'Test Set Evaluation (Epoch: {e + 1})')
        evl_start = time.time()
        acc1 = evaluation_test.evaluate(logits, dataset)
        if acc1 > best_acc:
            state = {
                'epoch': e,
                'state_dict': trainer.model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict() if scheduler else None,
                'use_learnable_graph': setting.use_learnable_graph,
                'use_mlp_edge': setting.use_mlp_edge,
                'use_interest_fusion': setting.use_interest_fusion,
                'model_config': trainer.get_model_info()
            }
            torch.save(state, setting.model_files)
            best_acc = copy.deepcopy(acc1)
        evl_end = time.time()
        log_string(log, 'One evaluation needs {:.2f}s'.format(evl_end - evl_start))

bar.close()

# training completion summary
log_string(log, 'Training completed successfully!')

if setting.use_learnable_graph:
    edge_method = "MLP-based" if setting.use_mlp_edge else "dot-product"
    log_string(log, f'Learnable Graph Training Summary:')
    log_string(log, f'   - Edge calculation method: {edge_method}')
    log_string(log, f'   - Graph update frequency: {setting.graph_update_freq} epochs')
    log_string(log, f'   - Total graph updates: {(setting.epochs // setting.graph_update_freq) + 1}')

    if setting.use_mlp_edge:
        log_string(log, f'   - MLP edge learning: 2-layer with ReLU and Dropout(0.1)')
        log_string(log, f'   - Input dimension: {setting.hidden_dim * 2}')
        log_string(log, f'   - Output dimension: 1 (edge weight)')

if setting.use_interest_fusion:
    final_fusion_info = trainer.model.get_interest_fusion_info()
    log_string(log, f'Interest Fusion Training Summary:')
    log_string(log, f'   - Fusion method: {setting.fusion_type}')
    log_string(log, f'   - Initial alpha: {setting.init_alpha}')
    log_string(log, f'   - Final fusion weight: {final_fusion_info["fusion_weight"]}')
    log_string(log, f'   - Long-term preference dimensions: {setting.hidden_dim}')
    log_string(log, f'   - Users modeled: {poi_loader.user_count()}')

# save final graph info
if setting.use_learnable_graph:
    final_poi_emb = trainer.model.encoder.weight.detach()
    final_edge_index, final_edge_weight = trainer.model.learnable_graph_builder(final_poi_emb)

    final_graph_info = {
        'edge_index': final_edge_index.cpu(),
        'edge_weight': final_edge_weight.cpu(),
        'poi_embeddings': final_poi_emb.cpu(),
        'num_edges': final_edge_index.size(1),
        'num_nodes': poi_loader.locations(),
        'candidate_graph': candidate_graph,
        'k_geo': setting.k_geo,
        'k_freq': setting.k_freq,
        'use_mlp_edge': setting.use_mlp_edge,
        'edge_weight_method': "MLP-based" if setting.use_mlp_edge else "dot-product"
    }

    final_graph_file = setting.candidate_file.replace('.pkl', '_final.pkl')
    with open(final_graph_file, 'wb') as f:
        pickle.dump(final_graph_info, f)
    log_string(log, f'Final graph saved to {final_graph_file}')
# 最终结果记录
log_string(log, f'Best accuracy achieved: {best_acc:.6f}')

# 关闭日志文件
log.close()

# 控制台输出
print(f'Training completed with best accuracy: {best_acc:.6f}')
print(f'Training log saved to: {setting.log_file}')
print(f'Best model saved to: {setting.model_files}')

if setting.use_learnable_graph:
    final_graph_file = setting.candidate_file.replace('.pkl', '_final.pkl')
    print(f'Final graph saved to: {final_graph_file}')

# completion message
components = []
if setting.use_learnable_graph:
    edge_method = "MLP-based" if setting.use_mlp_edge else "dot-product"
    components.append(f"{edge_method} learnable graph")
else:
    components.append("no graph")

if setting.use_interest_fusion:
    components.append(f"interest fusion ({setting.fusion_type})")

print(f'Enhanced LoTNext training completed with: {" + ".join(components)}!')