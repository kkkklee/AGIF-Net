import pickle
import numpy as np
import torch
import random
import torch.nn.functional as F
from math import radians, cos, sin, asin, sqrt
from scipy.sparse import csr_matrix, coo_matrix, identity, dia_matrix
import scipy.sparse as sp
import os
import sys

# 设置默认编码为 UTF-8（跨平台兜底）
import locale
try:
    # Linux/macOS 常见
    locale.setlocale(locale.LC_ALL, 'C.UTF-8')
except locale.Error:
    try:
        # Windows 上使用系统默认区域设置
        locale.setlocale(locale.LC_ALL, '')
    except locale.Error:
        try:
            # 最后退回到 'C'
            locale.setlocale(locale.LC_ALL, 'C')
        except locale.Error:
            pass

# 设置环境变量强制使用 UTF-8
os.environ['PYTHONIOENCODING'] = 'utf-8'

# 设置随机种子
seed = 0
global_seed = 0

# 使用 try-catch 包装随机种子设置
try:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
except Exception as e:
    print(f"Warning: Could not set random seed due to encoding issue: {e}")
    print("Continuing without setting torch random seed...")


def load_graph_data(pkl_filename):
    graph = load_pickle(pkl_filename)  # list
    # graph = np.array(graph[0])
    return graph


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ': ', e)
        raise
    return pickle_data


def calculate_preference_similarity(m1, m2, pref):
    """
        m1: (user_len, hidden_size)
        m2：(user_len, seq_len, hidden_size)
        return: calculate the similarity between user and location, which means user's preference about location
    """
    user_len = m1.shape[0]
    seq_len = m2.shape[1]
    pref = pref.squeeze()  # (1, hidden_size) -> (hidden_size, )
    similarity = torch.zeros(user_len, seq_len, dtype=torch.float32)
    for i in range(user_len):
        v1 = m1[i]
        for j in range(seq_len):
            v2 = m2[i][j]
            similarity[i][j] = (1 + torch.cosine_similarity(v1 + pref, v2, dim=0).item()) / 2  # 归一化到[0, 1]

    return similarity


def compute_preference(m1, m2, pref):
    m1 = (m1 + pref).unsqueeze(1)
    s = m1 - m2
    sim = torch.exp(-(torch.norm(s, p=2, dim=-1)))
    return sim


def get_user_static_preference(pref, locs):
    """
        pref: (user_len, seq_len)
        locs: (user_len, seq_len, hidden_size)
        return: 返回用户对于所访问POI的全局偏好
    """
    user_len, seq_len = pref.shape[0], pref.shape[1]
    hidden_size = locs.shape[2]
    user_preference = torch.zeros(user_len, seq_len, hidden_size)
    for i in range(user_len):
        for j in range(seq_len):  # (hidden_size, )
            user_preference[i][j] = torch.sum(torch.softmax(pref[i, :j + 1], dim=0).unsqueeze(1) * locs[i, :j + 1],
                                              dim=0)
    user_preference = user_preference.permute(1, 0, 2)  # (seq_len, user_len, hidden_size)

    return user_preference


def sampling_prob(prob, label, num_neg):
    num_label, l_m = prob.shape[0], prob.shape[1]  # prob (batch_size, loc_count)
    init_label = torch.zeros(num_label, dtype=torch.int64)  # (batch_size, )
    init_prob = torch.zeros(size=(num_label, num_neg + 1))  # (batch_size, num_neg + 1)

    for batch in range(num_label):
        random_ig = random.sample(range(l_m), num_neg)  # (num_neg) from (0 -- l_max - 1)
        while label[batch].item() in random_ig:  # no intersection
            # print('循环查找')
            random_ig = random.sample(range(l_m), num_neg)

        # place the pos labels ahead and neg samples in the end
        for i in range(num_neg + 1):
            if i < 1:
                init_prob[batch, i] = prob[batch, label[batch]]
            else:
                init_prob[batch, i] = prob[batch, random_ig[i - 1]]

    global global_seed
    random.seed(global_seed)
    global_seed += 1

    return torch.FloatTensor(init_prob), torch.LongTensor(init_label)  # (batch_size, num_neg+1), (batch_size)


def bprLoss(pos, neg, target=1.0):
    loss = - F.logsigmoid(target * (pos - neg))
    return loss.mean()


def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # Radius of earth in kilometers. Use 3956 for miles
    return c * r


def top_transition_graph(transition_graph):
    graph = coo_matrix(transition_graph)
    data = graph.data
    row = graph.row
    threshold = 20

    for i in range(0, row.size, threshold):
        row_data = data[i: i + threshold]
        norm = row_data.max()
        row_data = row_data / norm
        data[i: i + threshold] = row_data

    return graph


def sparse_matrix_to_tensor(graph):
    graph = coo_matrix(graph)
    vaules = graph.data
    indices = np.vstack((graph.row, graph.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(vaules)
    shape = graph.shape
    graph = torch.sparse_coo_tensor(i, v, torch.Size(shape))

    return graph


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()

    return random_walk_mx  # D^-1 W


def calculate_reverse_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def log_string(log, string):
    """安全的日志记录函数"""
    try:
        if not isinstance(string, str):
            string = str(string)

        log.write(string + '\n')
        log.flush()
        print(string)

    except UnicodeEncodeError:
        safe_string = string.encode('utf-8', errors='replace').decode('utf-8')
        log.write(f"[ENCODING_FIXED] {safe_string}\n")
        log.flush()
        print(f"[ENCODING_WARNING] {safe_string}")


# 【新增】支持可学习图的POI相关函数
def get_poi_coordinate_from_loader(poi_id, poi_loader):
    """
    从PoiDataloader获取POI坐标

    Args:
        poi_id: POI的ID
        poi_loader: PoiDataloader实例

    Returns:
        tuple: (latitude, longitude)
    """
    if hasattr(poi_loader, 'poi2gps') and poi_id in poi_loader.poi2gps:
        return poi_loader.poi2gps[poi_id]
    return (0.0, 0.0)  # 默认值


def get_poi_frequency_from_loader(poi_id, poi_loader):
    """
    从PoiDataloader计算POI访问频率

    Args:
        poi_id: POI的ID
        poi_loader: PoiDataloader实例

    Returns:
        int: POI的访问次数
    """
    if not hasattr(poi_loader, '_poi_freq_cache'):
        # 构建频率缓存
        poi_loader._poi_freq_cache = {}
        for user_locs in poi_loader.locs:
            for poi in user_locs:
                poi_loader._poi_freq_cache[poi] = poi_loader._poi_freq_cache.get(poi, 0) + 1

    return poi_loader._poi_freq_cache.get(poi_id, 1)


def build_poi_info_from_loader(poi_loader):
    """
    从PoiDataloader提取所有POI坐标和频率信息

    Args:
        poi_loader: PoiDataloader实例

    Returns:
        tuple: (poi_coords_dict, poi_freq_dict)
    """
    # 坐标信息直接从poi2gps获取
    poi_coords = poi_loader.poi2gps.copy()

    # 计算频率信息
    poi_freq = {}
    for user_locs in poi_loader.locs:
        for poi in user_locs:
            poi_freq[poi] = poi_freq.get(poi, 0) + 1

    return poi_coords, poi_freq


# 【新增】为测试提供的随机生成函数
def generate_random_poi_coordinate(poi_id, lat_range=(30.0, 50.0), lon_range=(-120.0, -80.0)):
    """为测试生成随机POI坐标"""
    import random
    random.seed(poi_id)
    lat = random.uniform(lat_range[0], lat_range[1])
    lon = random.uniform(lon_range[0], lon_range[1])
    return (lat, lon)


def generate_random_poi_frequency(poi_id, freq_range=(1, 100)):
    """为测试生成随机POI频率"""
    return poi_id % (freq_range[1] - freq_range[0]) + freq_range[0]


# 【新增】图构建相关的辅助函数
def validate_candidate_graph(candidate_graph, poi_count):
    """
    验证候选图的有效性

    Args:
        candidate_graph: 候选图字典
        poi_count: POI总数

    Returns:
        bool: 是否有效
    """
    if not isinstance(candidate_graph, dict):
        return False

    for poi_id, neighbors in candidate_graph.items():
        if not isinstance(neighbors, list):
            return False

        # 检查POI ID是否在合理范围内
        if poi_id < 0 or poi_id >= poi_count:
            return False

        # 检查邻居ID是否在合理范围内
        for neighbor_id in neighbors:
            if neighbor_id < 0 or neighbor_id >= poi_count:
                return False

    return True


def calculate_graph_statistics(candidate_graph):
    """
    计算候选图的统计信息

    Args:
        candidate_graph: 候选图字典

    Returns:
        dict: 统计信息
    """
    if not candidate_graph:
        return {'total_nodes': 0, 'total_edges': 0, 'avg_degree': 0, 'max_degree': 0, 'min_degree': 0}

    degrees = [len(neighbors) for neighbors in candidate_graph.values()]
    total_edges = sum(degrees)

    return {
        'total_nodes': len(candidate_graph),
        'total_edges': total_edges,
        'avg_degree': total_edges / len(candidate_graph) if candidate_graph else 0,
        'max_degree': max(degrees) if degrees else 0,
        'min_degree': min(degrees) if degrees else 0
    }


if __name__ == '__main__':
    graph_path = 'data/user_similarity_graph.pkl'
    user_similarity_matrix = torch.tensor(load_graph_data(pkl_filename=graph_path))
    print(user_similarity_matrix[1])
    print('................')
    print(user_similarity_matrix[1][:10])
    count = 0
    print('count: ', count)