#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_saved_model.py
测试已保存的模型脚本
"""

import torch
from torch.utils.data import DataLoader
import numpy as np
import time
import pickle
import os
import sys
import argparse
from setting import Setting
from trainer import FlashbackTrainer
from dataloader import PoiDataloader
from dataset import Split
from utils import *
from network import create_h0_strategy
from evaluation import Evaluation
from collections import defaultdict, Counter

from learnable_graph import CandidateSelector
from interest_fusion import InterestFusionModule


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='测试已保存的LoTNext模型')
    parser.add_argument('--model', type=str,
                        default="./model_log/model_4sq_20250708190518.pth",
                        help='模型文件路径')
    parser.add_argument('--dataloader', type=str,
                        default="./data/poi_loader-4sq.pkl",
                        help='数据加载器文件路径')
    parser.add_argument('--dataset', type=str,
                        default="checkins-4sq.txt",
                        help='数据集名称')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU设备号，-1表示使用CPU')
    parser.add_argument('--output', type=str, default="./test_results.log",
                        help='测试结果输出文件')

    return parser.parse_args()


def load_and_test_model(model_file, dataloader_file, dataset_name, gpu_id, output_file):
    """
    加载保存的模型并进行测试
    """
    print("=== 加载已保存的模型进行测试 ===")
    print(f"模型文件: {model_file}")
    print(f"数据文件: {dataloader_file}")

    # 检查文件是否存在
    if not os.path.exists(model_file):
        print(f" 错误: 模型文件不存在: {model_file}")
        return

    if not os.path.exists(dataloader_file):
        print(f" 错误: 数据文件不存在: {dataloader_file}")
        return

    try:
        # 加载模型检查点
        print("正在加载模型...")
        checkpoint = torch.load(model_file, map_location='cpu')

        # 显示模型信息
        print(f"模型训练轮数: {checkpoint.get('epoch', 'Unknown')}")
        print(f"模型配置: {checkpoint.get('model_config', 'Unknown')}")

        # 重建设置（模拟原始设置）
        setting = Setting()
        setting.dataset_file = f'./data/{dataset_name}'
        setting.loader_file = dataloader_file

        # 设置设备
        if gpu_id == -1:
            setting.device = torch.device('cpu')
        else:
            setting.device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')

        setting.sequence_length = 20
        setting.batch_size = 256  # 4sq 默认是256
        setting.hidden_dim = 10
        setting.is_lstm = False

        # 添加所有必要的 transformer 参数
        setting.transformer_nhid = 32
        setting.transformer_nlayers = 2
        setting.transformer_nhead = 2
        setting.transformer_dropout = 0.3
        setting.attention_dropout_rate = 0.1
        setting.time_embed_dim = 32
        setting.user_embed_dim = 128

        # 添加其他必要参数
        setting.lambda_t = 0.1
        setting.lambda_s = 100
        setting.lambda_loc = 1.0
        setting.lambda_user = 1.0
        setting.use_weight = False
        setting.learning_rate = 0.01
        setting.weight_decay = 0
        setting.epochs = 70
        setting.validate_epoch = 5
        setting.report_user = -1
        setting.min_checkins = 101
        setting.max_users = 0

        # learnable graph 相关参数
        setting.candidate_file = './data/candidate_graph.pkl'
        setting.k_geo = 60
        setting.k_freq = 20
        setting.graph_update_freq = 5
        setting.top_k_neighbors = 10

        # interest fusion 相关参数
        setting.fusion_type = 'gate'
        setting.init_alpha = 0.7

        # 从检查点恢复配置
        setting.use_learnable_graph = checkpoint.get('use_learnable_graph', False)
        setting.use_mlp_edge = checkpoint.get('use_mlp_edge', True)
        setting.use_interest_fusion = checkpoint.get('use_interest_fusion', True)

        print(f"设备: {setting.device}")
        print(f"可学习图: {setting.use_learnable_graph}")
        print(f"兴趣融合: {setting.use_interest_fusion}")

        # 加载数据
        print("正在加载数据...")
        with open(setting.loader_file, 'rb') as f:
            poi_loader = pickle.load(f)

        print(f'POI数量: {poi_loader.locations()}')
        print(f'用户数量: {poi_loader.user_count()}')
        print(f'签到总数: {poi_loader.checkins_count()}')

        # 创建测试数据集
        dataset_test = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TEST)
        dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False)

        print(f'测试数据集批次数: {len(dataset_test)}')

        # 重建trainer（需要与训练时相同的配置）
        print("正在重建模型...")

        # 候选图（如果使用了learnable graph）
        candidate_graph = None
        if setting.use_learnable_graph:
            try:
                candidate_graph = CandidateSelector.load_candidates('./data/candidate_graph.pkl')
                print("已加载候选图")
            except:
                print("⚠  警告: 无法加载候选图，将跳过图功能")
                setting.use_learnable_graph = False

        # 创建trainer
        trainer = FlashbackTrainer(
            lambda_t=0.1,  # 4sq 默认参数
            lambda_s=100,  # 4sq 默认是100，不是1000
            lambda_loc=1.0,
            lambda_user=1.0,
            use_weight=False,
            use_learnable_graph=setting.use_learnable_graph,
            candidate_graph=candidate_graph,
            use_mlp_edge=setting.use_mlp_edge,
            top_k_neighbors=10,
            use_interest_fusion=setting.use_interest_fusion,
            fusion_type='gate',  # 从日志推断
            init_alpha=0.7
        )

        # 创建RNN工厂
        from network import RnnFactory
        rnn_factory = RnnFactory('rnn')  # 从日志推断是RNN

        # 准备模型
        trainer.prepare(poi_loader.locations(), poi_loader.user_count(), setting.hidden_dim,
                        rnn_factory, setting.device, setting)

        # 加载模型权重
        print("正在加载模型权重...")
        trainer.model.load_state_dict(checkpoint['state_dict'])
        trainer.model.to(setting.device)
        trainer.model.eval()

        print(" 模型加载成功！")

        # 创建h0策略
        h0_strategy = create_h0_strategy(setting.hidden_dim, setting.is_lstm)

        # 创建日志文件
        test_log = open(output_file, 'w', encoding='utf-8')
        log_string(test_log, f"模型测试开始: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log_string(test_log, f"模型文件: {model_file}")
        log_string(test_log, f"数据文件: {dataloader_file}")
        log_string(test_log, f"设备: {setting.device}")

        # 创建评估器
        evaluation_test = Evaluation(dataset_test, dataloader_test,
                                     poi_loader.user_count(), h0_strategy, trainer, setting, test_log)

        # 计算调整参数（如果需要）
        try:
            dataset_train = poi_loader.create_dataset(setting.sequence_length, setting.batch_size, Split.TRAIN)

            def compute_adjustment(label_freq, setting):
                label_freq_array = np.array(list(label_freq.values()))
                max_freq = label_freq_array.max()
                tau = 1.2
                adjustments = tau * (1 - (np.log(label_freq_array + 1e-4) / np.log(max_freq + 1e-4)))
                adjustments = torch.from_numpy(adjustments)
                adjustments = adjustments.to(setting.device)
                return adjustments

            logits = compute_adjustment(dataset_train.freq, setting)
        except:
            print("⚠  警告: 无法计算logits调整，使用零向量")
            logits = torch.zeros(poi_loader.locations()).to(setting.device)

        # 开始测试
        print("\n=== 开始测试 ===")
        test_start = time.time()

        with torch.no_grad():
            acc1 = evaluation_test.evaluate(logits, dataset_test)

        test_end = time.time()

        # 关闭日志文件
        log_string(test_log, f"测试完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        log_string(test_log, f"测试耗时: {test_end - test_start:.2f}s")
        log_string(test_log, f"最终准确率: {acc1:.6f}")
        test_log.close()

        # 显示结果
        print(f"\n=== 测试完成 ===")
        print(f"测试时间: {test_end - test_start:.2f}s")
        print(f"最终准确率: {acc1:.6f}")
        print(f"详细测试日志已保存到: {output_file}")

        return acc1

    except Exception as e:
        print(f" 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()

        # 确保日志文件被关闭
        if 'test_log' in locals():
            test_log.close()

        return None


def main():
    """主函数"""
    print("启动模型测试...")

    # 解析命令行参数
    args = parse_arguments()

    print(f"参数配置:")
    print(f"  模型文件: {args.model}")
    print(f"  数据文件: {args.dataloader}")
    print(f"  数据集: {args.dataset}")
    print(f"  GPU设备: {args.gpu}")
    print(f"  输出文件: {args.output}")
    print()

    result = load_and_test_model(args.model, args.dataloader, args.dataset, args.gpu, args.output)

    if result is not None:
        print(f"\n 测试成功完成！")
        print(f"模型性能: recall@1 = {result:.6f}")
    else:
        print(f"\n 测试失败！")


if __name__ == "__main__":
    main()