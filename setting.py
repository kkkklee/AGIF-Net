import torch
import argparse
import sys

from network import RnnFactory


class Setting:
    """Defines all settings in a single place using a command line interface."""

    def parse(self):
        self.guess_foursquare = any(['4sq' in argv for argv in sys.argv])
        self.guess_wuhan = any(['wuhan' in argv for argv in sys.argv])
        self.guess_tokyo = any(['tky' in argv for argv in sys.argv])  # 新增
        self.guess_nyc = any(['nyc' in argv for argv in sys.argv])  # 新增

        parser = argparse.ArgumentParser()
        if self.guess_foursquare:
            self.parse_foursquare(parser)
        elif self.guess_wuhan:
            self.parse_wuhan(parser)
        elif self.guess_tokyo:  # 新增
            self.parse_tokyo(parser)
        elif self.guess_nyc:  # 新增
            self.parse_nyc(parser)
        else:
            self.parse_gowalla(parser)
        self.parse_arguments(parser)
        args = parser.parse_args()

        # training
        self.gpu = args.gpu
        self.hidden_dim = args.hidden_dim
        self.weight_decay = args.weight_decay
        self.learning_rate = args.lr
        self.epochs = args.epochs
        self.rnn_factory = RnnFactory(args.rnn)
        self.is_lstm = self.rnn_factory.is_lstm()
        self.lambda_t = args.lambda_t
        self.lambda_s = args.lambda_s

        self.transformer_nhid = args.transformer_nhid
        self.transformer_nlayers = args.transformer_nlayers
        self.transformer_nhead = args.transformer_nhead
        self.transformer_dropout = args.transformer_dropout
        self.attention_dropout_rate = args.attention_dropout_rate
        self.time_embed_dim = args.time_embed_dim
        self.user_embed_dim = args.user_embed_dim

        self.logit_adj_post = args.logit_adj_post
        self.tro_post_range = args.tro_post_range
        self.logit_adj_train = args.logit_adj_train
        self.tro_train = args.tro_train

        # data management
        self.dataset_file = './data/{}'.format(args.dataset)
        self.friend_file = './data/{}'.format(args.friendship)
        self.loader_file = './data/{}'.format(args.dataloader)
        self.max_users = 0
        self.sequence_length = 20
        self.batch_size = args.batch_size
        self.min_checkins = 101

        # evaluation
        self.validate_epoch = args.validate_epoch
        self.report_user = args.report_user

        # log
        self.log_file = args.log_file
        self.model_file = args.model_file

        self.lambda_user = args.lambda_user
        self.lambda_loc = args.lambda_loc
        self.use_weight = args.use_weight

        # learnable graph parameters
        self.use_learnable_graph = args.use_learnable_graph
        self.candidate_file = args.candidate_file
        self.k_geo = args.k_geo
        self.k_freq = args.k_freq
        self.graph_update_freq = args.graph_update_freq
        self.use_mlp_edge = args.use_mlp_edge
        self.top_k_neighbors = args.top_k_neighbors

        # interest fusion parameters
        self.use_interest_fusion = args.use_interest_fusion
        self.fusion_type = args.fusion_type
        self.init_alpha = args.init_alpha


        self.use_enhanced_loss = args.use_enhanced_loss

        # CUDA Setup
        self.device = torch.device('cpu') if args.gpu == -1 else torch.device('cuda', args.gpu)

    def parse_arguments(self, parser):
        # training
        parser.add_argument('--gpu', default=0, type=int, help='the gpu to use')
        parser.add_argument('--hidden-dim', default=10, type=int, help='hidden dimensions to use')
        parser.add_argument('--weight_decay', default=0, type=float, help='weight decay regularization')
        parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
        parser.add_argument('--epochs', default=60, type=int, help='amount of epochs')
        parser.add_argument('--rnn', default='rnn', type=str, help='the GRU implementation to use: [rnn|gru|lstm]')

        parser.add_argument('--transformer_nhid', type=int, default=32, help='Hid dim in TransformerEncoder')
        parser.add_argument('--transformer_nlayers', type=int, default=2, help='Num of TransformerEncoderLayer')
        parser.add_argument('--transformer_nhead', type=int, default=2, help='Num of heads in multiheadattention')
        parser.add_argument('--transformer_dropout', type=float, default=0.3, help='Dropout rate for transformer')
        parser.add_argument('--attention_dropout_rate', type=float, default=0.1, help='Dropout rate for attention')
        parser.add_argument('--time_embed_dim', type=int, default=32, help='Time embedding dimensions')
        parser.add_argument('--user_embed_dim', type=int, default=128, help='User embedding dimensions')
        parser.add_argument('--logit_adj_post', help='adjust logits post hoc', type=int, default=1, choices=[0, 1])
        parser.add_argument('--tro_post_range', help='check different val of tro in post hoc', type=list,
                            default=[0.25, 0.5, 0.75, 1, 1.5, 2])
        parser.add_argument('--logit_adj_train', help='adjust logits in training', type=int, default=1, choices=[0, 1])
        parser.add_argument('--tro_train', default=1.0, type=float, help='tro for logit adj train')

        # data management
        parser.add_argument('--dataset', default='checkins-4sq.txt', type=str,
                            help='the dataset under ./data/<dataset.txt> to load')
        parser.add_argument('--dataloader', default='poi_loader-4sq.pkl', type=str,
                            help='the dataloader under ./data/<dataloader.pkl> to load')
        parser.add_argument('--friendship', default='4sq_friend.txt', type=str,
                            help='the friendship file under ../data/<edges.txt> to load')

        # evaluation
        parser.add_argument('--validate-epoch', default=5, type=int,
                            help='run each validation after this amount of epochs')
        parser.add_argument('--report-user', default=-1, type=int,
                            help='report every x user on evaluation (-1: ignore)')

        # log
        parser.add_argument('--log_file', default='./results/log_4sq', type=str, help='store result logs')
        parser.add_argument('--model_file', default='./model_log/model_4sq', type=str, help='store model logs')

        parser.add_argument('--lambda_loc', default=1.0, type=float, help='weight factor for location')
        parser.add_argument('--lambda_user', default=1.0, type=float, help='weight factor for user')
        parser.add_argument('--use_weight', default=False, type=bool, help='whether to use weight in GCN')

        # learnable graph parameters
        parser.add_argument('--use_learnable_graph', default=False, type=bool, help='whether to use learnable graph')
        parser.add_argument('--candidate_file', default='./data/candidate_graph.pkl', type=str,
                            help='candidate graph file path')
        parser.add_argument('--k_geo', default=60, type=int, help='number of geographic neighbors')
        parser.add_argument('--k_freq', default=20, type=int, help='number of neighbors after frequency filtering')
        parser.add_argument('--graph_update_freq', default=5, type=int, help='graph update frequency (every N epochs)')
        parser.add_argument('--use_mlp_edge', default=True, type=bool, help='whether to use MLP for edge weights')
        parser.add_argument('--top_k_neighbors', default=10, type=int, help='final number of neighbors per POI')

        # interest fusion parameters
        parser.add_argument('--use_interest_fusion', default=True, type=bool,
                            help='whether to enable interest drift modeling')
        parser.add_argument('--fusion_type', default='gate', type=str, choices=['linear', 'gate'],
                            help='interest fusion method: linear or gate')
        parser.add_argument('--init_alpha', default=0.7, type=float, help='initial alpha for linear fusion')

        parser.add_argument('--use_enhanced_loss', default=True, type=bool,
                            help='使用增强损失函数处理长尾POI问题')

    def parse_gowalla(self, parser):
        # defaults for gowalla dataset
        parser.add_argument('--batch-size', default=128, type=int,
                            help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=1000, type=float, help='decay factor for spatial data')

    def parse_foursquare(self, parser):
        # defaults for foursquare dataset
        parser.add_argument('--batch-size', default=256, type=int,
                            help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')

    def parse_wuhan(self, parser):
        # defaults for wuhan dataset
        parser.add_argument('--batch-size', default=128, type=int,
                            help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=500, type=float, help='decay factor for spatial data')

    def parse_tokyo(self, parser):
        # defaults for tokyo dataset
        parser.add_argument('--batch-size', default=256, type=int,
                            help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')

    def parse_nyc(self, parser):
        # defaults for nyc dataset
        parser.add_argument('--batch-size', default=256, type=int,
                            help='amount of users to process in one pass (batching)')
        parser.add_argument('--lambda_t', default=0.1, type=float, help='decay factor for temporal data')
        parser.add_argument('--lambda_s', default=100, type=float, help='decay factor for spatial data')