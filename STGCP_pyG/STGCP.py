import numpy as np

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F
from torch_geometric.utils import dropout_edge
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import Sequential, BatchNorm, InstanceNorm
from typing import Callable, Iterable, Union, Tuple, Optional
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops, remove_self_loops
from random import sample
from functools import partial

from utils import get_feature

class STGCP(nn.Module):
    def __init__(self, 
                hidden_dims, #[512, 30]
                Conv_type = 'GCNConv',
                mask_rate=0.15,#0.15
                drop_edge_rate=0.1,
                mask_token_rate=0.1,
                loss_fn="mse",
                alpha_l=2,
                device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
        super(STGCP, self).__init__()
        self.device = device
        [in_dim, conv_hidden1, conv_hidden2] = hidden_dims #conv_hidden1：512， conv_hidden2：30， in_dim:3000
        self.Conv_type = Conv_type
        self._mask_rate = mask_rate
        self._drop_edge_rate = drop_edge_rate
        self._mask_token_rate = mask_token_rate
        self._replace_rate = 1 - self._mask_token_rate
        self.encoder_to_decoder = nn.Linear(conv_hidden2, conv_hidden2, bias=False)
        # GCN layers
        self.enc_mask_token = nn.Parameter(torch.zeros(1, in_dim))
        self.criterion = self.setup_loss_fn(loss_fn=loss_fn, alpha_l=alpha_l)
        if self.Conv_type == "GCNConv":
            '''https://arxiv.org/abs/1609.02907'''
            from torch_geometric.nn import GCNConv
            self.encoder_conv1 = Sequential('x, edge_index', [
                        (GCNConv(in_dim, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.encoder_conv2 = Sequential('x, edge_index', [
                        (GCNConv(conv_hidden1 * 2, conv_hidden2), 'x, edge_index -> x1'),
                        ])
            self.decoder_conv1 = Sequential('x, edge_index', [
                        (GCNConv(conv_hidden2, conv_hidden1 * 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True),
                        ])
            self.decoder_conv2 = Sequential('x, edge_index', [
                        (GCNConv(conv_hidden1* 2, in_dim), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "SAGEConv":
            from torch_geometric.nn import SAGEConv
            self.encoder_conv1 = Sequential('x, edge_index', [
                        (SAGEConv(in_dim, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.encoder_conv2 = Sequential('x, edge_index', [
                        (SAGEConv(conv_hidden1* 2, conv_hidden2), 'x, edge_index -> x1'),
                        ])
            self.decoder_conv1 = Sequential('x, edge_index', [
                        (SAGEConv(conv_hidden2, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True),
                        ])
            self.decoder_conv2 = Sequential('x, edge_index', [
                        (SAGEConv(conv_hidden1* 2, in_dim), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "GraphConv":
            from torch_geometric.nn import GraphConv
            self.encoder_conv1 = Sequential('x, edge_index', [
                        (GraphConv(in_dim, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.encoder_conv2 = Sequential('x, edge_index', [
                        (GraphConv(conv_hidden1* 2, conv_hidden2), 'x, edge_index -> x1'),
                        ])
            self.decoder_conv1 = Sequential('x, edge_index', [
                        (GraphConv(conv_hidden2, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True),
                        ])
            self.decoder_conv2 = Sequential('x, edge_index', [
                        (GraphConv(conv_hidden1* 2, in_dim), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "GatedGraphConv":
            from torch_geometric.nn import GatedGraphConv
            self.encoder_conv1 = Sequential('x, edge_index', [
                        (GatedGraphConv(in_dim, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.encoder_conv2 = Sequential('x, edge_index', [
                        (GatedGraphConv(conv_hidden1* 2, conv_hidden2), 'x, edge_index -> x1'),
                        ])
            self.decoder_conv1 = Sequential('x, edge_index', [
                        (GatedGraphConv(conv_hidden2, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True),
                        ])
            self.decoder_conv2 = Sequential('x, edge_index', [
                        (GatedGraphConv(conv_hidden1* 2, in_dim), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "ResGatedGraphConv":
            from torch_geometric.nn import ResGatedGraphConv
            self.encoder_conv1 = Sequential('x, edge_index', [
                        (ResGatedGraphConv(in_dim, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.encoder_conv2 = Sequential('x, edge_index', [
                        (ResGatedGraphConv(conv_hidden1* 2, conv_hidden2), 'x, edge_index -> x1'),
                        ])
            self.decoder_conv1 = Sequential('x, edge_index', [
                        (ResGatedGraphConv(conv_hidden2, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True),
                        ])
            self.decoder_conv2 = Sequential('x, edge_index', [
                        (ResGatedGraphConv(conv_hidden1* 2, in_dim), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "TransformerConv":
            from torch_geometric.nn import TransformerConv
            self.encoder_conv1 = Sequential('x, edge_index', [
                        (TransformerConv(in_dim, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.encoder_conv2 = Sequential('x, edge_index', [
                        (TransformerConv(conv_hidden1* 2, conv_hidden2), 'x, edge_index -> x1'),
                        ])
            self.decoder_conv1 = Sequential('x, edge_index', [
                        (TransformerConv(conv_hidden2, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True),
                        ])
            self.decoder_conv2 = Sequential('x, edge_index', [
                        (TransformerConv(conv_hidden1* 2, in_dim), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "TAGConv":
            from torch_geometric.nn import TAGConv
            self.encoder_conv1 = Sequential('x, edge_index', [
                        (TAGConv(in_dim, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.encoder_conv2 = Sequential('x, edge_index', [
                        (TAGConv(conv_hidden1* 2, conv_hidden2), 'x, edge_index -> x1'),
                        ])
            self.decoder_conv1 = Sequential('x, edge_index', [
                        (TAGConv(conv_hidden2, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True),
                        ])
            self.decoder_conv2 = Sequential('x, edge_index', [
                        (TAGConv(conv_hidden1* 2, in_dim), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "ARMAConv":
            from torch_geometric.nn import ARMAConv
            self.encoder_conv1 = Sequential('x, edge_index', [
                        (ARMAConv(in_dim, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.encoder_conv2 = Sequential('x, edge_index', [
                        (ARMAConv(conv_hidden1* 2, conv_hidden2), 'x, edge_index -> x1'),
                        ])
            self.decoder_conv1 = Sequential('x, edge_index', [
                        (ARMAConv(conv_hidden2, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True),
                        ])
            self.decoder_conv2 = Sequential('x, edge_index', [
                        (ARMAConv(conv_hidden1* 2, in_dim), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "SGConv":
            from torch_geometric.nn import SGConv
            self.encoder_conv1 = Sequential('x, edge_index', [
                        (SGConv(in_dim, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.encoder_conv2 = Sequential('x, edge_index', [
                        (SGConv(conv_hidden1* 2, conv_hidden2), 'x, edge_index -> x1'),
                        ])
            self.decoder_conv1 = Sequential('x, edge_index', [
                        (SGConv(conv_hidden2, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True),
                        ])
            self.decoder_conv2 = Sequential('x, edge_index', [
                        (SGConv(conv_hidden1* 2, in_dim), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "MFConv":
            from torch_geometric.nn import MFConv
            self.encoder_conv1 = Sequential('x, edge_index', [
                        (MFConv(in_dim, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.encoder_conv2 = Sequential('x, edge_index', [
                        (MFConv(conv_hidden1* 2, conv_hidden2), 'x, edge_index -> x1'),
                        ])
            self.decoder_conv1 = Sequential('x, edge_index', [
                        (MFConv(conv_hidden2, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True),
                        ])
            self.decoder_conv2 = Sequential('x, edge_index', [
                        (MFConv(conv_hidden1* 2, in_dim), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "RGCNConv":
            from torch_geometric.nn import RGCNConv
            self.encoder_conv1 = Sequential('x, edge_index', [
                        (RGCNConv(in_dim, conv_hidden1* 2, num_relations=3), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.encoder_conv2 = Sequential('x, edge_index', [
                        (RGCNConv(conv_hidden1* 2, conv_hidden2, num_relations=3), 'x, edge_index -> x1'),
                        ])
            self.decoder_conv1 = Sequential('x, edge_index', [
                        (RGCNConv(conv_hidden2, conv_hidden1* 2, num_relations=3), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True),
                        ])
            self.decoder_conv2 = Sequential('x, edge_index', [
                        (RGCNConv(conv_hidden1* 2, in_dim, num_relations=3), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "FeaStConv":
            from torch_geometric.nn import FeaStConv
            self.encoder_conv1 = Sequential('x, edge_index', [
                        (FeaStConv(in_dim, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.encoder_conv2 = Sequential('x, edge_index', [
                        (FeaStConv(conv_hidden1* 2, conv_hidden2), 'x, edge_index -> x1'),
                        ])
            self.decoder_conv1 = Sequential('x, edge_index', [
                        (FeaStConv(conv_hidden2, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True),
                        ])
            self.decoder_conv2 = Sequential('x, edge_index', [
                        (FeaStConv(conv_hidden1* 2, in_dim), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "LEConv":
            from torch_geometric.nn import LEConv
            self.encoder_conv1 = Sequential('x, edge_index', [
                        (LEConv(in_dim, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.encoder_conv2 = Sequential('x, edge_index', [
                        (LEConv(conv_hidden1* 2, conv_hidden2), 'x, edge_index -> x1'),
                        ])
            self.decoder_conv1 = Sequential('x, edge_index', [
                        (LEConv(conv_hidden2, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True),
                        ])
            self.decoder_conv2 = Sequential('x, edge_index', [
                        (LEConv(conv_hidden1* 2, in_dim), 'x, edge_index -> x1'),
                        ])

        elif self.Conv_type == "ClusterGCNConv":
            from torch_geometric.nn import ClusterGCNConv
            self.encoder_conv1 = Sequential('x, edge_index', [
                        (ClusterGCNConv(in_dim, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True), 
                        ])
            self.encoder_conv2 = Sequential('x, edge_index', [
                        (ClusterGCNConv(conv_hidden1* 2, conv_hidden2), 'x, edge_index -> x1'),
                        ])
            self.decoder_conv1 = Sequential('x, edge_index', [
                        (ClusterGCNConv(conv_hidden2, conv_hidden1* 2), 'x, edge_index -> x1'),
                        BatchNorm(conv_hidden1* 2),
                        nn.ReLU(inplace=True),
                        ])
            self.decoder_conv2 = Sequential('x, edge_index', [
                        (ClusterGCNConv(conv_hidden1* 2, in_dim), 'x, edge_index -> x1'),
                        ])

    # def forward(self, x, edge_index):
    #     loss = self.mask_attr_prediction_with_contrastive(x, edge_index)
    #     loss_item = {'loss': loss}
    #     return loss, loss_item


    def forward(self, adata, x, edge_index):
        loss = self.prediction_with_contrastive(adata, x, edge_index)
        loss_item = {'loss': loss}
        return loss, loss_item

    def prediction_with_contrastive(self, adata, x, edge_index):
        adata, use_x_1, use_x_2 = get_feature(adata)
        use_x_1 = torch.FloatTensor(adata.obsm['feat'].copy()).to(self.device)
        use_x_2 = torch.FloatTensor(adata.obsm['feat_a'].copy()).to(self.device)

        # if self._mask_rate > 0:
        #     use_x_1, (mask_nodes_1, keep_nodes_1) = self.encoding_mask_noise(use_x_1)
        #     use_x_2, (mask_nodes_2, keep_nodes_2) = self.encoding_mask_noise(use_x_2)
        # else:
        #     use_x_1 = use_x_1
        #     use_x_2 = use_x_2



        if self._drop_edge_rate > 0:
            use_edge_index, masked_edges = dropout_edge(edge_index, self._drop_edge_rate)
        else:
            use_edge_index = edge_index


        enc_rep_1 = self.encoder_conv1(use_x_1, use_edge_index)
        enc_rep_1 = self.encoder_conv2(enc_rep_1, use_edge_index)
        enc_rep_1 = self.encoder_to_decoder(enc_rep_1)

        enc_rep_2 = self.encoder_conv1(use_x_2, use_edge_index)
        enc_rep_2 = self.encoder_conv2(enc_rep_2, use_edge_index)
        enc_rep_2 = self.encoder_to_decoder(enc_rep_2)


        # if self._concat_hidden:
        #     enc_rep_1 = torch.cat(all_hidden_1, dim=1)
        #     enc_rep_2 = torch.cat(all_hidden_1, dim=1)


        # rep_1 = self.encoder_to_decoder(enc_rep_1)
        # rep_2 = self.encoder_to_decoder(enc_rep_2)
        rep_1 = enc_rep_1
        # print("enc_rep_1大小：", enc_rep_1.shape) #enc_rep_1大小： torch.Size([3460, 30])
        # print("rep_1大小：", rep_1.shape) #rep_1大小： torch.Size([3460, 30])
        rep_2 = enc_rep_2


        # rep_1[mask_nodes_1] = 0
        # rep_2[mask_nodes_1] = 0
        recon_1 = self.decoder_conv1(rep_1, use_edge_index)
        recon_1 = self.decoder_conv2(recon_1, use_edge_index)
        # print("recon_1大小：", recon_1.shape) #recon_1大小： torch.Size([3460, 3000])

        recon_2 = self.decoder_conv1(rep_2, use_edge_index)
        recon_2 = self.decoder_conv2(recon_2, use_edge_index)

        # x_init_1 = x[mask_nodes_1]
        # x_init_2 = x[mask_nodes_2]
        # x_rec_1 = recon_1[mask_nodes_1]
        # x_rec_2 = recon_2[mask_nodes_2]


        # num_graph = torch.sum(news_node).int()


        rec_1 = global_mean_pool(recon_1, None)[0]
        rec_2 = global_mean_pool(recon_2, None)[0]


        loss_c = torch.cosine_similarity(rec_1, rec_2, dim=0)


        loss = self.criterion(use_x_1, recon_1) + self.criterion(use_x_2, recon_2)

        loss -= loss_c
        return loss

    def encoding_mask_noise(self, x):
        if self.enc_mask_token.device != x.device:
            self.enc_mask_token = nn.Parameter(self.enc_mask_token.to(x.device))

        num_nodes = x.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        num_mask_nodes = int(self._mask_rate * num_nodes)

        # random masking
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes: ]

        if self._replace_rate > 0:
            num_noise_nodes = int(self._replace_rate * num_mask_nodes)
            perm_mask = torch.randperm(num_mask_nodes, device=x.device)
            token_nodes = mask_nodes[perm_mask[: int(self._mask_token_rate * num_mask_nodes)]]
            noise_nodes = mask_nodes[perm_mask[-int(self._replace_rate * num_mask_nodes):]]
            noise_to_be_chosen = torch.randperm(num_nodes, device=x.device)[:num_noise_nodes]

            out_x = x.clone()
            out_x[token_nodes] = 0.0
            out_x[noise_nodes] = x[noise_to_be_chosen]
        else:
            out_x = x.clone()
            token_nodes = mask_nodes
            out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token

        return out_x, (mask_nodes, keep_nodes)

    def mask_attr_prediction_with_contrastive(self, adata, x, edge_index):
        # adata, use_x_1, use_x_2 = get_feature(adata)
        # use_x_1 = torch.FloatTensor(adata.obsm['feat'].copy()).to(self.device)
        # use_x_2 = torch.FloatTensor(adata.obsm['feat_a'].copy()).to(self.device)

        use_x_1, (mask_nodes_1, keep_nodes_1) = self.encoding_mask_noise(x)
        use_x_2, (mask_nodes_2, keep_nodes_2) = self.encoding_mask_noise(x)


        if self._drop_edge_rate > 0:
            use_edge_index, masked_edges = dropout_edge(edge_index, self._drop_edge_rate)
        else:
            use_edge_index = edge_index


        enc_rep_1 = self.encoder_conv1(use_x_1, use_edge_index)
        enc_rep_1 = self.encoder_conv2(enc_rep_1, use_edge_index)
        enc_rep_1 = self.encoder_to_decoder(enc_rep_1)

        enc_rep_2 = self.encoder_conv1(use_x_2, use_edge_index)
        enc_rep_2 = self.encoder_conv2(enc_rep_2, use_edge_index)
        enc_rep_2 = self.encoder_to_decoder(enc_rep_2)


        # if self._concat_hidden:
        #     enc_rep_1 = torch.cat(all_hidden_1, dim=1)
        #     enc_rep_2 = torch.cat(all_hidden_1, dim=1)


        # rep_1 = self.encoder_to_decoder(enc_rep_1)
        # rep_2 = self.encoder_to_decoder(enc_rep_2)
        rep_1 = enc_rep_1
        rep_2 = enc_rep_2


        rep_1[mask_nodes_1] = 0
        rep_2[mask_nodes_1] = 0
        recon_1 = self.decoder_conv1(rep_1, use_edge_index)
        recon_1 = self.decoder_conv2(recon_1, use_edge_index)
        recon_2 = self.decoder_conv1(rep_2, use_edge_index)
        recon_2 = self.decoder_conv2(recon_2, use_edge_index)


        x_init_1 = x[mask_nodes_1]
        x_init_2 = x[mask_nodes_2]
        x_rec_1 = recon_1[mask_nodes_1]
        x_rec_2 = recon_2[mask_nodes_2]


        # num_graph = torch.sum(news_node).int()


        rec_1 = global_mean_pool(recon_1, None)[0]
        rec_2 = global_mean_pool(recon_2, None)[0]


        loss_c = torch.cosine_similarity(rec_1, rec_2, dim=0)


        loss = self.criterion(x_rec_1, x_init_1) + self.criterion(x_rec_2, x_init_2)
        loss -= loss_c
        return loss

    def get_latent_representation(self, x, edge_index):
        encoded  = self.encoder_conv1(x, edge_index)
        # print("encoded_1大小：", encoded_1.shape)
        encoded = self.encoder_conv2(encoded, edge_index)
        # print("encoded_2大小：", encoded_2.shape)
        encoded = self.encoder_to_decoder(encoded)
        return encoded

    def setup_loss_fn(self, loss_fn, alpha_l):

        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(self.sce_loss, alpha=alpha_l)
        else:
            raise NotImplementedError
        return criterion
    
    def sce_loss(x, y, alpha=3):
        x = F.normalize(x, p=2, dim=-1)
        y = F.normalize(y, p=2, dim=-1)

        # loss =  - (x * y).sum(dim=-1)
        # loss = (x_h - y_h).norm(dim=1).pow(alpha)

        loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

        loss = loss.mean()
        return loss