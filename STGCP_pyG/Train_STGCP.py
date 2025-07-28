import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp

from STGCP import STGCP
from utils import add_contrastive_label, Transfer_pytorch_Data
import os
import random
import torch
import torch.backends.cudnn as cudnn
cudnn.deterministic = True
cudnn.benchmark = True
import torch.nn.functional as F

#500
def train_STGCP(adata, hidden_dims=[512, 30], n_epochs=500, lr=0.001, key_added='STGCP',
                gradient_clipping=5.,  weight_decay=0.0001, verbose=True, save_loss=False, save_reconstrction=False, loss_fn="mse", alpha_l=2,
                Conv_type = 'GCNConv', device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):
    """\
    Training graph attention auto-encoder.

    Parameters
    ----------
    adata
        AnnData object of scanpy package.
    hidden_dims
        The dimension of the encoder.
    n_epochs
        Number of total epochs in training.
    lr
        Learning rate for AdamOptimizer.
    key_added
        The latent embeddings are saved in adata.obsm[key_added].
    gradient_clipping
        Gradient Clipping.
    weight_decay
        Weight decay for AdamOptimizer.
    save_loss
        If True, the training loss is saved in adata.uns['STAGATE_loss'].
    save_reconstrction
        If True, the reconstructed expression profiles are saved in adata.layers['STAGATE_ReX'].
    device
        See torch.device.

    Returns
    -------
    AnnData
    """

    # seed_everything()

    adata.X = sp.csr_matrix(adata.X)
    
    if 'highly_variable' in adata.var.columns:
        adata_Vars =  adata[:, adata.var['highly_variable']]
    else:
        adata_Vars = adata

    if 'label_CSL' not in adata.obsm.keys():
        add_contrastive_label(adata_Vars)

    if verbose:
        print('Size of Input: ', adata_Vars.shape)
    if 'Spatial_Net' not in adata.uns.keys():
        raise ValueError("Spatial_Net is not existed! Run Cal_Spatial_Net first!")

    data = Transfer_pytorch_Data(adata_Vars)

    model = STGCP(hidden_dims = [data.x.shape[1]] + hidden_dims, Conv_type=Conv_type, loss_fn=loss_fn, alpha_l=alpha_l).to(device)
    data = data.to(device)
    print('device:', device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    loss_list = []
    for epoch in tqdm(range(1, n_epochs+1)):
        model.train()

        optimizer.zero_grad()
        loss, loss_item = model(adata, data.x, data.edge_index)
        loss_list.append(loss_item['loss'])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
        optimizer.step()
    
    model.eval()
    z = model.get_latent_representation(data.x, data.edge_index)
    # print('data.x', data.x)
    # print('data.edge_index', data.edge_index)
    # z, X_embedding = model.get_latent_representation(data.x, data.edge_index)
    
    # STGCP_rep = z.to('cpu').detach().numpy()
    STGCP_rep = z.detach().cpu().numpy()
    # print('STGCP_rep:', STGCP_rep)
    adata.obsm[key_added] = STGCP_rep

    if save_loss:
        adata.uns['STGCP_loss'] = loss_list
    # if save_reconstrction:
    #     ReX = out.to('cpu').detach().numpy()
    #     ReX[ReX<0] = 0
    #     adata.layers['STGCP_ReX'] = ReX

    return adata