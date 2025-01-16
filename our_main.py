import os
import sys
from torch_geometric.data import Data
import torch
import numpy as np
import torch.nn as nn


# 获取当前工作目录
current_dir = os.getcwd()

# 假设Notebook文件位于 'BotDGT' 目录中
# 通过向上导航到项目根目录
project_root = os.path.abspath(current_dir)
print(project_root)

# 将项目根目录添加到系统路径
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "New/SLATE/slate"))

from our_model import OurModel

import networkx as nx
import random
import pickle
import warnings

warnings.filterwarnings("ignore")

import argparse

import torch
from torch_geometric.utils import to_undirected
from slate.datasets import Discrete_graph

# 这里的稀疏矩阵都是为了高效计算，采用稀疏矩阵存储和计算
def create_graphs_from_sub_G(sub_G):
    graphs = []
    for data in sub_G:
        G = nx.Graph()
        
        # 将 tensor 转换为 numpy 数组，然后再转换为列表
        edge_index = data.edge_index.numpy().T  # 假设 edge_index 是一个 2xN 的 tensor
        edge_type = data.edge_type.numpy().tolist()
        n_id = data.n_id.numpy().tolist()
        
        # 添加节点
        G.add_nodes_from(n_id)
        
        # 添加边及其类型
        for i, (u, v) in enumerate(edge_index):
            G.add_edge(u, v, type=edge_type[i])
        
        graphs.append(G)
    return graphs

import scipy.sparse as sp

def create_sparse_adj_matrix(G):
    n = len(G.nodes)
    adj_matrix = sp.lil_matrix((n, n), dtype=int)  # 使用稀疏矩阵
    for u, v in G.edges():
        adj_matrix[u, v] = 1
        adj_matrix[v, u] = 1  # 如果是无向图，需要对称
    return adj_matrix

from scipy.sparse import diags

def create_sparse_ones_diagonal_matrix(n):
    # 创建一个长度为n的全1数组
    ones_diagonal = np.ones(n, dtype=int)
    
    # 使用diags函数创建稀疏对角矩阵
    ones_diagonal_matrix = diags(ones_diagonal, format='lil')
    
    # 将稀疏矩阵转换为COO格式
    coo_matrix = ones_diagonal_matrix.tocoo()
    
    # 将COO格式的稀疏矩阵转换为PyTorch稀疏张量
    indices = torch.tensor(np.vstack((coo_matrix.row, coo_matrix.col)), dtype=torch.long)
    values = torch.tensor(coo_matrix.data, dtype=torch.float)
    size = torch.Size(coo_matrix.shape)
    
    sparse_tensor = torch.sparse_coo_tensor(indices, values, size)
    
    return sparse_tensor


def parse_args():
    parser = argparse.ArgumentParser(description="Model Configuration")

    # Graphs info
    parser.add_argument('--num_nodes', type=int, default=100, help='Number of nodes in the graph')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--num_features', type=int, default=32, help='Number of features per node')
    parser.add_argument('--time_length', type=int, default=3, help='Length of the time series')
    parser.add_argument('--undirected', action='store_true', default=False, help='Whether the graph is undirected')
    parser.add_argument('--use_performer', default=False, help='Whether to use user-performer')
    # Training parameters
    parser.add_argument('--window', type=int, default=1, help='Window size for training')
    parser.add_argument('--dim_emb', type=int, default=32, help='Dimension of the embedding')
    parser.add_argument('--dim_pe',type=int, default=4, help='Dimension of the positional encoding')
    parser.add_argument('--norm_lap',type=str, default='sym', help='Type of normalization for laplacian')
    parser.add_argument('--add_eig_vals',type=bool, default=False, help='Type of eigen values and vectors')
    parser.add_argument('--which',type=str, default='SA', help='Type of eigen values and vectors')
    # Transformer and cross attn parameters
    parser.add_argument('--aggr', type=str, default='mean', choices=['mean', 'sum', 'max', 'last'], help='Aggregation method')
    parser.add_argument('--nhead', type=int, default=8, help='Number of heads in the tra    nsformer')
    parser.add_argument('--dropout_trsf', type=float, default=0.1, help='Dropout rate for the transformer')
    parser.add_argument('--num_layers_trsf', type=int, default=6, help='Number of layers in the transformer')
    parser.add_argument('--one_hot', action='store_true', default=False, help='Whether to use one-hot encoding')
    parser.add_argument('--norm_first', action='store_true', default=False, help='Whether to normalize before the attention block')
    parser.add_argument('--bias_lin_pe',default=False, help='Whether to use bias in linear layer')
    parser.add_argument('--dim_feedforward', type=int, default=256, help='Dimension of the feedforward layer in the transformer')
    parser.add_argument('--p_self_time', type=float, default=0.5, help='Dropout rate for the self-attention in the transformer')
    # parser.add_argument('--user_cross_attn', action='store_true', default=False, help='Whether to use cross attention in the transformer')
    # SupraLaplacian PE
    parser.add_argument('--add_vn', action='store_true', default=False, help='Whether to add vertex normals')
    parser.add_argument('--remove_isolated', action='store_true', default=False, help='Whether to remove isolated nodes')
    parser.add_argument('--isolated_in_transformer', action='store_true', default=False, help='Whether to include isolated nodes in the transformer')
    parser.add_argument('--add_time_connection', action='store_true', default=False, help='Whether to add time connections')
    parser.add_argument('--alpha1', type=float, default=0.5, help='Alpha1')
    parser.add_argument('--alpha2', type=float, default=0.5, help='Alpha2')

    return parser.parse_args()


# Example usage
if __name__ == "__main__":
    args = parse_args()
    print(args)
    ################## Data Loading ##################

    sub_G = torch.load('./New/SLATE/slate/data/Twi20/annual_graphs_2008_2020.pt')
    print(f'number of snapshot graphs:{len(sub_G)}')

    # 将 sub_G 转换为 graphs
    graphs = create_graphs_from_sub_G(sub_G)
    G1, G2, G3 = graphs[0], graphs[1], graphs[2]
    print('The size of the top-3 snapshot graphs:',len(G1.nodes),len(G2.nodes),len(G3.nodes))

    # 创建稀疏邻接矩阵
    adj_matrix_1 = create_sparse_adj_matrix(G1)
    adj_matrix_2 = create_sparse_adj_matrix(G2)
    adj_matrix_3 = create_sparse_adj_matrix(G3)

    

    # Convert networkx into pytorch geometric data used in SLATE
    T = 3
    N1, N2, N3 = len(G1.nodes),len(G2.nodes),len(G3.nodes)
    edge_index1 = sub_G[1].edge_index.contiguous()
    edge_index2 = sub_G[2].edge_index.contiguous()
    edge_index3 = sub_G[3].edge_index.contiguous()

    x1 = create_sparse_ones_diagonal_matrix(N1)
    x2 = create_sparse_ones_diagonal_matrix(N2)
    x3 = create_sparse_ones_diagonal_matrix(N3)
    time1 = torch.zeros(len(edge_index1[0]))
    time2 = torch.ones(len(edge_index2[0]))
    time3 = 2 * torch.ones(len(edge_index3[0]))
    weights1 = torch.ones(len(edge_index1[0]))
    weights2 = torch.ones(len(edge_index2[0]))
    weights3 = torch.ones(len(edge_index3[0]))

    dg1 = Discrete_graph(edge_index1, weights1, time1, x1, None)
    dg2 = Discrete_graph(edge_index2, weights2, time2, x2, None)
    dg3 = Discrete_graph(edge_index3, weights3, time3, x3, None)
    graphs = [dg1, dg2, dg3]

    train_idx = torch.tensor([i for i in range(len(G1.nodes))])

    model = OurModel(args)
    print(model)

    out = model(graphs,train_idx)