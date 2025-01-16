import torch
import numpy as np
from torch_geometric.utils import to_undirected
import networkx as nx


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

def graphs_to_supra_csr(graphs, num_nodes, add_vn=False, add_time_connection=False):
    """
    Create a supra adjacency matrix from a list of graphs.

    Arguments:
        graphs: List of torch_geometric.data.Data objects
        num_nodes: List of integers, number of nodes in each graph
        add_vn: Boolean, whether to add virtual nodes
        add_time_connection: Boolean, whether to add temporal connections

    Returns:
        edge_index: Tensor, shape [2, E], supra adjacency matrix edge indices
        edge_weight: Tensor, shape [E], supra adjacency matrix edge weights
        mask: Tensor, shape [N], mask indicating non-isolated nodes
    """
    num_graphs = len(graphs)
    edge_index = []
    edge_weight = []
    l_id_vn = []
    total_nodes = sum(num_nodes)

    # Supra graph creation
    for i in range(num_graphs):
        ei = (
            graphs[i].edge_index + i * num_nodes[i]
        )  # IMPORTANT: We consider nodes in different snapshots as different nodes
        ew = graphs[i].edge_weight

        if add_vn:
            # id_vn = num_nodes[i] * num_graphs + i  # Assign an id to the virtual node
            id_vn = total_nodes + i
            l_id_vn.append(id_vn)  # Necessary to identify the virtual node
            nodes_snapshot = torch.unique(
                ei.view(-1)
            )  # Get the connected nodes in the snapshot
            # Add connections between the virtual node and the nodes (deg > 0) in the snapshot
            # We do not connect the virtual node to isolated nodes
            vn_connections = torch.cat(
                (
                    torch.tensor([id_vn] * len(nodes_snapshot)).view(1, -1),
                    nodes_snapshot.view(1, -1),
                ),
                dim=0,
            )
            ei = torch.cat((ei, vn_connections), dim=1)
            ew = torch.cat((ew, torch.ones(len(nodes_snapshot))))

            print('ei:', ei)
            print('ew:', ew)
            print('='*100)

        if add_time_connection:
            # Add temporal connections between identical nodes in different snapshots
            if i < num_graphs - 1:
                ei_i = graphs[i].edge_index
                ei_next = graphs[i + 1].edge_index
                print('ei_i:', ei_i)
                print('ei_next:', ei_next)
                
                nodes_snapshot = torch.unique(ei_i.view(-1))
                nodes_snapshot_next = torch.unique(ei_next.view(-1))

                print('nodes_snapshot:', nodes_snapshot)
                print('nodes_snapshot_next:', nodes_snapshot_next)
                # Intersection
                common_nodes = torch.LongTensor(
                    np.intersect1d(nodes_snapshot, nodes_snapshot_next)
                )
                # Add temporal connections

                if i < 1:
                    src = common_nodes + i * num_nodes[i]
                    dst = common_nodes + num_nodes[i]
                else:
                    src = common_nodes + i * num_nodes[i-1]
                    dst = common_nodes + sum(num_nodes[:i+1])
                time_co = torch.vstack((src, dst))
                ei = torch.cat((ei, time_co), dim=1)
                ew = torch.cat((ew, torch.ones(len(common_nodes))))

                print('*'*100)
                print('common_nodes:', common_nodes, 'time_co:', time_co)

        edge_index.append(ei)
        edge_weight.append(ew)

    edge_index = torch.cat(edge_index, dim=1)
    edge_weight = torch.cat(edge_weight)

    print(edge_index.shape)

    # Now we have to create a mask to remove the isolated nodes
    total_nodes = total_nodes + num_graphs if add_vn else total_nodes

    print(total_nodes)

    mask = torch.zeros(total_nodes)
    mask[torch.unique(edge_index)] = 1
    # Indices of isolated nodes
    isolated_nodes = torch.where(mask == 0)[0]
    print("Isolated Nodes:", isolated_nodes)
    print("Virtual nodes:", l_id_vn)

    # Make the graph undirected
    edge_index, edge_weight = to_undirected(edge_index, edge_weight)
    edge_weight = torch.ones(edge_index.size(1))

    return edge_index, edge_weight, mask