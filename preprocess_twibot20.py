import json
from tqdm import tqdm
import ijson
import datetime
import os
import pickle
import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data
import shutil


def split_user_from_node(path):
    file_path = path+r"/raw/node.json"
    save_path = path+r"/raw/user.json"
    if os.path.exists(save_path):
        print("user.json already exists")
        return
    else:
        user_list = []
        with open(file_path, 'r') as f:
            object_iter = ijson.items(f, 'item')
            with open(save_path, 'w') as f2:
                for object in tqdm(object_iter):
                    # if object['id'].startswith('u'):
                    if 'id' in object and object['id'].startswith('u'):
                        user_list.append(object)
                    else:
                        continue
                print(len(user_list))
                json.dump(user_list, f2)


def generate_global_index(path):
    user_df = pd.read_json(path+r"/raw/user.json")
    print('user_df:',user_df.head())
    user_idx = user_df['id']
    uid2index = {uid: index for index, uid in enumerate(user_idx.values)}
    with open(path+"/processed_data/uid2global_index.pkl", "wb") as tf:
        pickle.dump(uid2index, tf)


def split_user_by_interval(path):
    user_df = pd.read_json(path+r"/raw/user.json")
    with open(path+r"/processed_data/uid2global_index.pkl", 'rb') as f:
        uid2global_index = pickle.load(f)
    dictionary = {
        'created_at': user_df['created_at'].apply(lambda x: x.date()),
        'id': user_df['id']
    }
    data = pd.DataFrame(data=dictionary, columns=['created_at', 'id'])
    # 96 users' "created_at" field are null
    date_str = '2006-03-01'
    data['created_at'].fillna(datetime.date(*map(int, date_str.split('-'))), inplace=True)
    assert data['created_at'].isnull().sum() == 0
    # min: 2006 - 03 - 21，max: 2020 - 09 - 05
    date_str_list = [str(year) + "-" + str(month) + "-01" for year in range(2008, 2021) for month in range(1, 13)]
    date_list = [datetime.date(*map(int, date_str.split('-'))) for date_str in date_str_list]
    if not os.path.exists(path+"/graph_data/global_index"):
        os.makedirs(path+"/graph_data/global_index")
    for date in date_list:
        y = data.loc[data['created_at'] < date]
        y = y['id'].values.tolist()
        global_index = [uid2global_index[uid] for uid in y]
        global_index = torch.LongTensor(global_index)
        subpath = path+"/graph_data/global_index/global_index_in_snapshot_{}.pt".format(str(date))
        torch.save(global_index, subpath)
        if len(y) == len(data):
            break


def generate_edge_for_all_snapshot(path):
    print("generate_edge_for_all_snapshot")
    edge_index_path = path+r'/processed_data/edge_index.pt'
    edge_type_path = path+r'/processed_data/edge_type.pt'
    edge_index = torch.load(edge_index_path)
    edge_index = edge_index.t().tolist()
    edge_type = torch.load(edge_type_path)
    if not os.path.exists(path+"/graph_data/edge_index"):
        os.makedirs(path+"/graph_data/edge_index")
    if not os.path.exists(path+"/graph_data/edge_type"):
        os.makedirs(path+"/graph_data/edge_type")

    global_index_dir_path = path+r"/graph_data/global_index/"
    for root, dirs, files in os.walk(global_index_dir_path):
        for file in files:
            global_index = torch.load(os.path.join(root, file))
            global_index = set(global_index.tolist())
            snapshot_edge = []
            index = []
            for idx, edge in enumerate(edge_index):
                if edge[0] in global_index and edge[1] in global_index:
                    snapshot_edge.append(edge)
                    index.append(idx)
            snapshot_edge = torch.LongTensor(snapshot_edge).t()
            snapshot_edge_file_name = file.replace('global_index', 'edge_index')
            snapshot_edge_file_path = os.path.join(path+"/graph_data/edge_index", snapshot_edge_file_name)
            torch.save(snapshot_edge, snapshot_edge_file_path)
            snapshot_edge_type = edge_type[index]
            assert len(snapshot_edge_type) == len(snapshot_edge[0])
            snapshot_edge_type_file_name = file.replace('global_index', 'edge_type')
            snapshot_edge_type_file_path = os.path.join(path+"/graph_data/edge_type", snapshot_edge_type_file_name)
            torch.save(snapshot_edge_type, snapshot_edge_type_file_path)


def generate_position_information_for_all_snapshot(path):
    print("generate_position_information_for_all_snapshot")
    edge_index_dir_path = path+r"/graph_data/edge_index"
    global_index_dir_path = path+r"/graph_data/global_index/"
    clustering_coefficient_dir_path = path+r"/graph_data/position_encoding/clustering_coefficient"
    betweenness_centrality_dir_path = path+r"/graph_data/position_encoding/betweenness_centrality"
    if not os.path.exists(clustering_coefficient_dir_path):
        os.makedirs(clustering_coefficient_dir_path)
    if not os.path.exists(clustering_coefficient_dir_path):
        os.makedirs(betweenness_centrality_dir_path)
    bidirectional_links_ratio_dir_path = path+r"/graph_data/position_encoding/bidirectional_links_ratio"
    if not os.path.exists(bidirectional_links_ratio_dir_path):
        os.makedirs(bidirectional_links_ratio_dir_path)
    for root, dirs, files in os.walk(edge_index_dir_path):
        for file in files:
            edge_index = torch.load(os.path.join(root, file))
            edge_index = edge_index.t().tolist()
            node = torch.load(os.path.join(global_index_dir_path, file.replace('edge_index', 'global_index')))
            G = nx.Graph()
            G.add_nodes_from(node.tolist())
            G.add_edges_from(edge_index)
            clustering_coefficient = nx.clustering(G)
            clustering_coefficient = torch.FloatTensor(list(clustering_coefficient.values()))
            assert clustering_coefficient.shape[0] == node.shape[0]
            file_name = file.replace('edge_index', 'clustering_coefficient')
            file_path = os.path.join(clustering_coefficient_dir_path, file_name)
            torch.save(clustering_coefficient, file_path)

            # G = nx.Graph()
            # G.add_nodes_from(node.tolist())
            # G.add_edges_from(edge_index)
            betweenness_centrality = nx.betweenness_centrality(G)
            betweenness_centrality = torch.FloatTensor(list(betweenness_centrality.values()))
            assert betweenness_centrality.shape[0] == node.shape[0]
            file_name = file.replace('edge_index', 'betweenness_centrality')
            file_path = os.path.join(betweenness_centrality_dir_path, file_name)
            torch.save(betweenness_centrality, file_path)

            edge_type = torch.load(os.path.join(root.replace('edge_index', 'edge_type'), file.replace('edge_index', 'edge_type')))
            edge_index = torch.load(os.path.join(root, file))
            edge_index = edge_index.t()
            edge_index_following = edge_index[torch.where(edge_type == 0)]

            G = nx.DiGraph()
            G.add_nodes_from(node.tolist())
            G.add_edges_from(edge_index_following.tolist())
            output_degree = torch.FloatTensor([val for (node, val) in G.out_degree])
            bidirectional_links = {
                v: 0 for v in G.nodes
            }
            for v in G.nodes:
                for neighbor in G.neighbors(v):
                    if v in G.neighbors(neighbor):
                        bidirectional_links[v] += 1
            bidirectional_links = torch.FloatTensor(list(bidirectional_links.values()))
            bidirectional_links_ratio = bidirectional_links / output_degree
            bidirectional_links_ratio[torch.isnan(bidirectional_links_ratio)] = 0
            assert bidirectional_links_ratio.shape[0] == node.shape[0]
            file_name = file.replace('edge_index', 'bidirectional_links_ratio')
            file_path = os.path.join(bidirectional_links_ratio_dir_path, file_name)
            torch.save(bidirectional_links_ratio, file_path)


def generate_pyg_graph_for_all_snapshot(path):
    print("generate_pyg_graph_for_all_snapshot")
    edge_dir_path = path+r"/graph_data/edge_index"
    graph_dir_path = path+r"/graph_data/graphs"
    global_index_dir_path = path+r"/graph_data/global_index"
    clustering_coefficient_dir_path = path+r"/graph_data/position_encoding/clustering_coefficient"
    betweenness_centrality_dir_path = path+r"/graph_data/position_encoding/betweenness_centrality"
    bidirectional_links_ratio_dir_path = path+r"/graph_data/position_encoding/bidirectional_links_ratio"
    edge_type_dir_path = path+r"/graph_data/edge_type"
    if not os.path.exists(graph_dir_path):
        os.mkdir(graph_dir_path)
    for root, dirs, files in os.walk(edge_dir_path):
        for file in files:
            file_name = file.replace('edge_index', 'graph')
            file_path = os.path.join(graph_dir_path, file_name)
            edge_index = torch.load(os.path.join(edge_dir_path, file))
            edge_type = torch.load(os.path.join(edge_type_dir_path, file.replace('edge_index', 'edge_type')))
            global_index = torch.load(os.path.join(global_index_dir_path, file.replace('edge_index', 'global_index')))
            clustering_coefficient = torch.zeros(229580)
            clustering_coefficient[global_index] = torch.load(os.path.join(clustering_coefficient_dir_path, file.replace('edge_index', 'clustering_coefficient')))

            betweenness_centrality = torch.zeros(229580)
            betweenness_centrality[global_index] = torch.load(os.path.join(betweenness_centrality_dir_path, file.replace('edge_index', 'betweenness_centrality')))

            bidirectional_links_ratio = torch.zeros(229580)
            bidirectional_links_ratio[global_index] = torch.load(os.path.join(bidirectional_links_ratio_dir_path, file.replace('edge_index', 'bidirectional_links_ratio')))
            exist_nodes = torch.zeros(229580)
            exist_nodes[global_index] = 1
            graph = Data(edge_index=edge_index,
                         edge_type=edge_type,
                         exist_nodes=exist_nodes,
                         clustering_coefficient=clustering_coefficient.reshape(-1, 1),
                         betweenness_centrality = betweenness_centrality.reshape(-1,1),
                         bidirectional_links_ratio=bidirectional_links_ratio.reshape(-1, 1),
                         n_id=torch.arange(229580))
            torch.save(graph, file_path)
        print('finished')


def delete_temp_data(path):
    shutil.rmtree(path+"/graph_data/global_index")
    shutil.rmtree(path+"/graph_data/edge_index")
    shutil.rmtree(path+"/graph_data/edge_type")
    shutil.rmtree(path+"/graph_data/position_encoding")


def generate_graph_data(path):
    print("start to generate graph data")
    split_user_by_interval(path)
    print("split_user_by_interval finished")
    generate_edge_for_all_snapshot(path)
    print("generate_edge_for_all_snapshot finished")
    generate_position_information_for_all_snapshot(path)
    print("generate_position_information_for_all_snapshot finished")
    generate_pyg_graph_for_all_snapshot(path)
    print("generate_pyg_graph_for_all_snapshot finished")
    delete_temp_data(path)
    print("delete_temp_data finished")

def train_val_test_mask():
    train_idx=torch.tensor(list(range(8278)),dtype=torch.long)
    val_idx=torch.tensor(list(range(8278,8278+2365)),dtype=torch.long)
    test_idx=torch.tensor(list(range(8278+2365,8278+2365+1183)),dtype=torch.long)
        
    return train_idx,val_idx,test_idx

if __name__ == '__main__':

    train_idx,val_idx,test_idx = train_val_test_mask()
    path = '/mnt/mydata'
    torch.save(train_idx,path+'/processed_data/train_idx.pt')
    torch.save(val_idx,path+'/processed_data/val_idx.pt')
    torch.save(test_idx,path+'/processed_data/test_idx.pt')
    split_user_from_node(path)
    generate_global_index(path)
    generate_graph_data(path)
