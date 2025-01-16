import torch
import os
import sys

import torch.nn as nn

from performer_pytorch import Performer
import torch.nn.functional as F

# 获取当前工作目录
current_dir = os.getcwd()

# 假设Notebook文件位于 'BotDGT' 目录中
# 通过向上导航到项目根目录
project_root = os.path.abspath(current_dir)
print(project_root)

# 将项目根目录添加到系统路径
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "New/SLATE/slate"))


from torch.nn import BCEWithLogitsLoss
from lib import g_to_device, feed_dict_to_device, edge_index_to_adj_matrix
from lib.logger import LOGGER
from lib.encoding import AddSupraLaplacianPE, AddSupraRWPE
from lib.supra import graphs_to_supra, reindex_edge_index

from models import *
import numpy as np
from our_utils import *

class OurModel(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        # Graphs info
        self.num_nodes = args.num_nodes
        self.num_classes = args.num_classes
        self.num_features = args.num_features
        self.time_length = args.time_length
        self.undirected = args.undirected
        self.device = torch.device('cpu')
        # self.device = torch.device(
        #     "cuda" if torch.cuda.is_available() else "cpu"
        # )
        # Training parameters
        self.window = args.window

        self.dim_emb = args.dim_emb
        self.dim_pe = args.dim_pe

        self.norm_lap = args.norm_lap
        self.add_eig_vals = args.add_eig_vals
        self.which = args.which
        self.use_performer = args.use_performer
        # self.use_cross_attn = args.use_cross_attn
        self.aggr = args.aggr
        # Transformer and cross attn parameters
        self.nhead = args.nhead
        self.dropout_trsf = args.dropout_trsf
        self.num_layers_trsf = args.num_layers_trsf
        self.one_hot = args.one_hot
        self.norm_first = args.norm_first
        self.bias_lin_pe = args.bias_lin_pe
        self.p_self_time = args.p_self_time
        self.alpha2 = args.alpha2
        self.alpha1 = args.alpha1
        # SupraLaplacian PE
        self.add_vn = args.add_vn
        self.remove_isolated = args.remove_isolated
        self.isolated_in_transformer = args.isolated_in_transformer
        self.add_time_connection = args.add_time_connection
        self.normalization = "sym"
        self.dim_feedforward = args.dim_feedforward
        self.bceloss = BCEWithLogitsLoss()
        self.build_model()
        if self.use_performer:
            self.flash = False  # Performer does not support flash (not tested )
        assert self.aggr in [
            "mean",
            "sum",
            "max",
            "last",
        ], "Aggregation must be either last, mean, sum or max"
        # assert self.decision in ["mlp", "dot"], "Decision must be either mlp or dot"


    def build_model(self):
        self.num_nodes_embedding = self.num_nodes + 1 if self.add_vn else self.num_nodes
        # Initialize node embedding
        self.node_embedding = nn.Embedding(
            num_embeddings=self.num_nodes_embedding, embedding_dim=self.dim_emb
        )

        # Initialize spatio temporal PE
        self.use_edge_attr = False
        self.supralaplacianPE = AddSupraLaplacianPE(
            k=self.dim_pe,
            normalization=self.norm_lap,
            is_undirected=self.undirected,
            add_eig_vals=self.add_eig_vals,
            which=self.which,
        )

        self.in_dim_pe = 2 * self.dim_pe if self.add_eig_vals else self.dim_pe

        # Initialize linear PE
        self.lin_pe = nn.Linear(self.in_dim_pe, self.in_dim_pe, bias=self.bias_lin_pe)

        # Initialize linear input
        self.lin_input = nn.Linear(
            self.dim_emb + self.in_dim_pe, self.dim_emb, bias=True
        )

        # Initialize projection layer (deprecated)
        self.proj = nn.Linear(self.dim_emb, self.dim_emb, bias=True)

        # # Decision function
        # if self.decision == "mlp" or self.use_cross_attn:
        #     self.pred = LinkPredScore(
        #         dim_emb=self.dim_emb, dropout=self.dropout_dec, edge=self.use_cross_attn
            # )

        # Initialize spatio-temporal attention

        norm = nn.LayerNorm(self.dim_emb)

        if self.use_performer:
            # PROTOTYPE with naive parameters for rebuttal
            print("ENCODER: Performer")
            self.spatio_temp_attn = Performer(
                dim=self.dim_emb,
                depth=self.num_layers_trsf,
                heads=self.nhead,
                causal=False,  # Set to True for autoregressive tasks
                dim_head=self.dim_emb // self.nhead,  # Dimension of each attention head
                ff_mult=self.dim_feedforward
                // self.dim_emb,  # Feedforward network multiplier
            )
        else:
            print("ENCODER: Transformer")
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.dim_emb,
                nhead=self.nhead,
                dropout=self.dropout_trsf,
                dim_feedforward=self.dim_feedforward,
                batch_first=True,
                norm_first=self.norm_first,
            )

            self.spatio_temp_attn = nn.TransformerEncoder(
                encoder_layer, num_layers=self.num_layers_trsf, norm=norm
            )

        # if self.use_cross_attn:
        #     self.cross_attn = CrossAttention(
        #         dim_emb=self.dim_emb,
        #         num_heads=self.nhead_ca,
        #         dropout=self.dropout_ca,
        #         bias=self.bias_ca,
        #         add_bias_kv=self.add_bias_kv,
        #         add_zero_attn=self.add_zero_attn,
        #         light=self.light_ca,
        #     )
        #     # TODO : IMPLEMENT CROSS ATTENTION WITH PERFORMER

        # # Aggregation  (Temporal aggregation in our Figure 2 .d )
        # if self.use_cross_attn:
        #     self.aggregation = EdgeAggregation(self.aggr)
        # else:
        # self.aggregation = NodeAggregation(self.aggr)
    def compute_st_pe(self, graphs):
        """
        Arguments:
            graphs: List of torch_geometric.data.Data
        """
        # First : from graphs, construct a supra adjacency matrix.
        w = len(graphs)

        # print('w:',w)
        # print('num_nodes:',self.num_nodes)


        edge_index, edge_weight, mask = graphs_to_supra_csr(
            graphs,
            self.num_nodes,
            add_vn=self.add_vn,
            add_time_connection=True
        )

        # edge_index, edge_weight, mask = graphs_to_supra(
        #     graphs,
        #     self.num_nodes,
        #     self.add_time_connection,
        #     self.remove_isolated,
        #     add_vn=self.add_vn,
        #     p=self.p_self_time,
        # )

        # Second : if we remove isolated nodes, reindex the edge to compute the laplacian
        if self.remove_isolated:
            # makes the graph connected by snapshot
            edge_index = reindex_edge_index(
                edge_index
            )  # reindex node from 0 to len(torch.unique(edge_index))

        print('edge_index:',edge_index.shape)
        print('edge_weight:',edge_weight.shape)
        print('mask:',mask.shape)

        edge_index = edge_index.to(self.device)
        edge_weight = edge_weight.to(self.device)
        mask = mask.to(self.device)
        # The number of nodes in the supra graph
        num_nodes_supra_adj = (
            len(torch.unique(edge_index))
            if self.remove_isolated
            else self.num_nodes * w
        )

        # Now we have to create a mask to remove the isolated nodes
        total_nodes = sum(self.num_nodes) + self.num_graphs if self.add_vn else sum(self.num_nodes)

        num_nodes_supra_adj = create_sparse_adj_matrix(total_nodes,edge_index)

        new_edge_index = reindex_edge_index(
                edge_index
            )  # Reindex the edge index to remove isolated nodes, Necessary to compute eigenvalues
        num_nodes_supra = len(torch.unique(new_edge_index[0]))
        # len(torch.unique(new_edge_index)) == total_nodes - len(isolated_nodes)

        # Third : compute the supralaplacian PE
        pe = self.supralaplacianPE(new_edge_index, edge_weight, num_nodes_supra_adj)
        if self.add_lin_pe:
            pe = self.lin_pe(pe)

        # Fourth : Construct the token for the transformer
        if not self.isolated_in_transformer:
            raise NotImplementedError  #
            # TODO : TEST WITHOUT ISOLATED NODES IN TRANSFORMER
        else:
            all_pe = torch.zeros((self.num_nodes_embedding * w, self.in_dim_pe)).to(
                self.device
            )
            all_pe[mask] = pe
            node_emb = self.node_embedding(
                torch.arange(self.num_nodes_embedding).to(self.device)
            )
            if self.add_vn:
                tokens = node_emb[:-1, :].repeat(w, 1)
                vn_emb = node_emb[-1].repeat(w, 1)
                tokens = torch.cat(
                    (tokens, vn_emb), dim=0
                )  # Add the virtual nodes at the end of the tokens matrix (easyer to process later)
            else:
                tokens = []
                for i in range(w):
                    tokens.append(node_emb)
                tokens = torch.vstack(tokens)
            tokens = torch.cat((tokens, all_pe), dim=1)
        # Fifth : Project linearly the tokens containing node emb and supraPE
        tokens = self.lin_input(
            tokens
        )  # [N',dim_emb] or [N*W,dim_emb] if integrate isolated node for transformer

        return tokens, mask  

    def forward(self, graphs, train_idx, eval=False):
        """
        Arguments:
            graphs: List of torch_geometric.data.Data
            eval: bool, if True, the model is in evaluation mode (for debug)

        Returns:
            final_emb: Tensor, shape [N, T, F]
        """
        w = len(graphs)
        # 打印每个图的属性
        for i, graph in enumerate(graphs):
            print(f"Graph {i}:")
            print('edge_index:',graph.edge_index.shape)
            print('edge_weight:',graph.edge_weight.shape)
            print('x:',graph.x.shape)
            print('time:',graph.time.shape)
            # print(graph.x.shape)
            print('score_mat:',graph.score_mat)
            print('='*100)
            
            
        # compute the spatio temporal positional encoding
        tokens, mask = self.compute_st_pe(graphs)  # tokens: [N',F]  mask: [W,N]

        # Perform a spatio-temporal full attention # We flat all nodes at w snapshots
        # The idea of SLATE is to consider same node at different snapshot as independant token
        z_tokens = self.spatio_temp_attn(tokens.unsqueeze(0)).squeeze(
            0
        )  # [N',F] # careful, with isolated nodes N' != N*len(graphs)

        if not self.isolated_in_transformer:
            # Remove vn of the token matrix
            # We need to proj the isolated nodes in the same emb space as the z_tokens
            # final_emb = self.proj(final_emb)  # [N, W, F]
            raise NotImplementedError  # TODO : TEST WITHOUT
        else:
            if self.add_vn:
                z_tokens = z_tokens[
                    :-w
                ]  # We dont need virtual nodes in the final embedding for predictions
            z_tokens = z_tokens[train_idx,:,:]
            final_emb = z_tokens.reshape(train_idx.shape[0], w, self.dim_emb)  # [N, W, F]

        print('-'*100)
        print('final_emb:',final_emb.shape)
        print('-'*100)

        

        loss1 = self.infonce_loss(self.proj_u(final_emb[:0,:]), self.proj_u(final_emb[:1,:]),
                                  self.args.temperature)
        loss2 = self.infonce_loss(self.proj_g(final_emb[:0,:]), self.proj_g(final_emb[:2,:]),
                                  self.args.temperature)
        
        if self.training:
            # Training
            train_out = self.out(final_emb)

            train_out = self.classifier(train_out)
            loss = F.cross_entropy(train_out,graph.y[train_idx])
            loss = loss + loss1 * self.alpha1 + loss2 * self.alpha2
            return loss
        

        return final_emb, loss

    # https://blog.csdn.net/weixin_44966641/article/details/120382198
    def infonce_loss(self,
                     emb_i,
                     emb_j,
                     temperature=0.1):  # emb_i, emb_j 是来自同一图像的两种不同的预处理方法得到
        batch_size = emb_i.shape[0]
        negatives_mask = (
            ~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float().to(
                self.args.device).float()  # (2*bs, 2*bs)
        z_i = F.normalize(emb_i, dim=1)  # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)  # (bs, dim)  --->  (bs, dim)

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*bs, dim)
        similarity_matrix = torch.mm(representations, representations.t())

        sim_ij = torch.diag(similarity_matrix, batch_size)  # bs
        sim_ji = torch.diag(similarity_matrix, -batch_size)  # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*bs

        nominator = torch.exp(positives / temperature)  # 2*bs
        denominator = negatives_mask * torch.exp(
            similarity_matrix / temperature)  # 2*bs, 2*bs

        loss_partial = -torch.log(
            nominator / torch.sum(denominator, dim=1))  # 2*bs
        loss = torch.sum(loss_partial) / (2 * batch_size)
        return loss