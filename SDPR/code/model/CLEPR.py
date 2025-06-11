# -*- coding: utf-8 -*-
# @Time    : 2024/1/10 10:18
# @Author  :
# @File    : CLEPR.py
# @Description :  the pytorch version of CLEPR

#os 和 sys 用于操作系统和系统路径相关的功能。

#helper 和 batch_test 是自定义的工具模块，可能包含了一些辅助函数和批次测试相关的功能。

#datetime 用于时间相关的操作。

#numpy 用于数值计算。

#torch.nn 和 torch 用于构建 PyTorch 神经网络模型。

#SelfAttention 从 Attention_layer.py 导入，表示一个自注意力机制的实现。

#math 用于数学计算。
import os
import sys
from utils.helper import *
from utils.batch_test import *
# from utils.batch_test_case_study import *
import datetime
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from model.Attention_layer import SelfAttention
import math

class CLEPR(nn.Module):
    def __init__(self, data_config, pretrain_data):
        super(CLEPR, self).__init__()
        self.model_type = 'CLEPR'#存储模型类型（这里是 'CLEPR'）。
        #分别是邻接矩阵的类型和算法类型，args.adj_type 和 args.alg_typ是从命令行参数中读取的。
        self.adj_type = args.adj_type
        self.alg_type = args.alg_type

        self.pretrain_data = pretrain_data#用于存储预训练的数据（如果有的话）。
        self.n_users = data_config['n_users']#存储用户数量。
        self.n_items = data_config['n_items']#存储物品数量。

        self.n_fold = 100#用于存储邻接矩阵的折叠数量。
        self.norm_adj = data_config['norm_adj']#存储归一化后的邻接矩阵。
        self.sym_pair_adj = data_config['sym_pair_adj']#存储对称配对邻接矩阵。
        self.herb_pair_adj = data_config['herb_pair_adj']#存储草药配对邻接矩阵。
        self.n_nonzero_elems = self.norm_adj.count_nonzero()#计算归一化邻接矩阵中非零元素的数量。
        self.lr = args.lr#存储学习率。
        # self.link_lr = args.link_lr
        self.emb_dim = args.embed_size#存储嵌入维度。
        self.batch_size = args.batch_size#存储批次大小。
        self.loss_weight = args.loss_weight#存储损失权重。
#self.lr 是学习率，self.emb_dim 是嵌入维度，self.batch_size 是每批次大小，
# self.loss_weight 是损失权重，这些值来自于命令行参数。
        self.weight_size = eval(args.layer_size)#存储权重大小。
        self.n_layers = len(self.weight_size)#存储层数。
        self.device = args.device#存储设备。


        self.fusion = args.fusion#存储融合方法。
        print('***********fusion method************ ', self.fusion)

        self.mlp_predict_weight_size = eval(args.mlp_layer_size)#存储MLP预测的权重大小。eval转换为python列表
        self.mlp_predict_n_layers = len(self.mlp_predict_weight_size)#存储MLP预测的层数。计算 MLP 的层数，就是列表的长度。
        print('mlp predict weight ', self.mlp_predict_weight_size)
        print('mlp_predict layer ', self.mlp_predict_n_layers)
        '''self.model_type 初始值是 'CLEPR'，代表模型的基本类型。

self.adj_type 是邻接矩阵的类型（例如 'norm'、'plain' 等）。

self.alg_type 是算法类型（例如 'CLEPR'、'SMGCN' 等）。

self.n_layers 是图卷积网络的层数。

这行代码把这些参数格式化成字符串，拼接到 self.model_type 后面，比如：

python
复制
编辑
'CLEPR_norm_CLEPR_l3'
表示这是 CLEPR 模型，使用规范化邻接矩阵，算法类型是 CLEPR，网络有3层。'''
        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers)#存储模型类型。

        self.regs = eval(args.regs)#存储正则化参数。
        print('regs ', self.regs)
        self.decay = self.regs[0]#存储衰减参数。
        self.verbose = args.verbose#存储是否启用详细输出。

        '''
        *********************************************************
        Create embedding for Input Data & Dropout.
        '''

        self.mess_dropout = args.mess_dropout#存储消息丢弃率。


        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights)
        """
        # initialization of model parameters
        self.weights = self._init_weights()#初始化模型参数。

        """
        *********************************************************
        self-attention for item embedding and item set embedding
        """
        if args.attention == 1:#如果启用注意力机制。
            self.attention_layer = SelfAttention(self.mlp_predict_weight_size_list[0],
                                                 attn_dropout_prob=args.attn_dropout_prob)#初始化注意力层。

    # 初始化权重，存在all weight字典中，键为权重的名字，值为权重的值
    def _init_weights(self):
        # xavier init,使用 Xavier 初始化方法初始化用户和药材的嵌入（user_embedding 和 item_embedding）。
        initializer = nn.init.xavier_uniform_#定义了初始化函数为 Xavier均匀初始化
        all_weights = nn.ParameterDict()#创建一个 参数字典，用来存放模型所有可训练参数。
        #初始化用户和药材的嵌入矩阵，形状是[n_users, emb_dim] 和 [n_items, emb_dim]
        # 使用 initializer 以 Xavier 均匀分布初始化它们。并用 nn.Parameter 包装，告诉PyTorch这是模型的可训练参数
        #用键 'user_embedding' 将其存入参数字典 all_weights。
        all_weights.update({'user_embedding': nn.Parameter(initializer(torch.empty(self.n_users, self.emb_dim)))})
        all_weights.update({'item_embedding': nn.Parameter(initializer(torch.empty(self.n_items, self.emb_dim)))})
        #如果没有预训练数据，使用 Xavier 初始化；如果有预训练数据，则加载预训练的嵌入数据。
        if self.pretrain_data is None:
            print('using xavier initialization')
        else:
            # pretrain
            all_weights['user_embedding'].data = self.pretrain_data['user_embed']
            all_weights['item_embedding'].data = self.pretrain_data['item_embed']
            print('using pretrained initialization')
#将嵌入维度和每层的大小组合成一个列表，weight_size_list 用于存储每层的输入和输出大小。
        self.weight_size_list = [self.emb_dim] + self.weight_size    # [embedding size(64), layer_size(128, 256)]
    #获取最后一层的维度，用于计算每对用户和药材的表示。    
        pair_dimension = self.weight_size_list[len(self.weight_size_list) - 1]
    #为每层图卷积网络（GCN）初始化权重矩阵，w_gc_user 和 b_gc_user 分别是用户信息的卷积权重和偏置项。
        for k in range(self.n_layers):
            #创建一个形状为[2×d_k,d_k+1]的空张量，作为第k层权重矩阵，d_k是第k层输入特征维度，d_k+1是第k+1层输出特征维度
            # 权重矩阵宽度是输入维度的两倍，原因是图卷积时输入特征通常由当前节点特征与邻居聚合特征拼接（列拼接）而成，维度翻倍。
            w_gc_user = torch.empty([2 * self.weight_size_list[k], self.weight_size_list[k + 1]])
            #创建了一个形状为[1,d_k+1]的空张量，作为该层的偏置向量。作用是在矩阵乘法后加上偏置，实现线性变换的平移。
            b_gc_user = torch.empty([1, self.weight_size_list[k + 1]])
           
            W_gc_item = torch.empty([2 * self.weight_size_list[k], self.weight_size_list[k + 1]])
            b_gc_item = torch.empty([1, self.weight_size_list[k + 1]])
            #这两行代码是在初始化图卷积网络中第 k 层用于构建邻居信息的权重矩阵
            # Q_user 和 Q_item 都是方阵，形状为[d_k,d_k]，其中d_k是第k层输入特征维度。
            # Q_user：用于用户节点邻居信息的线性变换。Q_item：用于物品（药材）节点邻居信息的线性变换。
            Q_user = torch.empty([self.weight_size_list[k], self.weight_size_list[k]])
            Q_item = torch.empty([self.weight_size_list[k], self.weight_size_list[k]])
        #将初始化后的权重矩阵和偏置项更新到 all_weights 字典中。每条语句向 all_weights 字典里 添加（或更新）一个键值对。
            all_weights.update({'W_gc_user_%d' % k: nn.Parameter(initializer(w_gc_user))})    # w,b 第K层聚合user信息的权重矩阵
            all_weights.update({'b_gc_user_%d' % k: nn.Parameter(initializer(b_gc_user))})
           
            all_weights.update({'W_gc_item_%d' % k: nn.Parameter(initializer(W_gc_item))})   # w, b 第K层聚合item信息的权重矩阵
            all_weights.update({'b_gc_item_%d' % k: nn.Parameter(initializer(b_gc_item))})
            all_weights.update({'Q_user_%d' % k: nn.Parameter(initializer(Q_user))})      # 第K层构建user邻居信息时的权重矩阵
            all_weights.update({'Q_item_%d' % k: nn.Parameter(initializer(Q_item))})    # 第K层构建item邻居信息时的权重矩阵
#更新 MLP 层的权重大小列表。
        self.mlp_predict_weight_size_list = [self.mlp_predict_weight_size[
                                                 len(self.mlp_predict_weight_size) - 1]] + self.mlp_predict_weight_size
        print('mlp_predict_weight_size_list ', self.mlp_predict_weight_size_list)
#初始化 MLP 层的权重矩阵和偏置项。
        for k in range(self.mlp_predict_n_layers):
            W_predict_mlp_user = torch.empty([self.mlp_predict_weight_size_list[k], self.mlp_predict_weight_size_list[k + 1]])
            b_predict_mlp_user = torch.empty([1, self.mlp_predict_weight_size_list[k + 1]])
            all_weights.update({'W_predict_mlp_user_%d' % k: nn.Parameter(initializer(W_predict_mlp_user))})
            all_weights.update({'b_predict_mlp_user_%d' % k: nn.Parameter(initializer(b_predict_mlp_user))})
            all_weights.update({'W_predict_mlp_item_%d' % k: nn.Parameter(initializer(W_predict_mlp_user))})
            all_weights.update({'b_predict_mlp_item_%d' % k: nn.Parameter(initializer(b_predict_mlp_user))})
        print("\n", "#" * 75, "pair_dimension is ", pair_dimension)
        #这两个张量用于更新维度，使得嵌入表示可以用于后续计算
        M_user = torch.empty([self.emb_dim, pair_dimension])
        M_item = torch.empty([self.emb_dim, pair_dimension])
        '''
        M_user 是一个形状为 [emb_dim, pair_dimension] 的矩阵，
emb_dim 是输入用户嵌入的维度（例如64），
pair_dimension 是最后一层图卷积输出的维度（例如256或者更大），
        在图卷积模块中，通过用户-用户配对邻接矩阵对用户嵌入进行邻居聚合后，
        得到的张量是temp = torch.sparse.mm(sym_pair_adj, user_embedding)
        这个 temp 的形状仍然是 [n_users, emb_dim]，但是它表示的是邻居用户聚合后的特征。
接下来通过 M_user 做线性映射：
user_pair_embeddings = torch.tanh(torch.matmul(temp, M_user))
        把聚合的邻居特征从原始维度（emb_dim）映射到更高维或更合适的特征空间（pair_dimension）。
让模型可以学习到更复杂的特征组合和配对关系。
通过激活函数 tanh 引入非线性，提高表达能力。

'''
        all_weights.update({'M_user': nn.Parameter(initializer(M_user))})
        all_weights.update({'M_item': nn.Parameter(initializer(M_item))})
        return all_weights

    # todo: 矩阵分解，加速计算
    #该方法将 SciPy 的稀疏矩阵转换成 PyTorch 稀疏张量，用于GPU计算。
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()#tocoo()转成坐标格式，提取行列索引和值。
        #将索引和数据转换成 PyTorch 的张量并放到设备（GPU/CPU）上。
        i = torch.tensor([coo.row, coo.col], dtype=torch.long).to(args.device)
        # v = torch.from_numpy(coo.data).float().to(args.device)
        v = torch.tensor(np.array(coo.data), dtype=torch.float32).to(args.device)
        return torch.sparse.FloatTensor(i, v, coo.shape)#返回稀疏浮点张量，方便后续稀疏矩阵乘法。

    # todo: 矩阵分解，加速计算
    '''这里的邻接矩阵 X 代表用户和物品的关系图。
维度是 (n_users + n_items) x (n_users + n_items)，非常大。
直接用整张矩阵做图卷积计算，计算量大且显存消耗高。
对于第 i_fold 块，取矩阵从 start 到 end 的所有行。
这些行对应图中节点的一部分邻居关系（因为矩阵的行代表节点）。
然后调用 _convert_sp_mat_to_sp_tensor 将这一部分稀疏矩阵转换为 PyTorch 稀疏张量，
方便后续 GPU 加速矩阵乘法。
返回一个列表，每个元素是一块稀疏张量。
在图卷积计算时，分别用这些稀疏张量和节点的嵌入做乘法，分块计算，再合并结果，既节省了显存，也方便并行。
'''
    def _split_A_hat(self, X):
        A_fold_hat = []
        #将大邻接矩阵按行分成 n_fold 份，避免一次性计算时内存爆炸。
        fold_len = (self.n_users + self.n_items) // self.n_fold#计算每个折叠块的行数
        for i_fold in range(self.n_fold):#对总行数分成n_fold块，遍历每个块的索引
            start = i_fold * fold_len#当前块的起始行索引
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items#最后一块取到矩阵末尾
            else:
                end = (i_fold + 1) * fold_len#非最后一块取对应块的终止行索引
#每份取矩阵 X 的[start:end] 行，通过 _convert_sp_mat_to_sp_tensor 转换为稀疏张量后存入列表。放到设备（GPU）上            
            # 返回切分后的稀疏张量列表
            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]).to(args.device))
        return A_fold_hat

    # 使用图卷积神经网络得到的user embedding用户嵌入
    def _create_graphsage_user_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)   # 将矩阵按行分为n_hold份，每份矩阵存在列表元素中
       #用户和药材的初始嵌入矩阵合并，按行拼接，方便统一计算。形状是[n_users + n_items, emb_dim]
        pre_embeddings = torch.cat([self.weights['user_embedding'], self.weights['item_embedding']], 0)  # [n_user+n_item, B]

        # print("*" * 20, "embeddings", pre_embeddings)
        all_embeddings = [pre_embeddings]#保存所有层的嵌入，初始包含原始嵌入。

    #分块计算邻居聚合，提升计算效率，最后拼接回完整形状。    
        for k in range(self.n_layers):
            temp_embed = []
            for f in range(self.n_fold):
                 # 对拆分的邻接块分别与嵌入矩阵做稀疏乘法（邻居聚合）
                temp_embed.append(torch.sparse.mm(A_fold_hat[f], pre_embeddings))   # 矩阵分解相乘，加速计算
            # 分解的矩阵拼回成一个, 每行表示邻居节点传来的信息，得到完整的邻居信息
            embeddings = torch.cat(temp_embed, 0)   # 前n_user行表示与该user相关的item邻居传递来的信息和其本身，后n_item行同理
            # 邻居信息通过Q_user权重矩阵非线性变换（先做矩阵乘法再用tanh激活函数处理）
            #Q_user_%d 是以字符串格式命名的权重键，%d 会被当前层索引 k 替代。每层 k 都有一个对应的 Q_user_k 权重矩阵。
# 维度是 [当前层输入维度, 当前层输入维度]，即方阵，大小与该层输入的特征维度相同。
            embeddings = torch.tanh(torch.matmul(embeddings, self.weights['Q_user_%d' % k]))   # 构建邻居信息
    
            '''这里的 pre_embeddings 是当前层的节点自身表示（上一层输出或初始嵌入）。

embeddings 是该层邻居信息经过 Q_user_k 变换后的结果（邻居聚合信息）。

通过列拼接，相当于把自身特征和邻居特征拼成一个更长的特征向量，让后续的线性变换（W_gc_user_k）可以综合考虑这两部分信息，进行更复杂的非线性映射。

这符合 GraphSAGE 设计思想：用节点自己和邻居特征的拼接作为输入，捕获更丰富的信息。'''
             # 消息聚合，# 将上一层节点自身embedding与邻居信息拼接，做消息聚合
            embeddings = torch.cat([pre_embeddings, embeddings], 1)  

            pre_embeddings = torch.tanh(
                torch.matmul(embeddings, self.weights['W_gc_user_%d' % k]) + self.weights['b_gc_user_%d' % k])# 计算高阶节点表示（GCN层）
            pre_embeddings = nn.Dropout(self.mess_dropout[k])(pre_embeddings)# 使用dropout防止过拟合

            norm_embeddings = F.normalize(pre_embeddings, p=2, dim=1)#L2归一化
            all_embeddings = [norm_embeddings]#存储归一化结果
#将所有层的嵌入拼接成一个完整的嵌入矩阵。得到用户和药材的最终嵌入。按用户和药材分开，按列拼接
        all_embeddings = torch.cat(all_embeddings, 1)
        '''torch.split 按第0维（行维度）把这个大张量拆成两部分：
第一部分长度为 self.n_users，对应用户的嵌入矩阵 u_g_embeddings，形状为(n_users+n_items,emb_dim)
torch.split 按第0维（行维度）把这个大张量拆成两部分：
第一部分长度为 self.n_users，对应用户的嵌入矩阵 u_g_embeddings，形状为(n_users,emb_dim)
第二部分长度为 self.n_items，对应药材的嵌入矩阵 i_g_embeddings，形状为(n_items,emb_dim)'''
        # 但由于是使用了user的权重矩阵, 所以这里仅将user的embedding拿出来计算
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], 0)# 分离用户和药材embedding
# # 基于用户-用户关联矩阵计算邻居信息
        temp = torch.sparse.mm(self._convert_sp_mat_to_sp_tensor(self.sym_pair_adj).to(args.device),
                                             self.weights['user_embedding'])      # 利用user-user图计算user的embedding
        # 进一步映射得到pair嵌入,乘以 M_user 权重矩阵并用 tanh 激活，进一步映射用户间配对关系的特征。
        user_pair_embeddings = torch.tanh(torch.matmul(temp, self.weights['M_user']))
#根据融合策略将两部分用户嵌入融合。加法融合：特征向量逐元素相加。拼接融合：特征向量在列方向拼接，维度翻倍。
        if self.fusion in ['add']:
            u_g_embeddings = u_g_embeddings + user_pair_embeddings
        if self.fusion in ['concat']:
            u_g_embeddings = torch.cat([u_g_embeddings, user_pair_embeddings], 1)
        return u_g_embeddings # 返回融合后的用户嵌入
#与上述用户嵌入生成类似，生成药材的嵌入。
    def _create_graphsage_item_embed(self):
        #与用户嵌入生成类似，只是权重矩阵换成针对item的。
        A_fold_hat = self._split_A_hat(self.norm_adj)

        pre_embeddings = torch.cat([self.weights['user_embedding'], self.weights['item_embedding']], 0)

        all_embeddings = [pre_embeddings]
        for k in range(self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(torch.sparse.mm(A_fold_hat[f], pre_embeddings))
            embeddings = torch.cat(temp_embed, 0)

            embeddings = torch.tanh(torch.matmul(embeddings, self.weights['Q_item_%d' % k]))
            embeddings = torch.cat([pre_embeddings, embeddings], 1)

            pre_embeddings = torch.tanh(
                torch.matmul(embeddings, self.weights['W_gc_item_%d' % k]) + self.weights['b_gc_item_%d' % k])

            pre_embeddings = nn.Dropout(self.mess_dropout[k])(pre_embeddings)

            norm_embeddings = F.normalize(pre_embeddings, p=2, dim=1)
            all_embeddings = [norm_embeddings]

        all_embeddings = torch.cat(all_embeddings, 1)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], 0)
#利用药材间共现关系矩阵，进一步强化药材的嵌入。
        temp = torch.sparse.mm(self._convert_sp_mat_to_sp_tensor(self.herb_pair_adj).to(args.device),
                                             self.weights['item_embedding'])
        item_pair_embeddings = torch.tanh(torch.matmul(temp, self.weights['M_item']))
#返回融合后的药材嵌入
        if self.fusion in ['add']:
            i_g_embeddings = i_g_embeddings + item_pair_embeddings

        if self.fusion in ['concat']:
            i_g_embeddings = torch.cat([i_g_embeddings, item_pair_embeddings], 1)

        return i_g_embeddings
#pos_items：物品嵌入矩阵；user_embeddings：用户嵌入矩阵；pos_scores：预测的用户对物品的评分概率
    def create_batch_rating(self, pos_items, user_embeddings):
        #pos_items.transpose(0, 1)将物品嵌入矩阵转置，用于后续点积计算，
        # 得到相似度矩阵后，用sigmoid函数将值映射到（0,1）
        pos_scores = torch.sigmoid(torch.matmul(user_embeddings, pos_items.transpose(0, 1)))
        return pos_scores

    def get_self_correlation(self, item_embeddings):
        #B：批大小（batch size），即一批有多少个药方。max_item_len：药方中药材的最大数量（经过padding）。emb:每个药材的嵌入维度。
        """
        Args:
            item_embeddings:  [B, max_item_len, emb] 两者相乘，减去对角阵后即为每个药方内部，两个中药的相似度矩阵，将这个相似度矩阵的和作为
            对比学习的一个约束加入loss中，这个loss需要调参，使得两两之间的相似度比较大
        Returns:
            self_correlation_matrix: 自相关矩阵的
        """
        #这里的“两者相乘”指的是用余弦相似度计算两个药材嵌入之间的相似度矩阵，矩阵中的每个元素表示药方内两个药材的相似度。
        # 减去对角矩阵，是为了排除药材自身与自身的相似度（通常为1），只关注不同药材之间的相似关系。
        # 
        # 这个相似度矩阵的总和（经过平方和归一化）作为对比学习的正则项，加入到模型训练的损失函数中。
        # 该约束帮助模型学习到同一药方内的药材嵌入之间应该更相似，提升模型效果。

        #这个约束的权重（即超参数）需要调节，控制其对训练的影响大小，以达到理想的相似度效果。
        #分别在第2维和第1维插入一个1，来扩充向量维度
        # 然后两两药材嵌入在最后一维做余弦相似度，得到每个药方内药材对的相似度矩阵。
        cor_matrix = torch.cosine_similarity(item_embeddings.unsqueeze(2), item_embeddings.unsqueeze(1), dim=-1)   # [B, max_len, max_len]
        diag = torch.diagonal(cor_matrix, dim1=1, dim2=2)#torch.diagonal()用于取矩阵对角线元素
        a_diag = torch.diag_embed(diag)#根据输入元素构建对角矩阵
        cor_matrix = cor_matrix - a_diag   # 对角元素置为0，即自己和自己不去计算相似度
        cor_matrix = cor_matrix.pow(2)     # 平方值得值大于0
        cor_value = torch.sum(cor_matrix) / 2    # 计算整个相似度矩阵的总和，padding的部分是0，所以不会计算到 这个值越小越好
        #返回的是“自相关矩阵”的一个值，这里其实是对药方内所有药材对的相似度总和的一个数值（标量），作为训练时的对比学习损失的一部分。
        return cor_value

#这部分实现了对一批药方中药材集合的嵌入计算，结合自注意力机制和位置编码
#输入一个批次的药材ID集合（已经padding，统一长度），返回该药材集合的整体嵌入表示item_padding_set 
# 并输出注意力得分item_attention_scores、经过注意力层后的药材嵌入和自相关值cor_value
    def get_set_embedding(self, item_padding_set, ia_embeddings):
        """
        Args:
            item_padding_set: list: [B, max_batch_item_len]  item id set并被padding后 列表id
            ia_embeddings: [n_items, emb]  所有药材的基础嵌入矩阵。
        过程：
        ia_embeddings -- > [n_item+1, emb] 最后一行是padding的嵌入表示
        Returns: set_embedding [B, emb]
        """
        #在药材嵌入最后添加一行全零向量，作为padding的嵌入，防止索引越界。
        padding_embedding = torch.zeros((1, ia_embeddings.size(1)), dtype=torch.float32).to(args.device)#创建一个全0张量
        ia_padding_embedding = torch.cat((ia_embeddings, padding_embedding), 0)  # [n_item + 1, emb]，将药材嵌入矩阵和上面的全0张量按行拼接
        #根据ID列表获取每个药方的所有药材嵌入，形状为批次大小 × 最大药材数量 × 嵌入维度。
        item_embeddings = ia_padding_embedding[item_padding_set, :]  # [B, max_batch_item_len, emb]
        #批量太大时跳过自相关计算，或者当超参数关闭时跳过。
        # 否则计算药材集合内部两两相似度的正则损失。
        if item_embeddings.size(0) > 1024 or args.co_lamda == 0.0:  # valid and test
            cor_value = 0
        else:
            cor_value = self.get_self_correlation(item_embeddings)   # value
        #构建药材集合中每个位置的编码，帮助模型识别序列顺序。注意代码中位置编码先计算了标准的sin/cos编码，但最后用的是简化版本的位置编码pe = 1 / (position_ids + 1)     
        position_ids = torch.arange(data_generator.max_item_len, dtype=torch.long, device=args.device).unsqueeze(1)
        #获取 item_embeddings 张量的第2维的大小，也就是嵌入向量的维度
        d_model = item_embeddings.size(2)
        #创建一个张量，初始全0，max_item_len是序列最大长度（药方最大药材数），d_model是嵌入向量维度，这个张量用来存储所有位置的编码。
        pe = torch.zeros(data_generator.max_item_len, d_model, device=args.device)
        #torch.arange(0.0, d_model, 2)：生成偶数索引序列 [0, 2, 4, ..., d_model-2]，长度是 d_model/2。
        # -(math.log(10000.0) / d_model)是一个负的缩放因子。这是频率衰减因子，保证不同维度编码的角频率按指数衰减，
        # 频率从低到高递增，给不同维度编码不同频率。div_term 形状是[d_model,/2]，用于后续和位置相乘。
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model)).to(args.device)
        #position_ids 是位置索引张量形状是[max_item_len,1],每行是位置号（0, 1, 2...）。
        #position_ids * div_term 通过广播机制计算每个位置对应每个频率的乘积。
        # torch.sin(...) 计算正弦值，得到偶数维度位置编码。
        pe[:, 0::2] = torch.sin(position_ids * div_term)#pe[:, 0::2] 表示选择 pe 中所有位置的 偶数索引维度，如第0维、第2维、第4维...
        #pe[:, 1::2] 表示选择 pe 中所有位置的 奇数索引维度，如第1维、第3维、第5维...
        # 用同样的频率乘积计算余弦值，作为奇数维度的位置编码。
        pe[:, 1::2] = torch.cos(position_ids * div_term)
        #这里使用的是简化版的 位置编码
        #这个表达式就是：位置越靠后，编码值越小，位置越靠前，编码值越大（因为除以位置+1）。
        # 它比sin/cos计算简单，易实现，也能简单表达“位置顺序”的信息。
        # pe[:, :] = 1 / (position_ids.T + 1)
        pe = 1 / (position_ids + 1)
        position_embedding = pe.repeat(item_embeddings.shape[0], 1, 1)  # 位置编码 [B, max_batch_item_len, emb]
        #根据药材嵌入生成掩码，避免padding位置参与注意力计算。
        attention_mask, value_attention_mask, presci_adj_matrix = self.get_attention_mask(item_embeddings)
    #如果开启了自注意力，药材嵌入经过自注意力层，增强不同药材间的交互关系。返回更新后的嵌入和注意力权重。    
        if args.attention == 1:
            item_embeddings, item_attention_scores = self.attention_layer(item_embeddings,
                                                                          attention_mask=attention_mask,
                                                                          value_attention_mask=value_attention_mask,
                                                                          presci_adj_matrix=presci_adj_matrix
                                                                          )  # 经过self attention 层[B, max_batch_item_len, emb]
#计算有效药材数目（不包括padding）并归一化。对药材嵌入做加和，然后除以有效药材数，实现平均池化。
            neigh = torch.sum(value_attention_mask, dim=2) / value_attention_mask.size(2)  # [B, max_len]
            neigh_num = torch.sum(neigh, dim=1)  # [B, 1]
            item_set_embedding = torch.sum(item_embeddings, dim=1)  # [B, emb]
            normal_matrix_item = torch.reciprocal(neigh_num)
            normal_matrix_item = normal_matrix_item.unsqueeze(1)
            # 复制embedding_size列  [B, embedding_size]
            extend_normal_item_embeddings = normal_matrix_item.repeat(1, item_set_embedding.shape[1])
            # 对应元素相乘
            item_set_embedding = torch.mul(item_set_embedding, extend_normal_item_embeddings)  # 平均池化
            #返回药方整体的嵌入表示、注意力权重、更新后的药材嵌入、以及自相关值。
            return item_set_embedding, item_attention_scores, item_embeddings, cor_value

    def get_attention_mask(self, item_seq):
        #该函数为Transformer类自注意力机制构造掩码，
        # 作用是让模型“忽略”padding的无效位置，防止注意力分数被污染。
        # 通过矩阵乘法得到位置对的有效掩码，非常适合序列到序列的自注意力结构。


        """Generate attention mask for attention."""
        #item_seq 是药方中药材的嵌入序列。
        # 判断每个位置是否是非padding（padding通常是0），得到一个布尔矩阵，再转为浮点数（1表示有效，0表示padding）。
        # 维度是 [batch_size,max_len,emb]，也就是转化为0或1的形式
        attention_mask = (item_seq != 0).to(dtype=torch.float32)  # [B, max_len, emb]
        # attention_p = attention_mask[0][0]
        #计算每个批次中，每个特征维度上非padding的位置数量。dim=1指按列计算，emb才是特征维度
        item_len = torch.sum(attention_mask, dim=1)   # [B, emb]
        #取出第一个特征维度的非padding数量，形状变为 [B,1]，表示每个药方有效药材的长度。注意这里没有在后续使用。
        #[:,:1]是取所有行第0列到第1列（不包含第1列）元素     
        item_len_matrix = item_len[:, :1]  # [B, 1]  得到每个药方的长度
        #这行通过批量矩阵乘法计算了每个样本中两个位置是否都有效的掩码
        # 每个元素表示位置 i 和位置 j 是否同时有效。
        '''torch.bmm是批量矩阵乘法，用于对一批矩阵进行矩阵乘法运算。
        输入是两个三维张量，形状分别是[B,N,M]和[B,M,P]，输出是[B,N,P]
        permute对维度进行了重新排列，将[B, max_len, emb]变为[B, emb, max_len]'''
        extended_attention_mask = torch.bmm(attention_mask, attention_mask.permute(0, 2, 1))  # [B, max_len, max_len]
        #将掩码转为0/1浮点数矩阵，1代表有效，0代表至少有一个位置是padding。
        extended_attention_mask = (extended_attention_mask > 0).to(dtype=torch.float32)
        #转为浮点数，方便和注意力分数相加。
        # 将掩码反转，padding位置对应1变为0，非padding位置对应0变为1，然后乘以极大负数−10000，
        # 目的是在softmax计算时使padding位置得分趋近负无穷，权重接近0。
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility\
        #把掩码取反（有效位置0，padding位置1），乘以大负数，让padding位置在softmax计算时权重趋近于0。
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        #占位符，表示这里没有额外的先验邻接矩阵。
        presci_adj_matrix = None
        # attention_mask = (1.0 - attention_mask) * - 10000.0
        #返回反转后的大负数掩码（用于softmax前加权），
        # 返回初始的有效位置掩码（1和0），返回空的先验邻接矩阵。
        return extended_attention_mask, attention_mask, presci_adj_matrix

#这段代码实现了“困难负样本（hard negative samples）”的生成和评分计算，
# 目的是在训练过程中帮助模型更有效地区分正负样本，提升对比学习效果。
    def get_hard_sample(self, user_embeddings, ia_embeddings, model_two_stage=None):
        step = args.step#每次取的top-k负样本数量。
        max_step_len = args.max_step_len#负样本采样的最大范围长度。
        random_id = [id for id in range(0, max_step_len-step)]#采样起始索引范围。
        #调用create_batch_rating计算用户对所有物品的评分。
        # 根据评分降序排序，得到评分值和对应物品索引。
        rating = self.create_batch_rating(ia_embeddings, user_embeddings)\
        #vals 记录了从高到低排序后的评分值，indices 记录了对应评分在原数据中的索引号，
        vals, indices = rating.sort(descending=True)
        #初始化负样本评分变量。
        # 遍历不同起点，分段取top-k负样本。
        logits_user_neg = None
        k_id = 0
        for k in random_id:
            # topK_items = indices[:, k:]
            #从排序结果中取每个用户的top-k负样本（连续的k个）。
            #k 和 step 是整数，k 是当前切片的起始位置，step 是每次选取的物品数量。
            topK_items = indices[:, k:k+step]
            #如果用注意力机制：
            # 为保证集合长度统一，用padding索引（一般是物品总数索引，代表全零向量）填充topK集合。
            # 维持统一的序列长度，方便自注意力处理。
            if args.attention == 1:
                # padding = torch.tensor([data_generator.n_items] * (k)).to(args.device)
                #创建一个长度为 data_generator.max_item_len - k 的一维张量，
                # 每个元素都是 data_generator.n_items，
                # 通常是所有药材数的索引范围之外的一个特殊索引，代表padding。
                padding = torch.tensor([data_generator.n_items] * (data_generator.max_item_len-k)).to(args.device)
                #给 padding 增加一个维度，变成二维张量，形状是 [长度, 1]。
                padding = padding.unsqueeze(1)
                #padding.repeat(1, topK_items.size(0)) 是在第1维复制列，
                # topK_items.size(0) 是批大小 B。
                # 复制次数是 topK_items 的行数（即批大小 B），变成形状 [长度, B]。
                # .transpose(0, 1) 交换第0维和第1维，变成形状 [B, 长度]。
                padding = padding.repeat(1, topK_items.size(0)).transpose(0, 1)
                #将原来的 topK_items（形状 [B, step]）和刚构造的 padding（形状 [B, 长度]）按列拼接，
                # 这样 topK_items 的长度变为 step + 长度 ≈ max_item_len，保持批次中所有药方长度相差不多。
                # 之后转成CPU上的numpy数组，再转成列表，方便后续处理。
                topK_items = torch.cat([topK_items, padding], dim=1).cpu().numpy().tolist()
                #调用get_set_embedding函数计算每个topK集合的整体嵌入。
                # 支持两阶段模型，第二阶段模型复用前者的结果。
                if model_two_stage is None:
                    topK_sets, _, topK_att_embedding, _ = self.get_set_embedding(topK_items, ia_embeddings)  # [B, emb]  内存占用过大
                else:
                    topK_sets, _, topK_att_embedding, _ = model_two_stage.get_set_embedding(topK_items, ia_embeddings)  # [B, emb]  内存占用过大
            #未启用注意力时，直接对topK物品嵌入做平均池化，得到集合表示。
            else:
                #将 GPU 上的 topK_items 张量转移到 CPU，转换为 NumPy 数组，再转成 Python 列表。
                topK_items = topK_items.cpu().numpy().tolist()
                #使用 topK_items（列表格式）索引药材嵌入矩阵 ia_embeddings，
                # 得到一个形状为 [B, k, emb] 的张量，表示批次中每个用户选中的 top-k 药材嵌入集合。
                item_set_embeddings = ia_embeddings[topK_items, :]  # [B, k,emb]
                #对嵌入集合沿第1维（top-k维度）求和，得到 [B, emb]。
                # 除以 step 实现平均池化，得到每个用户对应的top-k集合的整体嵌入表示。
                topK_sets = torch.sum(item_set_embeddings, dim=1) / step  # 平均池化
                # for ks in range(0, self.mlp_predict_n_layers):
                #     topK_sets = F.relu(
                #         torch.matmul(topK_sets, self.weights['W_predict_mlp_item_%d' % ks])
                #         + self.weights['b_predict_mlp_item_%d' % ks])
                #     topK_sets = F.dropout(topK_sets, self.mess_dropout[ks])  # [B, emb] 药方归纳, item set的整体表示
            #对用户嵌入与topK集合嵌入做点积评分（内积乘积求和）。
            # 逐步拼接不同k段的评分结果，最终形状为 [B, len(random_id)]。
            if k_id == 0:#当是第一个负样本集合时（k_id == 0），
                #对用户嵌入 user_embeddings 和负样本集合嵌入 topK_sets 做逐元素相乘（Hadamard乘积）
                logits_user_neg = torch.mul(user_embeddings, topK_sets)  # [B, emb]
                #然后沿着嵌入维度（dim=1）求和，得到每个用户对当前负样本集合的匹配得分，形状 [B, 1]，
                # 赋值给 logits_user_neg，作为负样本得分矩阵的第一列。
                logits_user_neg = torch.sum(logits_user_neg, dim=1).unsqueeze(1)  # [B, 1]  症状集合*对应的topk 集合
            else:#对后续每个负样本集合，重复同样的乘积和求和操作，
                neg = torch.mul(user_embeddings, topK_sets)
                #得到当前负样本集合的得分 neg，形状 [B, 1]，
                neg = torch.sum(neg, dim=1).unsqueeze(1)
                #通过 torch.cat 沿第1维（列）拼接到已有的 logits_user_neg 张量中，
                # logits_user_neg 最终形状为 [B, n]，n 是负样本集合的数量。
                logits_user_neg = torch.cat([logits_user_neg, neg], dim=1)  # [B, 12（5-65）] 每一行是每个top k集合的得分
            #计数器递增，标识下一个负样本集合。
            k_id += 1
            #返回所有困难负样本集合对用户的评分。
        return logits_user_neg

#这段代码实现了模型的损失函数计算，包含传统的矩阵分解损失和对比学习的监督对比损失（SupConLoss），结合正则化项。
    '''函数功能整体
输入用户和药材的嵌入及标签，计算：

传统的矩阵分解损失（mf_loss），

嵌入正则化损失（emb_loss），

对比学习的监督对比损失（contrastive_loss），

以及对比学习中的自相关正则项（cor_loss）。

根据参数 use_const 选择是否启用对比学习部分。

'''  
    def create_set2set_loss(self, items, item_weights, user_embeddings, all_user_embeddins,
                            ia_embeddings, item_embeddings, use_const=0, logits_user_neg=None,
                            items_repeat=None, repeat=0, neg_item_embeddings=None, cor_value=0):
        # item_embeddings [B, emd]
        if repeat == 1:#如果 repeat=1，使用重复的items标签。
            #预测概率用用户嵌入和药材嵌入矩阵的内积，通过 sigmoid 映射到 (0,1)。
            items = items_repeat
        predict_probs = torch.sigmoid(torch.matmul(user_embeddings, ia_embeddings.transpose(0, 1)))
        #计算真实标签和预测概率的均方误差（MSE）加权和，除以批量大小归一化。这部分是基础的重构损失。
        mf_loss = torch.sum(torch.matmul(torch.square((items - predict_probs)), item_weights), 0)
        # mf_loss = nn.MSELoss(reduction='elementwise_mean')(items, predict_probs)
        mf_loss = mf_loss / self.batch_size
        all_item_embeddins = ia_embeddings
        #计算用户和药材嵌入的L2范数平方和，用于防止过拟合的权重衰减。乘以权重衰减系数 self.decay。
        regularizer = torch.norm(all_user_embeddins) ** 2 / 2 + torch.norm(all_item_embeddins) ** 2 / 2
        regularizer = regularizer.reshape(1)#将张量内元素加和然后变为一维
        # F.normalize(all_user_embeddins, p=2) + F.normalize(all_item_embeddins, p=2)
        regularizer = regularizer / self.batch_size

        emb_loss = self.decay * regularizer#乘以权重衰减系数 self.decay。
#如果不启用对比学习，直接返回矩阵分解和正则化损失。

#torch.tensor([0.0])：创建一个包含单个元素0.0的一维张量，形状是 [1]
# dtype=torch.float64：指定数据类型为64位浮点数（双精度）。
# requires_grad=True：表示该张量需要计算梯度，支持反向传播优化。
# .to(args.device)：将张量移动到指定设备（如GPU或CPU），确保与模型和数据一致。
        reg_loss = torch.tensor([0.0], dtype=torch.float64, requires_grad=True).to(args.device)
        if use_const == 0:
            #如果 use_const 等于0，直接返回：mf_loss：矩阵分解损失
            # emb_loss：嵌入正则化损失
            # 两个 reg_loss，占位符
            # 不执行对比学习相关计算。
            return mf_loss, emb_loss, reg_loss, reg_loss
        else:#否则执行对比学习相关计算
            """
            user_embeddings: [B, n_e]
            item_embeddings: [B, n_e]
            """
            loss_use_cos = args.save_tail

            # 使用固定的t, 对比学习loss使用sup nce
            #使用温度参数 t 缩放内积。对用户和物品嵌入做L2归一化，计算批内相似度矩阵（余弦相似度）。
            t = torch.tensor(args.t).to(args.device)
            # Normalize to unit vectors
            #计算归一化后的用户嵌入和物品嵌入的矩阵乘积，得到形状 [B, B] 的相似度矩阵（logits）。
            user_embeddings = F.normalize(user_embeddings, p=2, dim=1)
            item_embeddings = F.normalize(item_embeddings, p=2, dim=1)
            #除以温度参数 t，缩放 logits，影响对比损失的“温度”。
            logits_user_batch = torch.matmul(user_embeddings, item_embeddings.transpose(0, 1)) / t  # [B, B]
           #结合困难负样本得分，构成完整的对比logits矩阵。
           # 这段代码主要是处理困难负样本的对比学习logits拼接
            if logits_user_neg is None:#如果没有预先计算的困难负样本得分 logits_user_neg（即为 None），
                #调用 get_hard_sample 函数动态生成困难负样本得分矩阵，形状为 [B, N]，N为困难负样本数量，
                logits_user_neg = self.get_hard_sample(user_embeddings, ia_embeddings)
                #再把原本的用户-物品批内相似度矩阵 logits_user_batch（形状 [B, B]）与困难负样本得分矩阵拼接，
                #拼接后形状是 [B, B+N]，同时包含正样本和困难负样本得分。
                logits_user = torch.cat([logits_user_batch, logits_user_neg], dim=1)  # [B， B+N]
            #如果困难负样本矩阵存在但大小为0（空），直接使用批内相似度矩阵，不拼接困难负样本。
            elif logits_user_neg.size(0) == 0:
                logits_user = logits_user_batch  # [B， B]
                #否则，困难负样本矩阵非空，直接拼接到批内相似度矩阵后面。
            else:
                logits_user = torch.cat([logits_user_batch, logits_user_neg], dim=1)  # [B， B]

            #######################################################################################################
            # 使用SupConLoss： 有监督的对比学习loss， 可以处理多个正例和负例
#使用标签构造 正例掩码矩阵， 计算每个样本与正例的对比损失。
# 通过温度缩放和softmax计算对比概率，最终得到对比损失。

#从数据生成器中取训练标签列表（通常是类别或用户标识），转为列向量形状 [B, 1]，方便后续比较。
            labels = torch.tensor(data_generator.train_positive_list).to(args.device)
            labels = labels.contiguous().view(-1, 1)  # [B, 1] 列向量
            #计算标签之间的相等关系矩阵，相等的标签位置置1，表示正样本对，否则为0，
            # 生成的 mask 用于标记哪些样本是正例对。
            mask = torch.eq(labels, labels.T).float().to(args.device)  # [B, B]
            
            if logits_user_neg.size(0) != 0:#如果存在困难负样本得分，
                #在掩码右侧拼接全零矩阵（形状与负样本得分矩阵相同），
                # 负样本不参与正例掩码，确保掩码和logits维度一致。
                mask = torch.cat((mask, torch.zeros_like(logits_user_neg).to(args.device)), dim=1)  # [B, B + len(neg)]
            # for numerical stability
            #对每一行取最大值，减去最大值，防止exp运算溢出，提高数值稳定性。
            logits_max, _ = torch.max(logits_user, dim=1, keepdim=True)  # 每行的最大值 [B,1]
            logits = logits_user - logits_max.detach()  # [B, B + len(neg)]
            #计算softmax的对数概率，等价于 log(softmax)。这样方便计算对比损失。
            exp_logits = torch.exp(logits)   # [B, B + len(neg)]
            # exp_logits = exp_logits / torch.exp(logits_sim_score)    # 张凯老师那边处理自适应的权重
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
            #只统计正例对的对数概率，计算每个样本正例的平均log概率。
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  # 仅计算正例，分子是正例
            # loss
            #乘以温度参数 t，取负号得到损失。取平均作为batch内的最终对比损失。
            # 用超参数 contrast_reg 调节对比损失权重。
            loss = -t * mean_log_prob_pos
            loss = loss.view(1, logits_user.shape[0]).mean()
            contrastive_loss = loss
            contrastive_loss = args.contrast_reg * contrastive_loss
            #结合对药材集合自相关性的正则约束，提升嵌入表达一致性。
            #计算药材集合自相关正则项，控制药材嵌入内部相似性，乘以系数 co_lamda 加权。
            cor_loss = args.co_lamda * cor_value
            if cor_loss > 0:
                #判断计算得到的自相关正则损失 cor_loss 是否大于0。
                # 如果是，将其赋值给 reg_loss，方便在训练日志中打印显示。
                # 这一步主要是为了观察和监控自相关正则项的数值大小，不会影响训练逻辑。
                reg_loss = cor_loss   # 只是为了打印出来看下值
                #将自相关正则项 cor_loss 加入到监督对比损失 contrastive_loss 中，
                #使得模型在训练时不仅优化对比损失，还要同时考虑药材嵌入的自相关特性。
            contrastive_loss = contrastive_loss + cor_loss
            #返回矩阵分解损失，嵌入正则，协同正则和对比学习损失。
            return mf_loss, emb_loss, reg_loss, contrastive_loss


    def normalize(self, *xs):#定义一个成员函数normalize，接收任意个输入张量xs。
        #用来对输入的一个或多个张量沿最后一个维度做归一化处理。
        #列表推导式遍历所有输入张量x：如果x是None，直接返回None；
        # 否则对x调用F.normalize(x, dim=-1)。
        # F.normalize是PyTorch的归一化函数，这里沿dim=-1（最后一个维度）计算每个向量的L2范数并除以它，实现单位向量归一化。
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]

    def forward(self, users, item_padding_set=None, items=None, user_set=None, pos_items=None, train=True, user_padding_set=None):
        #这段forward函数是模型的前向传播入口，负责计算用户和药材的图卷积嵌入，
        # 核心是通过图神经网络的消息传递机制获得用户和药材的表示。
        """
          *********************************************************
          Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
          Different Convolutional Layers:
              1. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
              2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
              3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
          """
        # todo: todo 应该在主函数中

        #当算法类型是 SMGCN 或 CLEPR 时，且处于训练模式，
        # 调用对应方法计算所有用户和药材的图卷积嵌入，输出分别是所有用户和所有药材的最终嵌入矩阵。
        if self.alg_type in ['SMGCN', 'CLEPR']:
            if train:
                ua_embeddings = self._create_graphsage_user_embed()  # [n_user, 最后一层size的大小]: 所有user的embedding
                ia_embeddings = self._create_graphsage_item_embed()  # [n_item, 最后一层size的大小]
                #如果批次中涉及到的用户集合 user_set 不为空，
                # 使用 torch.index_select 从所有用户嵌入中选取当前批次用户的嵌入，
                # all_user_embeddins 是当前批次用户的嵌入表示。
                if user_set is None:
                    all_user_embeddins = None
                else:
                    all_user_embeddins = torch.index_select(ua_embeddings, 0,
                                                        user_set)  # [一个Batch中涉及到的user个数, 最后一层embedding_size]: 选择一个batch中user的embedding

                # todo:change: 构建item/user set embedding(注释提醒后续会对用户或药材集合做进一步聚合或构建更复杂的嵌入表示。)
                """
                1. 将ia_embedding与multi-hot编码items相乘得到batch中item set的表示 item_set_embeddings
                2. 将item_set_embedding中与user进行相同的平均池化操作
                3. 同样使用一个一层的MLP进行非线性表示，同时使用drop out防止过拟合
                """
                #这段代码实现了对用户和药材集合的表示计算，结合图卷积嵌入与多层感知机（MLP）预测层
                #users 是一个多热编码矩阵（通常是batch_size × n_users），表示批次内每个样本涉及的用户集合，
                # ua_embeddings 是所有用户的嵌入矩阵（n_users × emb_dim），
                # 矩阵乘法得到每个样本用户集合的加权和表示，形状 [B, emb_dim]。
                sum_embeddings = torch.matmul(users, ua_embeddings)  # [B,  最后一层embedding_size]
                #计算每个样本用户集合的大小的倒数（归一化因子），扩展成和嵌入维度相同的形状，
                # 与加权和对应元素相乘，实现平均池化（对用户嵌入求均值）。

                #items 是一个二维张量，形状一般是 [B, max_item_len]，
                # 表示批次中每个样本的药材集合的多热编码（1代表该药材存在，0代表不存在）。
                # torch.sum(items, 1)
                # 对第1维（药材维度）求和，得到一个形状为 [B] 的张量，
                # 表示每个样本中实际包含的药材数量（非0元素的个数）。
                # torch.reciprocal(...)
                # 对上述求和结果求倒数，得到一个形状为 [B] 的张量，表示每个样本的药材数量的倒数。
#这个倒数作为归一化因子，用来后续对药材嵌入求和的结果进行平均池化，防止不同样本因集合大小不同而导致嵌入尺度不一。

                normal_matrix = torch.reciprocal(torch.sum(users, 1))
                normal_matrix = normal_matrix.unsqueeze(1)  # [B, 1]
                # 复制embedding_size列  [B, embedding_size]
                extend_normal_embeddings = normal_matrix.repeat(1, sum_embeddings.shape[1])
                # 对应元素相乘
                user_embeddings = torch.mul(sum_embeddings, extend_normal_embeddings)  # 平均池化 [B, emb]
                #通过多层感知机（MLP）逐层变换用户集合表示，每层先做线性变换，再经过ReLU激活和dropout，
                # 用于非线性映射和防止过拟合。
                for k in range(0, self.mlp_predict_n_layers):
                    user_embeddings = F.relu(
                        torch.matmul(user_embeddings, self.weights['W_predict_mlp_user_%d' % k])
                        + self.weights['b_predict_mlp_user_%d' % k])
                    user_embeddings = F.dropout(user_embeddings, self.mess_dropout[k])  # 证候归纳, user set的整体表示 [B, emb]

                cor_value = 0
                #如果启用注意力，调用 get_set_embedding 计算药材集合的注意力嵌入，
                # 否则对药材集合做平均池化（多热编码乘嵌入矩阵，再归一化），
                # 结果是药材集合的整体表示。
                if args.attention == 1 and item_padding_set is not None:
                    item_embeddings, _, item_att_embedding, cor_value = self.get_set_embedding(item_padding_set,
                                                                ia_embeddings)  # [B,  最后一层embedding_size]
                else:
                    # *********************************** item set  ***********************************
                    item_set_embeddings = torch.matmul(items, ia_embeddings)   # [B,  最后一层embedding_size]
                    normal_matrix_item = torch.reciprocal(torch.sum(items, 1))
                    normal_matrix_item = normal_matrix_item.unsqueeze(1)  # [B, 1]
                    # 复制embedding_size列  [B, embedding_size]
                    extend_normal_item_embeddings = normal_matrix_item.repeat(1, item_set_embeddings.shape[1])
                    # 对应元素相乘
                    item_embeddings = torch.mul(item_set_embeddings, extend_normal_item_embeddings)  # 平均池化
                    # for k in range(0, self.mlp_predict_n_layers):
                    #     item_embeddings = F.relu(
                    #         torch.matmul(item_embeddings, self.weights['W_predict_mlp_item_%d' % k])
                    #         + self.weights['b_predict_mlp_item_%d' % k])
                    #     item_embeddings = F.dropout(item_embeddings, self.mess_dropout[k])      # 药方归纳, item set的整体表示
                #返回：用户集合的非线性表示
                # 批次中所有用户的原始嵌入
                # 所有药材的图卷积嵌入
                # 当前批次药材集合表示
                # 自相关损失值 cor_value
                return user_embeddings, all_user_embeddins, ia_embeddings, item_embeddings, cor_value
            else:
                #这段代码是模型在测试或非训练阶段时的前向计算流程，
                # 主要是计算用户和物品（药材）的图卷积嵌入以及用户集合嵌入，
                # 返回用于预测的用户和目标物品的嵌入表示。

                #调用图卷积网络计算所有用户和所有药材的嵌入，
                # 形状分别为 [n_users, emb_dim] 和 [n_items, emb_dim]。
                ua_embeddings = self._create_graphsage_user_embed()
                ia_embeddings = self._create_graphsage_item_embed()
                #将测试集中目标药材ID转换为张量，放到计算设备上（GPU或CPU）。
                # 用 torch.index_select 从所有药材嵌入中抽取对应的目标药材嵌入，形状 [batch_size, emb_dim]。
                pos_items = torch.tensor(pos_items, dtype=torch.long).to(args.device)
                pos_i_g_embeddings = torch.index_select(ia_embeddings, 0, pos_items)   # 根据test中使用的item id选择item的embedding
                #users 是批次用户的多热编码矩阵，表示每个样本包含哪些用户。
                # sum_embeddings 是对应用户嵌入的加权和。
                # normal_matrix 计算每个样本包含用户数的倒数。
                # 用倒数对加权和进行归一化，实现平均池化，得到用户集合的整体表示。
                sum_embeddings = torch.matmul(users, ua_embeddings)

                normal_matrix = torch.reciprocal(torch.sum(users, 1))

                normal_matrix = normal_matrix.unsqueeze(1)

                extend_normal_embeddings = normal_matrix.repeat(1, sum_embeddings.shape[1])

                user_embeddings = torch.mul(sum_embeddings, extend_normal_embeddings)
#对用户嵌入做多层非线性变换：线性变换 + ReLU激活 + dropout，用于提升表达能力并防止过拟合。
                for k in range(0, self.mlp_predict_n_layers):
                    user_embeddings = F.relu(
                        torch.matmul(user_embeddings,
                                     self.weights['W_predict_mlp_user_%d' % k]) + self.weights[
                            'b_predict_mlp_user_%d' % k])
                    user_embeddings = F.dropout(user_embeddings, self.mess_dropout[k])
                #返回：当前批次用户集合的嵌入表示，形状 [B, emb_dim]；
                # 对应的目标药材嵌入，形状 [B, emb_dim]。
                return user_embeddings, pos_i_g_embeddings












