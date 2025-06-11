# -*- coding: utf-8 -*-                         ## 文件编码声明
# @Time    : 2024/1/10 10:18                   ## 生成/修改时间
# @Author  :                                   ## 作者留空
# @File    : CLEPR.py                          ## 原文件名
# @Description :  the pytorch version of CLEPR ## 简要说明

import os                                      ## Python 标准库：文件/路径
import sys                                     ## Python 标准库：解释器
from utils.helper import *                     ## 项目工具函数
from utils.batch_test import *                 ## 批量测试脚本
# from utils.batch_test_case_study import *    ## 如需案例研究可取消
import datetime                                ## 时间处理
import numpy as np                             ## 科学计算库
import torch.nn as nn                          ## PyTorch 神经网络模块
import torch                                   ## PyTorch 主包
import torch.nn.functional as F                ## 常用函数/激活
from model.Attention_layer import SelfAttention## 自定义 Self-Attention
import math                                    ## 数学工具

class SMGCN(nn.Module):                        ## 主模型：Symptom-Medicine GCN
    def __init__(self, data_config, pretrain_data):
        super(SMGCN, self).__init__()          ## 调用 nn.Module 构造
        self.model_type = 'CLEPR'              ## 基础模型名
        self.adj_type = args.adj_type          ## 邻接矩阵类别
        self.alg_type = args.alg_type          ## 算法类别

        self.pretrain_data = pretrain_data     ## 预训练嵌入(若有)
        self.n_users = data_config['n_users']  ## 症状数量
        self.n_items = data_config['n_items']  ## 中药数量

        self.n_fold = 100                      ## 稀疏图切分块数
        self.norm_adj = data_config['norm_adj']## U-I 归一化图
        self.sym_pair_adj = data_config['sym_pair_adj']  ## 症状-症状
        self.herb_pair_adj = data_config['herb_pair_adj']## 中药-中药
        self.n_nonzero_elems = self.norm_adj.count_nonzero() ## 非零边
        self.lr = args.lr                      ## 学习率
        # self.link_lr = args.link_lr          ## 可选：链接学习率
        self.emb_dim = args.embed_size         ## 嵌入维度
        self.batch_size = args.batch_size      ## 批大小
        self.loss_weight = args.loss_weight    ## 损失权重

        self.weight_size = eval(args.layer_size)  ## 每层输出维度
        self.n_layers = len(self.weight_size)     ## GCN 层数
        self.device = args.device                ## 运算设备

        self.fusion = args.fusion               ## pair 融合策略
        print('***********fusion method************ ', self.fusion)

#args.mlp_layer_size 通常在命令行或配置文件中以字符串形式传入，例如 "64,128,64" 或 "[64, 128, 64]"。
#eval() 将该字符串转换为 Python 列表/元组 → [64, 128, 64]。这里的64,128,64是每一层的特征维度
# 结果保存在实例变量 self.mlp_predict_weight_size，后续用于构造预测 MLP 的线性层。
        self.mlp_predict_weight_size = eval(args.mlp_layer_size)  # 存放MLP每一层输出维度的python列表
        #列表长度 = 需要创建的线性层数量。例如 [64, 128, 64] → 3 层。
        #打印超参数，方便在控制台确认 MLP 结构是否按预期读取。
        print('mlp predict weight ', self.mlp_predict_weight_size)
        print('mlp_predict layer ', self.mlp_predict_n_layers)
        ## 为模型名动态追加标签，记录变体。原始 self.model_type 是 "CLEPR"。
        # adj_type（邻接矩阵类型）、alg_type（算法变体）、l{层数}（GCN 层数）。
        # 例如 adj_type='norm', alg_type='gcn', n_layers=2 ⇒ 结果 "CLEPR_norm_gcn_l2"。
        self.model_type += '_%s_%s_l%d' % (self.adj_type, self.alg_type, self.n_layers) 

        self.regs = eval(args.regs)             ## L2 正则参数
        print('regs ', self.regs)
        self.decay = self.regs[0]               ## 权重衰减系数
        self.verbose = args.verbose             ## 日志开关

        '''
        *********************************************************
        Create embedding for Input Data & Dropout.
        *********************************************************
        '''
        self.mess_dropout = args.mess_dropout   ## 每层消息 Dropout 比例

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights)
        *********************************************************
        """
        self.weights = self._init_weights()     ## 初始化全部可学习参数

        """
        *********************************************************
        self-attention for item embedding and item set embedding
        *********************************************************
        """
        if args.attention == 1:                 ## 若启用 Self-Attention
            self.attention_layer = SelfAttention(
                self.mlp_predict_weight_size_list[0],      ## 隐藏维度
                attn_dropout_prob=args.attn_dropout_prob)  ## dropout

    # 初始化权重，存在all weight字典中，键为权重的名字，值为权重的值
    def _init_weights(self):
        # xavier init
        initializer = nn.init.xavier_uniform_   ## Xavier 均匀初始化
        all_weights = nn.ParameterDict()        ## 用来存放所有可训练参数
        '''nn.Parameter用来将张量转换为可训练参数，并将其添加到模型参数列表中。
        nn.Parameter 继承自 torch.Tensor，但多做两件事：
• 会自动设置 requires_grad=True（除非显式关闭）。
• 只要它被赋值到 nn.Module（或 nn.ParameterDict）的属性中，就会被注册为模型参数，从而：
  • 会出现在 model.parameters() 迭代器里，被优化器更新；
  • 会随 state_dict() 一起保存/加载；
  • 调用 model.to(device) 时会自动搬到对应设备。'''
        all_weights.update({'user_embedding': nn.Parameter(
            initializer(torch.empty(self.n_users, self.emb_dim)))})## 用户嵌入
        all_weights.update({'item_embedding': nn.Parameter(
            initializer(torch.empty(self.n_items, self.emb_dim)))})## 物品嵌入
        if self.pretrain_data is None:
            print('using xavier initialization')## 不提供预训练时
        else:
            # pretrain
            all_weights['user_embedding'].data = self.pretrain_data['user_embed']
            all_weights['item_embedding'].data = self.pretrain_data['item_embed']
            print('using pretrained initialization')       ## 使用预训练嵌入
#这行代码的作用就是把“输入维度”拼到每层隐藏维度列表的最前面，方便后面按 (in_dim, out_dim) 成对地创建每一层的权重矩阵。
        self.weight_size_list = [self.emb_dim] + self.weight_size    # [embedding size(64), layer_size(128, 256)]
        pair_dimension = self.weight_size_list[len(self.weight_size_list) - 1]  ## 最后一层维度
        for k in range(self.n_layers):
            #这里本层输入维度乘2的原因是上一轮消息传递中作者把自身维度与邻居维度拼接起来
            w_gc_user = torch.empty([2 * self.weight_size_list[k], self.weight_size_list[k + 1]])## 第K层聚合user信息的权重矩阵
            b_gc_user = torch.empty([1, self.weight_size_list[k + 1]])## 第K层聚合user信息的偏置
            W_gc_item = torch.empty([2 * self.weight_size_list[k], self.weight_size_list[k + 1]])## 第K层聚合item信息的权重矩阵
            b_gc_item = torch.empty([1, self.weight_size_list[k + 1]])## 第K层聚合item信息的偏置
            Q_user = torch.empty([self.weight_size_list[k], self.weight_size_list[k]])## 第K层构建user邻居信息时的权重矩阵，用于更新邻居向量
            Q_item = torch.empty([self.weight_size_list[k], self.weight_size_list[k]])# 第K层构建item邻居信息时的权重矩阵，用于更新邻居向量
            #将几个参数的键值对添加到all_weights字典中，以便在训练时进行更新。
            all_weights.update({'W_gc_user_%d' % k: nn.Parameter(initializer(w_gc_user))})    # w,b 第K层聚合user信息的权重矩阵
            all_weights.update({'b_gc_user_%d' % k: nn.Parameter(initializer(b_gc_user))})
            all_weights.update({'W_gc_item_%d' % k: nn.Parameter(initializer(W_gc_item))})   # w, b 第K层聚合item信息的权重矩阵
            all_weights.update({'b_gc_item_%d' % k: nn.Parameter(initializer(b_gc_item))})
            all_weights.update({'Q_user_%d' % k: nn.Parameter(initializer(Q_user))})      # 第K层构建user邻居信息时的权重矩阵
            all_weights.update({'Q_item_%d' % k: nn.Parameter(initializer(Q_item))})    # 第K层构建item邻居信息时的权重矩阵

#self.mlp_predict_weight_size[len(self.mlp_predict_weight_size) - 1]是取self.mlp_predict_weight_size列表最后一个元素，即MLP最后一层的输出维度。
#此外，这段代码中，作为将最后一层的输出维度同时当做MLP的输入维度，并进行拼接
# 例：原列表 [64, 128, 64]  → [64] + [...]  → [64, 64, 128, 64]
        self.mlp_predict_weight_size_list = [self.mlp_predict_weight_size[
                                                 len(self.mlp_predict_weight_size) - 1]] + self.mlp_predict_weight_size
        print('mlp_predict_weight_size_list ', self.mlp_predict_weight_size_list)

        for k in range(self.mlp_predict_n_layers):## 逐层创建预测-MLP 参数
            W_predict_mlp_user = torch.empty([self.mlp_predict_weight_size_list[k],# in_dim
                                              self.mlp_predict_weight_size_list[k + 1]]) # out_dim
            b_predict_mlp_user = torch.empty([1, self.mlp_predict_weight_size_list[k + 1]])# 1 × out_dim
            # —— 包成 nn.Parameter 并写入字典 ——
            all_weights.update({'W_predict_mlp_user_%d' % k: nn.Parameter(initializer(W_predict_mlp_user))})
            all_weights.update({'b_predict_mlp_user_%d' % k: nn.Parameter(initializer(b_predict_mlp_user))})
            '''此处物品相关的权重和偏置被注释，原因是用户(症状)和物品(中药)的表示学习采用了不同的路径
用户表示通过GCN和MLP进行转换
而物品表示则直接使用GCN的输出，不再经过额外的MLP层
在forward函数中，我们可以看到物品表示(item_embeddings)是通过简单的平均池化得到的，没有经过MLP层
这种设计可能是为了保持物品表示的原始语义信息，避免过度转换
从中医角度来说，症状(用户)需要更复杂的特征提取来理解其组合关系，而中药(物品)的特征可能相对更直接'''
            # all_weights.update({'W_predict_mlp_item_%d' % k: nn.Parameter(initializer(W_predict_mlp_user))})
            # all_weights.update({'b_predict_mlp_item_%d' % k: nn.Parameter(initializer(b_predict_mlp_user))})

            # 日志输出，打印75个 # 作为分割线，便于在控制台快速定位。pair_dimension 是 self.weight_size_list[-1] —— 即 GCN 最后一层的隐藏维度。
        print("\n", "#" * 75, "pair_dimension is ", pair_dimension)
        #self.emb_dim：原始嵌入维度。- pair_dimension：最后一层 GCN 输出维度。
        # 因此 M_user / M_item 形状都为 [emb_dim, pair_dimension]，用来把 原始节点嵌入 投影到与 pair 图嵌入 相同的维度空间。
        M_user = torch.empty([self.emb_dim, pair_dimension])
        M_item = torch.empty([self.emb_dim, pair_dimension])
        #1. initializer 是 nn.init.xavier_uniform_，先用 Xavier 均匀分布填充权重；
        # 2. nn.Parameter(...) 把张量变成可训练参数（requires_grad=True）；
        all_weights.update({'M_user': nn.Parameter(initializer(M_user))})
        all_weights.update({'M_item': nn.Parameter(initializer(M_item))})
        return all_weights                         ## 返回 ParameterDict


    # todo: 矩阵分解，加速计算                    ## 将 scipy 稀疏矩阵转 PyTorch 稀疏张量
    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()                          ## 转成 COO 格式便于读取索引
        i = torch.tensor([coo.row, coo.col], dtype=torch.long).to(args.device)  ## 索引张量 [2, nnz]，行索引第一行列索引第二行
        #旧写法直接调用原本数据，新写法先创建一个连续的 numpy 数组，再将数据copy过来然后转换为张量
        # v = torch.from_numpy(coo.data).float().to(args.device)               ## 旧写法
        v = torch.tensor(np.array(coo.data), dtype=torch.float32).to(args.device)  ## 值张量[nnz]
        return torch.sparse.FloatTensor(i, v, coo.shape)  ## 构造稀疏张量 (同形状)，浮点型

    # todo: 矩阵分解，加速计算                    ## 把大邻接矩阵按行均匀切成 n_fold 份
    def _split_A_hat(self, X):#将一个大的稀疏矩阵按行划分为n_fold份，作者这里定义的超参数是100
        A_fold_hat = []                          ## 存放分块稀疏张量
        fold_len = (self.n_users + self.n_items) // self.n_fold  ## 每块行数
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:#最后一块
                end = self.n_users + self.n_items
            else:#每一块终止行号（除了最后一块）
                end = (i_fold + 1) * fold_len
            A_fold_hat.append(                   ## 将分块转稀疏张量并放列表
                self._convert_sp_mat_to_sp_tensor(X[start:end]).to(args.device))
        return A_fold_hat

    # 使用图卷积神经网络得到的user embedding      ## 主体：GraphSAGE 聚合，输出所有用户嵌入
    #self.norm_adj 本质上是一个 (n_users + n_items) × (n_users + n_items) 的稀疏邻接矩阵（Heterogeneous Graph：用户–中药 二部图）。
    def _create_graphsage_user_embed(self):
        A_fold_hat = self._split_A_hat(self.norm_adj)   ## 预先切分 U-I 图
        #self.weights['user_embedding'] 是一个 [n_users, emb_dim] 的张量，保存了“每个用户（症状）节点”的初始嵌入向量。
        # self.weights['item_embedding'] 是一个 [n_items, emb_dim] 的张量，保存了“每个物品（中药）节点”的初始嵌入向量。
        #拼接后，self.weights['user_embedding'] 是一个 [n_users, emb_dim] 的张量，保存了“每个用户（症状）节点”的初始嵌入向量。
        # self.weights['item_embedding'] 是一个 [n_items, emb_dim] 的张量，保存了“每个物品（中药）节点”的初始嵌入向量。
        pre_embeddings = torch.cat(                    ## 拼接初始 U/I 嵌入
            [self.weights['user_embedding'], # 形状 = [n_users, emb_dim]
             self.weights['item_embedding']], # 形状 = [n_items, emb_dim]
            0)  #按行方向拼 ，得到 [n_users + n_items, emb_dim]

        # print("*" * 20, "embeddings", pre_embeddings)
        all_embeddings = [pre_embeddings]              ## 保存每层结果
        for k in range(self.n_layers):#逐层做图卷积
            temp_embed = []
            for f in range(self.n_fold):
                #pre_embeddings 是当前层（或初始化时）的所有节点嵌入矩阵
                temp_embed.append(                     ## 稀疏乘，聚合同层邻居信息
                    torch.sparse.mm(A_fold_hat[f], pre_embeddings))
            embeddings = torch.cat(temp_embed, 0)      ## 按行拼回整体顺序（毕竟是按行切分的），相当于分块计算
            embeddings = torch.tanh( ## 将聚合的邻居信息通过用户权重矩阵处理得到邻居特征 线性映射 Q_user → 邻居特征
                torch.matmul(embeddings, self.weights['Q_user_%d' % k]))
            embeddings = torch.cat([pre_embeddings, embeddings], 1)  ## 将原本邻接矩阵与聚合的邻居特征拼接

            pre_embeddings = torch.tanh(               ## 图卷积 W_gc_user
                torch.matmul(embeddings,
                             self.weights['W_gc_user_%d' % k]) +
                self.weights['b_gc_user_%d' % k])      # 加偏置
            pre_embeddings = nn.Dropout(self.mess_dropout[k])(pre_embeddings)  ## 消息 Dropout

            norm_embeddings = F.normalize(pre_embeddings, p=2, dim=1)  ## L2 归一化
            all_embeddings = [norm_embeddings]          ## 仅保留最新层
#把收集的多层表示在列方向拼接
        all_embeddings = torch.cat(all_embeddings, 1)    ## 最终 shape [U+I, D']
        # 但由于是使用了user的权重矩阵, 所以这里仅将user的embedding拿出来计算
        u_g_embeddings, i_g_embeddings = torch.split(    ## 分离 U / I
            all_embeddings, [self.n_users, self.n_items], 0)

        # --- 证候-证候 pair 图补充特征 ---
        #将症状-症状同构图与用户（症状）嵌入进行稀疏矩阵乘法
        temp = torch.sparse.mm(
            self._convert_sp_mat_to_sp_tensor(self.sym_pair_adj).to(args.device),
            self.weights['user_embedding'])              ## [U, D]
        # 线性投影 + 非线性激活，把同构图汇聚到的症状特征送入与主干 GCN 相同的维度空间，方便后续融合。
        #temp 是“症状-症状图”邻居聚合向量 [U, emb_dim]。
        # M_user 的形状 [emb_dim, pair_dim]（pair_dim 恰好是主干 GCN 最后一层输出维度）。
        # 矩阵乘法后得到 [U, pair_dim]——把同构图特征映射到与主干一致的维度。
        #激活函数将数值压缩到-1~1
        user_pair_embeddings = torch.tanh(
            torch.matmul(temp, self.weights['M_user']))  ## 线性映射到同维度

        if self.fusion in ['add']:                       ## 融合策略 add
            u_g_embeddings = u_g_embeddings + user_pair_embeddings
        if self.fusion in ['concat']:                    ## 融合策略 concat
            u_g_embeddings = torch.cat(
                [u_g_embeddings, user_pair_embeddings], 1)
        return u_g_embeddings                            ## 返回最终用户嵌入


    def _create_graphsage_item_embed(self):             ## 生成所有中药 (item) 嵌入
        A_fold_hat = self._split_A_hat(self.norm_adj)    ## 复用切分后的 U-I 图

        pre_embeddings = torch.cat(                      ## 拼接用户+物品初始嵌入
            [self.weights['user_embedding'], self.weights['item_embedding']], 0)

        all_embeddings = [pre_embeddings]                ## 保存每层结果
        for k in range(self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):                 ## 逐块稀疏乘，加速
                temp_embed.append(torch.sparse.mm(A_fold_hat[f], pre_embeddings))
            embeddings = torch.cat(temp_embed, 0)        ## 拼回完整顺序

            embeddings = torch.tanh(                     ## 线性映射 Q_item
                torch.matmul(embeddings, self.weights['Q_item_%d' % k]))
            embeddings = torch.cat([pre_embeddings, embeddings], 1)  ## 拼接实现 skip

            pre_embeddings = torch.tanh(                 ## 图卷积更新 W_gc_item
                torch.matmul(embeddings,
                             self.weights['W_gc_item_%d' % k]) +
                self.weights['b_gc_item_%d' % k])

            pre_embeddings = nn.Dropout(self.mess_dropout[k])(pre_embeddings)  ## Dropout

            norm_embeddings = F.normalize(pre_embeddings, p=2, dim=1)          ## L2 归一化
            all_embeddings = [norm_embeddings]            ## 只保留最后一层

        all_embeddings = torch.cat(all_embeddings, 1)     ## [U+I, D]
        u_g_embeddings, i_g_embeddings = torch.split(     ## 拆分 U / I
            all_embeddings, [self.n_users, self.n_items], 0)

        # ------- 中药-中药共现图补充特征 -------
        temp = torch.sparse.mm(
            self._convert_sp_mat_to_sp_tensor(self.herb_pair_adj).to(args.device),
            self.weights['item_embedding'])               ## [I, D]
        item_pair_embeddings = torch.tanh(
            torch.matmul(temp, self.weights['M_item']))   ## 线性映射

        if self.fusion in ['add']:                        ## 融合策略 add
            i_g_embeddings = i_g_embeddings + item_pair_embeddings
        if self.fusion in ['concat']:                     ## 融合策略 concat
            i_g_embeddings = torch.cat(
                [i_g_embeddings, item_pair_embeddings], 1)

        return i_g_embeddings                             ## 返回所有物品嵌入

    def create_batch_rating(self, pos_items, user_embeddings):
        ## 计算一批用户对给定正样本集合的匹配得分 σ(U·Iᵀ)
        pos_scores = torch.sigmoid(
            torch.matmul(user_embeddings, pos_items.transpose(0, 1)))
        return pos_scores

    def get_self_correlation(self, item_embeddings):
        """
        Args:
            item_embeddings:  [B, max_item_len, emb]
                先做两两 cos 相似 -> 去掉对角 -> 平方 -> 求和，
                最终作为对比学习的自相关约束（越小越好）
        Returns:
            cor_value: 自相关总和标量
        """
        cor_matrix = torch.cosine_similarity(             ## [B, L, L]
            item_embeddings.unsqueeze(2),                 ## e_i ⊗ 1
            item_embeddings.unsqueeze(1), dim=-1)         ## 1 ⊗ e_j
        diag = torch.diagonal(cor_matrix, dim1=1, dim2=2) ## 取对角
        cor_matrix = cor_matrix - torch.diag_embed(diag)  ## 对角置 0
        cor_matrix = cor_matrix.pow(2)                    ## 平方
        cor_value = torch.sum(cor_matrix) / 2             ## 总和/2
        return cor_value                                  ## 作为惩罚项



    def get_set_embedding(self, item_padding_set, ia_embeddings):
        """
        Args:
            item_padding_set: list: [B, max_batch_item_len]  item id set并被padding后 列表id
            ia_embeddings: [n_items, emb]
        过程：
        ia_embeddings -- > [n_item+1, emb] 最后一行是padding的嵌入表示
        Returns: set_embedding [B, emb]
        """
        padding_embedding = torch.zeros((1, ia_embeddings.size(1)), dtype=torch.float32).to(args.device)  ## 构造全零向量供 padding
        ia_padding_embedding = torch.cat((ia_embeddings, padding_embedding), 0)  # [n_item + 1, emb]  ## 在尾部追加 padding 嵌入
        item_embeddings = ia_padding_embedding[item_padding_set, :]  # [B, max_batch_item_len, emb]  ## 根据 id 列表取批量中药向量
        if item_embeddings.size(0) > 1024 or args.co_lamda == 0.0:  # valid and test  ## 过大批量或关闭相关超参时不计算自相关
            cor_value = 0
        else:
            cor_value = self.get_self_correlation(item_embeddings)   # value  ## 计算自相关惩罚
        position_ids = torch.arange(data_generator.max_item_len, dtype=torch.long, device=args.device).unsqueeze(1)  ## [L,1]
        d_model = item_embeddings.size(2)                             ## 嵌入维度
        pe = torch.zeros(data_generator.max_item_len, d_model, device=args.device)  ## 位置编码矩阵
        div_term = torch.exp(torch.arange(0.0, d_model, 2) * -(math.log(10000.0) / d_model)).to(args.device)
        pe[:, 0::2] = torch.sin(position_ids * div_term)              ## 偶数位置用 sin
        pe[:, 1::2] = torch.cos(position_ids * div_term)              ## 奇数位置用 cos
        # pe[:, :] = 1 / (position_ids.T + 1)
        pe = 1 / (position_ids + 1)                                   ## 此行覆盖上面的 sin/cos (保留原逻辑)
        position_embedding = pe.repeat(item_embeddings.shape[0], 1, 1)  # 位置编码 [B, max_batch_item_len, emb]
        attention_mask, value_attention_mask, presci_adj_matrix = self.get_attention_mask(item_embeddings)  ## 构造 mask
        if args.attention == 1:
            item_embeddings, item_attention_scores = self.attention_layer(item_embeddings,  ## 经过多头自注意力
                                                                          attention_mask=attention_mask,
                                                                          value_attention_mask=value_attention_mask,
                                                                          presci_adj_matrix=presci_adj_matrix
                                                                          )  # 经过self attention 层[B, max_batch_item_len, emb]

            neigh = torch.sum(value_attention_mask, dim=2) / value_attention_mask.size(2)  # [B, max_len]  ## 每个 token 的有效邻居占比
            neigh_num = torch.sum(neigh, dim=1)  # [B, 1]                            ## 每个处方有效 token 数
            item_set_embedding = torch.sum(item_embeddings, dim=1)  # [B, emb]       ## 所有中药向量求和
            normal_matrix_item = torch.reciprocal(neigh_num)        ## 1/len  (平均池化因子)
            normal_matrix_item = normal_matrix_item.unsqueeze(1)
            # 复制embedding_size列  [B, embedding_size]
            extend_normal_item_embeddings = normal_matrix_item.repeat(1, item_set_embedding.shape[1])
            # 对应元素相乘
            item_set_embedding = torch.mul(item_set_embedding, extend_normal_item_embeddings)  # 平均池化
            return item_set_embedding, item_attention_scores, item_embeddings, cor_value  ## 返回处方级向量及注意力/自相关

    def get_attention_mask(self, item_seq):
        """Generate attention mask for attention."""      ## 生成自注意力所需的 mask
        attention_mask = (item_seq != 0).to(dtype=torch.float32)  # [B, max_len, emb]  ## padding==0 记 0
        # attention_p = attention_mask[0][0]
        item_len = torch.sum(attention_mask, dim=1)   # [B, emb]  ## 统计每行有效 token 数
        item_len_matrix = item_len[:, :1]  # [B, 1]  得到每个药方的长度
        extended_attention_mask = torch.bmm(attention_mask, attention_mask.permute(0, 2, 1))  # [B, max_len, max_len]
        extended_attention_mask = (extended_attention_mask > 0).to(dtype=torch.float32)       ## 互相可见位置=1
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)  # fp16 compatibility\
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0       ## 不可见位置填 -1e4
        presci_adj_matrix = None                           ## 处方内部先验图（暂未用）
        # attention_mask = (1.0 - attention_mask) * - 10000.0
        return extended_attention_mask, attention_mask, presci_adj_matrix  ## 返回三种 mask

    def get_hard_sample(self, user_embeddings, ia_embeddings, model_two_stage=None):
        step = args.step
        max_step_len = args.max_step_len
        random_id = [id for id in range(0, max_step_len-step)]   ## 滑窗起点
        rating = self.create_batch_rating(ia_embeddings, user_embeddings)  ## U×I 得分
        vals, indices = rating.sort(descending=True)             ## 每行降序索引
        logits_user_neg = None
        k_id = 0
        for k in random_id:                                      ## 逐滑窗生成“伪处方”
            # topK_items = indices[:, k:]
            topK_items = indices[:, k:k+step]                     ## 连续 step 个高分中药
            if args.attention == 1:
                # padding = torch.tensor([data_generator.n_items] * (k)).to(args.device)
                padding = torch.tensor([data_generator.n_items] * (data_generator.max_item_len-k)).to(args.device)
                padding = padding.unsqueeze(1)
                padding = padding.repeat(1, topK_items.size(0)).transpose(0, 1)  ## 补齐为固定长度
                topK_items = torch.cat([topK_items, padding], dim=1).cpu().numpy().tolist()
                if model_two_stage is None:                    ## 默认使用当前模型
                    topK_sets, _, topK_att_embedding, _ = self.get_set_embedding(topK_items, ia_embeddings)  # [B, emb]
                else:                                          ## 若提供第二阶段模型
                    topK_sets, _, topK_att_embedding, _ = model_two_stage.get_set_embedding(topK_items, ia_embeddings)  # [B, emb]
            else:
                topK_items = topK_items.cpu().numpy().tolist()
                item_set_embeddings = ia_embeddings[topK_items, :]  # [B, k,emb]
                topK_sets = torch.sum(item_set_embeddings, dim=1) / step  # 平均池化
                # 可选：对 topK_sets 继续过 MLP 进行非线性变换
            if k_id == 0:
                logits_user_neg = torch.mul(user_embeddings, topK_sets)  # [B, emb]
                logits_user_neg = torch.sum(logits_user_neg, dim=1).unsqueeze(1)  # [B,1]
            else:
                neg = torch.mul(user_embeddings, topK_sets)
                neg = torch.sum(neg, dim=1).unsqueeze(1)
                logits_user_neg = torch.cat([logits_user_neg, neg], dim=1)  # [B, (#neg)]
            k_id += 1
        return logits_user_neg                                ## 返回困难负样本打分矩阵


    def create_set2set_loss(self, items, item_weights, user_embeddings, all_user_embeddins,
                            ia_embeddings, item_embeddings, use_const=0, logits_user_neg=None,
                            items_repeat=None, repeat=0, neg_item_embeddings=None, cor_value=0):
        # item_embeddings [B, emd]
        if repeat == 1:                                   ## 若是第二阶段(全正样本)则替换 items
            items = items_repeat
        predict_probs = torch.sigmoid(torch.matmul(user_embeddings, ia_embeddings.transpose(0, 1)))  ## U×I 预测概率
        mf_loss = torch.sum(torch.matmul(torch.square((items - predict_probs)), item_weights), 0)      ## 加权 MSE
        # mf_loss = nn.MSELoss(reduction='elementwise_mean')(items, predict_probs)
        mf_loss = mf_loss / self.batch_size              ## 平均到 batch
        all_item_embeddins = ia_embeddings
        regularizer = torch.norm(all_user_embeddins) ** 2 / 2 + torch.norm(all_item_embeddins) ** 2 / 2  ## L2
        regularizer = regularizer.reshape(1)
        # F.normalize(all_user_embeddins, p=2) + F.normalize(all_item_embeddins, p=2)
        regularizer = regularizer / self.batch_size

        emb_loss = self.decay * regularizer              ## 嵌入正则损失

        reg_loss = torch.tensor([0.0], dtype=torch.float64, requires_grad=True).to(args.device)
        if use_const == 0:                               ## 第一阶段：只训练重建，不用对比损失
            return mf_loss, emb_loss, reg_loss, reg_loss
        else:
            """
            user_embeddings: [B, n_e]
            item_embeddings: [B, n_e]
            """
            loss_use_cos = args.save_tail                ## 是否用余弦(保留原变量)

            # ---------- SupCon 对比学习 ----------
            t = torch.tensor(args.t).to(args.device)     ## 温度参数
            user_embeddings = F.normalize(user_embeddings, p=2, dim=1)  ## 单位化
            item_embeddings = F.normalize(item_embeddings, p=2, dim=1)
            logits_user_batch = torch.matmul(user_embeddings, item_embeddings.transpose(0, 1)) / t  # [B,B]
            if logits_user_neg is None:                  ## 如未提前构造困难负样本
                logits_user_neg = self.get_hard_sample(user_embeddings, ia_embeddings)
                logits_user = torch.cat([logits_user_batch, logits_user_neg], dim=1)  # [B,B+neg]
            elif logits_user_neg.size(0) == 0:
                logits_user = logits_user_batch
            else:
                logits_user = torch.cat([logits_user_batch, logits_user_neg], dim=1)

            # ---------- SupConLoss 计算 ----------
            labels = torch.tensor(data_generator.train_positive_list).to(args.device)
            labels = labels.contiguous().view(-1, 1)     # [B,1]
            mask = torch.eq(labels, labels.T).float().to(args.device)  # [B,B]
            if logits_user_neg.size(0) != 0:              ## 给负样本扩 0-mask
                mask = torch.cat((mask, torch.zeros_like(logits_user_neg).to(args.device)), dim=1)
            logits_max, _ = torch.max(logits_user, dim=1, keepdim=True)  # 数值稳定性
            logits = logits_user - logits_max.detach()
            exp_logits = torch.exp(logits)
            # exp_logits = exp_logits / torch.exp(logits_sim_score)    # 自适应权重 (保留注释)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)  ## 仅正例
            loss = -t * mean_log_prob_pos
            loss = loss.view(1, logits_user.shape[0]).mean()           ## 求均值
            contrastive_loss = loss
            contrastive_loss = args.contrast_reg * contrastive_loss    ## 缩放因子

            cor_loss = args.co_lamda * cor_value                       ## 自相关惩罚
            if cor_loss > 0:
                reg_loss = cor_loss   # 只是为了打印出来看下值

            contrastive_loss = contrastive_loss + cor_loss             ## 总对比损失
            return mf_loss, emb_loss, reg_loss, contrastive_loss       ## 返回四个损失项


    def normalize(self, *xs):
        return [None if x is None else F.normalize(x, dim=-1) for x in xs]# ## 便捷函数：对可变参数逐个做 L2 归一化

    def forward(self, users, item_padding_set=None, items=None, user_set=None, pos_items=None, train=True, user_padding_set=None):
        """
          *********************************************************
          Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
          Different Convolutional Layers:
              1. ngcf: defined in 'Neural Graph Collaborative Filtering', SIGIR2019;
              2. gcn:  defined in 'Semi-Supervised Classification with Graph Convolutional Networks', ICLR2018;
              3. gcmc: defined in 'Graph Convolutional Matrix Completion', KDD2018;
          """
        # todo: todo 应该在主函数中
        if self.alg_type in ['SMGCN', 'CLEPR']:
            if train:
                ua_embeddings = self._create_graphsage_user_embed()  # [n_user, 最后一层size的大小]: 所有user的embedding
                ia_embeddings = self._create_graphsage_item_embed()  # [n_item, 最后一层size的大小]
                if user_set is None:
                    all_user_embeddins = None
                else:
                    all_user_embeddins = torch.index_select(ua_embeddings, 0,
                                                        user_set)  # [一个Batch中涉及到的user个数, 最后一层embedding_size]: 选择一个batch中user的embedding

                # todo:change: 构建item/user set embedding
                """
                1. 将ia_embedding与multi-hot编码items相乘得到batch中item set的表示 item_set_embeddings
                2. 将item_set_embedding平均池化
                """
                sum_embeddings = torch.matmul(users, ua_embeddings)  # [B,  最后一层embedding_size]
                normal_matrix = torch.reciprocal(torch.sum(users, 1))
                normal_matrix = normal_matrix.unsqueeze(1)  # [B, 1]
                # 复制embedding_size列  [B, embedding_size]
                extend_normal_embeddings = normal_matrix.repeat(1, sum_embeddings.shape[1])
                # 对应元素相乘
                user_embeddings = torch.mul(sum_embeddings, extend_normal_embeddings)  # 平均池化 [B, emb]
                for k in range(0, self.mlp_predict_n_layers):
                    user_embeddings = F.relu(
                        torch.matmul(user_embeddings, self.weights['W_predict_mlp_user_%d' % k])
                        + self.weights['b_predict_mlp_user_%d' % k])
                    user_embeddings = F.dropout(user_embeddings, self.mess_dropout[k])  # 证候归纳, user set的整体表示 [B, emb]

                cor_value = 0
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
                return user_embeddings, all_user_embeddins, ia_embeddings, item_embeddings, cor_value
            else:
                ua_embeddings = self._create_graphsage_user_embed()
                ia_embeddings = self._create_graphsage_item_embed()
                pos_items = torch.tensor(pos_items, dtype=torch.long).to(args.device)
                pos_i_g_embeddings = torch.index_select(ia_embeddings, 0, pos_items)   # 根据test中使用的item id选择item的embedding
                sum_embeddings = torch.matmul(users, ua_embeddings)

                normal_matrix = torch.reciprocal(torch.sum(users, 1))

                normal_matrix = normal_matrix.unsqueeze(1)

                extend_normal_embeddings = normal_matrix.repeat(1, sum_embeddings.shape[1])

                user_embeddings = torch.mul(sum_embeddings, extend_normal_embeddings)

                for k in range(0, self.mlp_predict_n_layers):
                    user_embeddings = F.relu(
                        torch.matmul(user_embeddings,
                                     self.weights['W_predict_mlp_user_%d' % k]) + self.weights[
                            'b_predict_mlp_user_%d' % k])
                    user_embeddings = F.dropout(user_embeddings, self.mess_dropout[k])
                return user_embeddings, pos_i_g_embeddings ## 返回用户嵌入 & 目标 item 嵌入












