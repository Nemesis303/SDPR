# -*- coding: utf-8 -*-
# @Time    : 2024/3/29 16:13
# @Author  : Author
# @File    : Attention_layer.py
# @Description : self-attention layer

'''SelfAttention 实现了多头自注意力机制，支持丢弃层和层归一化。
它将输入张量通过查询、键和值的变换计算注意力分数，并将其归一化，
然后通过注意力机制计算上下文向量，最后经过线性层和残差连接得到输出。'''


import torch#torch: PyTorch 库，用于张量运算。
import torch.nn as nn#torch.nn: 包含所有神经网络层（如 Linear、Dropout 和 LayerNorm）的子模块。
import math#math: 提供数学函数，在此用来缩放注意力分数。


class SelfAttention(nn.Module):#SelfAttention 类继承自 nn.Module，这是 PyTorch 中所有模型的基类。
    """
    该文档说明该类实现的是多头自注意力机制，默认情况下使用 1 个注意力头。

输入: 接受 input_tensor，这是输入到注意力层的张量。

输出: 返回 hidden_states，即经过多头自注意力层处理后的输出。

    [multi-head(default=1)] Self-attention layers, a attention score dropout layer is introduced.

    Args:
        input_tensor (torch.Tensor): the input of the self-attention layer
    Returns:
        hidden_states (torch.Tensor): the output of the multi-head self-attention layer

        
        hidden_size：隐藏状态的大小（即每个 token 的特征数）。

layer_norm_eps：LayerNorm 的小 epsilon 值（默认为 None）。

n_heads：注意力头的数量。

hidden_dropout_prob：隐藏状态的丢弃率。

attn_dropout_prob：注意力分数的丢弃率。

super() 调用父类 nn.Module 的构造函数来初始化父类。
    """

    def __init__(self, hidden_size, layer_norm_eps=None, n_heads=1, hidden_dropout_prob=0, attn_dropout_prob=0):
        super(SelfAttention, self).__init__()
        #检查 hidden_size 是否能被 n_heads 整除，因为每个注意力头需要均等地共享总的隐藏大小。
        if hidden_size % n_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention " 
                "heads (%d)" % (hidden_size, n_heads)
            )

        self.num_attention_heads = n_heads#存储注意力头的数量。
        self.attention_head_size = int(hidden_size / n_heads)#每个注意力头的大小（即 hidden_size / n_heads）。
        self.all_head_size = self.num_attention_heads * self.attention_head_size#所有头的总大小，等于 hidden_size。
        #通过计算得到QKV
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(attn_dropout_prob)#应用于注意力分数的丢弃层。
#将注意力机制输出的上下文向量通过线性层投影回原来的隐藏大小。
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_dropout = nn.Dropout(hidden_dropout_prob)#应用于线性层输出的丢弃层。
#定义LayerNorm层，LayerNorm：用于输出的层归一化。LayerNorm 能够帮助稳定训练过程，通过对每个 token 的特征进行归一化，减少梯度消失或爆炸的问题。
        if layer_norm_eps is not None:
            self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        else:
            self.LayerNorm = None

#该方法将张量的形状转换为适应多头注意力的格式，即将隐藏维度分割成多个注意力头。
    def transpose_for_scores(self, x):
  
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)#改变张量的形状。
        return x.permute(0, 2, 1, 3)#重新排列维度，以便在计算注意力时能够正确地分配给每个注意力头。

    def forward(self, input_tensor, attention_mask=None, value_attention_mask=None, presci_adj_matrix=None):
        """
        Args:
        input_tensor: 输入到注意力层的张量，形状为 [B, max_set_len, hidden_size]。

attention_mask: 可选的掩码，用于在计算注意力时忽略某些 token。

value_attention_mask: 应用到值的掩码。

presci_adj_matrix: 可选的邻接矩阵，通常用于图神经网络模型中。
            input_tensor:  [B, max_set_len, hidden_size]
            attention_mask:  [B, max_set_len, max_set_len]

        Returns: hidden_states [B, max_set_len, hidden_size]

        """
        #通过 query、key 和 value 线性层将输入张量转换为查询、键和值。    
        query_layer = self.query(input_tensor)
        key_layer = self.key(input_tensor)
        value_layer = self.value(input_tensor)
   #计算查询和键之间的点积，得到原始的注意力分数。    
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # [B, max_len, max_len]
#对注意力分数进行缩放，以防止在训练过程中出现梯度消失或爆炸问题。
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#如果提供了 attention_mask，将其添加到注意力分数中，以忽略特定位置的计算。        
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

#如果提供了 presci_adj_matrix，将其加到注意力分数中（通常用于图神经网络中的邻接矩阵）。
        # Normalize the attention scores to probabilities.
        if presci_adj_matrix is not None:
            attention_scores = attention_scores + presci_adj_matrix
            #使用 Softmax 函数将注意力分数转化为概率，表示每个 token 对其他 token 的注意力权重。
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.

#对注意力概率应用丢弃层，随机丢弃某些 token 的注意力。
        attention_probs = self.attn_dropout(attention_probs)
    #通过加权求和（即用注意力概率加权值）得到上下文向量。    
        context_layer = torch.matmul(attention_probs, value_layer)
    #将上下文向量通过一个线性层，映射回原来的隐藏状态大小。    
        hidden_states = self.dense(context_layer)
        #对线性层的输出应用丢弃层。
        hidden_states = self.out_dropout(hidden_states)
        #应用 value_attention_mask，用来在计算过程中屏蔽掉某些值。
        hidden_states = hidden_states * value_attention_mask
        #如果使用了 LayerNorm，则对输出和输入进行加和后再做归一化（实现了残差连接）。
        if self.LayerNorm is not None:
            hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states, attention_probs

#用于从输出张量中收集特定位置的向量。通常用于提取特定位置（例如序列中的最后一个元素）的嵌入。
    def gather_indexes(self, output, gather_index):
        """
        Gathers the vectors at the specific positions over a minibatch
        output: [B max_set_len H]
        """
        # 在expand中的-1表示取当前所在维度的尺寸，也就是表示当前维度不变
        gather_index = gather_index.view(-1, 1, 1).expand(-1, -1, output.shape[-1])   # [B 1 H]
        # print("\n gather", gather_index[0])
        # todo:notice 取出每个item_seq的最后一个item的H维向量
        output_tensor = output.gather(dim=1, index=gather_index)        # [B item_num H]
        #squeeze(1) 去掉了 output_tensor 中第 1 维的大小为 1 的维度。

#最终返回的张量形状为 [B, item_num * H]，即每个样本中收集的向量拼接成一个张量。
        return output_tensor.squeeze(1)










