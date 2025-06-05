import torch
import torch.nn as nn
from functorch.einops import rearrange


class Attention(nn.Module):
    """
    多头注意力机制模块
    
    该模块实现了Transformer架构中的自注意力机制，允许模型关注输入序列的不同部分。
    通过将输入投影到查询(query)、键(key)和值(value)，然后计算注意力权重和加权和来实现。
    
    参数:
        dim (int): 输入特征的维度
        heads (int, 可选): 注意力头的数量，默认为8
        dim_heads (int, 可选): 每个注意力头的维度，默认为64
        dropout (float, 可选): Dropout概率，用于防止过拟合，默认为0.1
    """
    def __init__(self, dim, heads=8, dim_heads=64, dropout=0.1):
        super(Attention, self).__init__()
        inner_dim = dim_heads * heads
        project_out = not (heads==1 and dim_heads==dim)

        self.heads = heads
        self.scale = dim_heads ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        # 注意：这里的to_qkv是一个线性层，将输入x投影到更高维度空间
        # inner_dim * 3表示同时为query、key、value三个矩阵创建投影
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, sequence_length, dim]
            
        返回:
            torch.Tensor: 注意力机制的输出，形状与输入相同
        """
        # 这行代码执行了两个步骤:
        # 1. self.to_qkv(x): 将输入x通过线性变换层，输出形状为[batch_size, sequence_length, inner_dim*3]
        # 2. .chunk(3, dim=-1): 沿着最后一个维度将结果分成3个相等大小的块
        # 结果qkv是一个包含3个张量的元组，分别对应query、key、value
        # 每个张量的形状为[batch_size, sequence_length, inner_dim]
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        # 计算注意力分数 (Q * K^T / sqrt(dim_head))
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        # 应用softmax获取注意力权重
        attn = self.attend(dots)
        attn = self.dropout(attn)
        
        # 注意力权重与值的矩阵乘法
        output = torch.matmul(attn, v)

        # 重新排列多头输出为原始形状
        output = rearrange(output, 'b h n d -> b n (h d)')
        
        return output
