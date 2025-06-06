import torch
import torch.nn as nn
from functorch.einops import rearrange
from numpy.core.fromnumeric import repeat


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
        project_out = not (heads == 1 and dim_heads == dim)

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


class FeedForward(nn.Module):
    """
    前馈神经网络模块

    该模块实现了Transformer架构中的前馈神经网络，允许模型学习非线性关系。
    通过将输入映射到隐藏层，然后通过激活函数和线性映射到输出，实现了非线性映射。

    参数:
        dim (int): 输入特征的维度
        hidden_dim (int, 可选): 隐藏层的维度，默认为dim*4
        dropout (float, 可选): Dropout概率，用于防止过拟合，默认为0.1
    """

    def __init__(self, dim, hidden_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        前向传播函数

        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, sequence_length, dim]

        返回:
            torch.Tensor: 前馈神经网络的输出，形状与输入相同
        """
        return self.net(x)


class Transformer(nn.Module):
    """
    Transformer模块

    该模块实现了Transformer架构的基本组成部分，包括多头自注意力机制和前馈神经网络。
    通过堆叠多个这样的模块，可以构建更深的Transformer模型。

    参数:
        dim (int): 输入特征的维度
        depth (int): Transformer模块的层数
        heads (int, 可选): 注意力头的数量，默认为8
        dim_heads (int, 可选): 每个注意力头的维度，默认为64
        hidden_dim (int, 可选): 前馈网络的隐藏层维度，默认为dim*4
        dropout (float, 可选): Dropout概率，用于防止过拟合，默认为0.1
    """

    def __init__(self, dim, depth, heads=8, dim_heads=64, hidden_dim=None, dropout=0.1):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PerNorm(dim, Attention(dim, heads=heads, dim_heads=dim_heads, dropout=dropout)),
                PerNorm(dim, FeedForward(dim, hidden_dim or dim * 4, dropout=dropout))
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        """
        前向传播函数

        参数:
            x (torch.Tensor): 输入张量，形状为 [batch_size, sequence_length, dim]

        返回:
            torch.Tensor: Transformer模块的输出，形状与输入相同
        """
        for attn, ff in self.layers:
            x = attn(x) + x  # 残差连接
            x = ff(x) + x  # 残差连接
        return x


class PerNorm(nn.Module):
    """
    PerNorm模块

    该模块实现了PerNorm（Per-token Normalization）机制，对每个时间步进行归一化处理。
    通过将输入张量进行逐元素归一化，实现了对每个时间步的归一化处理。

    参数:
        dim (int): 输入特征的维度
        eps (float, 可选): 缩放系数，默认为1e-5
    """

    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3,
                 dim_head=64, dropout=0., emb_dropout=0., ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)
