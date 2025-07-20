import torch.nn as nn
from .attention import MultiHeadAttention
from .feed_forward import FeedForward
from .layer_norm import LayerNorm

class TransformBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in = cfg["emb_dim"],
            d_out = cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["num_heads"],
            drop_rate=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )

        self.ffn = FeedForward(cfg["emb_dim"])
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        # 层归一化，使得每一层的输出被调整到合适的范围（零均值和单位方差，即均值为 0 方差和为 1）
        x = self.norm1(x)
        # 多头注意力机制，计算词元的上下文向量组合，是数据包含了上下文信息
        x = self.attn(x)
        # dropout层，避免过拟合
        x = self.drop_shortcut(x)
        # 梯度消失
        # 就是在反向传播的过程中，梯度值不断变小，因为是通过将参数减去梯度下降的步长不断更新参数，
        # 所以如果梯度值很小，步长也会很小，最终导致参数无法更新，模型无法收敛。
        # 残差连接（也叫快捷连接）
        # 通过将一层的输出添加到后续层的输出。
        # 残差连接为什么会避免梯度消失？
        # 残差连接相当于在网络中添加了一条捷径，使得梯度可以直接从输出层传递到输入层，从而避免梯度消失问题。
        # 残差连接为什么不会导致梯度爆炸？
        # 残差连接不会导致梯度爆炸，因为残差连接只是将输出层和输入层相加，而不是相乘，所以不会导致梯度爆炸。
        x = x + shortcut

        shortcut = x
        # 层归一化，使得每一层的输出被调整到合适的范围（零均值和单位方差，即均值为 0 方差和为 1）
        x = self.norm2(x)
        # 前馈神经网络层，通过维度扩展-激活-压缩结构增强表达能力
        x = self.ffn(x)
        # dropout层，避免过拟合
        x = self.drop_shortcut(x)
        # 残差连接
        x = x + shortcut
        return x
