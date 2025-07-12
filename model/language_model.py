import torch
import torch.nn as nn
from .transform_block import TransformBlock
from .layer_norm import LayerNorm

# 神经网络
# 通过线性函数y=wx+b和非线性激活函数y=g(x)的不断嵌套组合，可以模拟出任意复杂的函数
# 例如 y = g(w3 * g(w2 * g(w1 * x + b1) + b2) + b3)
# 但是函数表述过于复杂，因此使用神经网络这种形式来表述
# 后续的所有操作都是在已知输入x和输出y的情况下，求解参数w和b的值
# 就像三角形面积公式 s = (1/2) * base * height
# 如果不知道参数值 1/2，即 s = w * base * height，则通过已知的几组s/base/height的值，来求解w的值

# 前向传播
# 就是将输入x通过神经网络，得到输出y的过程

# 反向传播
# 计算损失函数关于每个参数的梯度，每个参数都向着梯度下降的方向变化一点点，来实现损失值最小的过程，这就构成了神经网络的训练过程。下面是详细讲解：
# 损失函数可以是交叉熵函数，也可以是均方误差函数，这里先选择均方误差函数来讲解，
# L(w, b) = 1/n * sum((y - y_hat)^2) 其中y为真实值，y_hat为预测值，即 y_hat = x * w + b
# 在大模型中主要计算某个词的出现概率，所以 y 可以视为一个 one-hot 向量，即 y = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]，数组长度为词汇表大小
# 即某个词的出现概率为 1，其他词的出现概率为 0，而 y_hat 就是模型预测的向量 y_hat = [0.2, 0.6, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1]
# 这里损失函数的值就是表示预测词概率和真实词概率的差异。

# 反向传播目标：通过调整参数 w 和 b 的值，使得损失值最小，即求解让 L 最小的 w 和 b 的值。

# 首先先看单个参数 w，通过损失函数 L 对参数 w 的偏导数，即 dL/dw，因为偏导数是函数值上升最快的方向，
# 所以为了让函数值变小，让 w 向着偏导数的反方向移动，即 w = w - dL/dw，
# 具体变化快慢可以增加一个系数，即 w = w - lr * dL/dw，其中 lr 是学习率，表示每次移动的步长。
# 同理参数 b 也是同样的计算方式，即 b = b - lr * dL/db。
# 梯度是一个向量，包含了每个参数的偏导数，即 [dL/dw1, dL/db1, dL/dw2, dL/db2, ...]，
# 而不断调整参数 w 和 b 的值，让损失函数值减小的过程，进而求出最终的 w 和 b 值的过程，就是梯度下降。
# 可以理解成下山的过程，每次都朝着梯度下降最快的方向移动，最终到达山底，即损失值最小的地方。
# 但是梯度下降存在一个问题，就是每次移动的步长是固定的，如果步长太大，可能会错过最低点，如果步长太小，又需要很多步才能到达最低点。
# 因此需要动态调整步长，即学习率，让步长随着梯度变化，如果梯度很大，步长就大，如果梯度很小，步长就小。
# 这样就可以在梯度下降的过程中，快速到达最低点，这就是梯度下降的原理。

# 接下来问题是如何求得偏导数 dL/dw 和 dL/db。由最上面的介绍可以知道神经网络是多层函数嵌套的，
# 这里就会用到微积分的复合函数求导法则，即链式法则。下面举例：
# x --g(w1x + b1)--> a --g(w2a + b2)--> y_hat --1/n*sum((y-y_hat)^2)--> L
# L对w1的偏导数 dL/dw1 = dL/dy_hat * dy_hat/da * da/dw1，即每个阶段的偏导数相乘。
# 这个过程就是从后往前，一层一层求偏导数，然后逐步更新每一层的参数，
# 直到把所有的神经网络参数都更新一遍，即完成一次训练，也可形象地称之为反向传播。

# 梯度消失
# 就是在反向传播的过程中，梯度值不断变小，因为是通过将参数减去梯度下降的步长不断更新参数，
# 所以如果梯度值很小，步长也会很小，最终导致参数无法更新，模型无法收敛。

# 梯度爆炸
# 就是在反向传播的过程中，梯度值不断变大，因为是通过将参数减去梯度下降的步长不断更新参数，
# 所以如果梯度值很大，步长也会很大，参数的调整幅度很大，导致模型无法收敛。

class LanguageModel(nn.Module):
    def __init__(self, cfg):
        """
        初始化语言模型

        Args:
            cfg (dict): 配置字典，包含以下字段：
                "vocab_size": 词汇表大小，被 BPE 分词器处理后的词汇数量
                "context_length": 上下文长度，模型通过位置嵌入能够处理的最大输入词元数量
                "emb_dim": 词元嵌入的维度，可将每个词元转化为多少维的向量，输出的上下文向量维度也相同
                "num_heads": 多头注意力中的注意力头的数量
                "num_layers": 模型中的 Transformer 块的数量
                "drop_rate": 用于 drop_rate 的丢弃率，防止过拟合
                "qkv_bias": 是否在 MultiHeadAttention 中使用可学习的偏置项
        """
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        # 根据配置字典中的参数，创建多个 Transformer 块，每个块都包含多头注意力和前馈神经网络，并带有 dropout 和层归一化功能
        # 在 1.24亿参数的 GPT-2 模型中，Transformer 块被重复使用 12 次
        self.trf_blocks = nn.Sequential(*[TransformBlock(cfg) for _ in range(cfg["num_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 将 Transform 块的输出投射到分词器的词汇空间，为每个词元生成一个分数 logits，表示该词元在词汇表中的概率分布
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    def forward(self, token_ids):
        batch_size, num_tokens = token_ids.shape
        # 将输入 token_ids 转换为词元嵌入
        # 例如，如果 token_ids 是一个 3x5 的张量，其中 3 是批处理大小，5 是上下文长度，
        # 那么 tok_embeds 的形状为 (3, 5, 768)，768 是词元嵌入的维度
        tok_embeddings = self.tok_emb(token_ids)

        # 默认采用绝对位置嵌入，因此只需创建一个维度与 tok_embeds 相同的张量即可
        # pos_embeddings 输入通常是是占位符向量 torch.arange(num_tokens)，
        # torch.arange(num_tokens) 会返回一个形状为 (num_tokens,) 的张量，其中包含从 0 到 num_tokens-1 的值
        # 例如，如果上下文长度为 5，那么 pos_embeddings 的形状为 (5, 768)，其中 5 是上下文长度，768 是词元嵌入的维度
        pos_embeddings = self.pos_emb(torch.arange(num_tokens, device=token_ids.device))

        # 将词元嵌入和位置嵌入相加
        # 例如，pytorch 会为每个批次中的每个 5x768 的词元嵌入张量添加一个 5x768 的位置嵌入张量
        x = tok_embeddings + pos_embeddings

        # 对词元嵌入和位置嵌入进行 dropout 处理，防止过拟合
        x = self.drop_emb(x)

        # 多层的 Transformer 块处理
        x = self.trf_blocks(x)

        # 对输出进行层归一化处理，然后传递到线性输出层
        x = self.final_norm(x)

        # 将 Transform 块的输出投射到分词器的词汇空间，为每个词元生成一个分数 logits，表示该词元在词汇表中的概率分布
        # 通过线性输出层，将上下文向量转化为词汇表大小的向量，即每个词元在词汇表中的概率分布
        # 例如，如果词汇表大小为 10000，那么输出向量的维度就是 10000
        logits = self.out_head(x)

        return logits
