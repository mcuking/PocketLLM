import torch
from torch import nn

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# 当我们定义一个PyTorch模型时，通常会继承`nn.Module`类
class CausalfAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        # 生成三个权重矩阵：
        # 查询矩阵Wq：表示“我需要什么信息“
        # 键矩阵Wk：表示“我能提供什么信息“
        # 值矩阵Wv：表示“我的实际价值内容“
        # 假设我们有一个句子：“The cat sat on the mat”
        # 查询空间：让“sat”的查询向量能够询问“谁在坐？”（关注主语）和“坐在哪里？”（关注地点）。
        # 键空间：让“cat”的键向量能够表示“我是主语”，而“mat”的键向量表示“我是地点”。
        # 值空间：当“cat”被关注时，它的值向量提供关于主语的详细信息（如“cat”是动物，单数等）；当“mat”被关注时，它的值向量提供关于地点的信息（如“mat”是物体，在下面等）。
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

        # 掩码（mask）用于屏蔽某些位置的注意力分数
        # 例如，对于一个3x3的矩阵，`torch.triu(torch.ones(3,3), diagonal=1)`会得到：
        # [[0, 1, 1],
        #  [0, 0, 1],
        #  [0, 0, 0]]
        # 其中 `diagonal=1`表示从主对角线向上偏移1的位置开始（即不包括主对角线）。
        # 所以主对角线上方的元素（包括对角线上面的一条）会被保留
        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        # 在这里，我们使用 `register_buffer` 将掩码注册为模型的一个缓冲区。
        # 缓冲区会和模型一起自动移动到适当的设备（如GPU或CPU），无需手动确保这些张量和模型参数在同一设备上，从而避免设备不匹配的错误
        self.register_buffer("mask", mask)

        # dropout 是一种深度学习中的正则化技术，用于防止过拟合。
        # 在训练过程中，dropout 会随机丢弃一部分神经元的输出，
        # 以减少模型对特定神经元的依赖，从而提高模型的泛化能力。
        # 需要注意：仅在训练过程中使用 dropout，在推理（测试）阶段不使用。
        self.dropout = nn.Dropout(dropout)

    # 我们在子类中定义`forward`方法，该方法描述了前向传播的计算过程。
    def forward(self, x):
        # 计算查询、键和值向量
        # 通过输入 x 和权重矩阵进行矩阵乘法
        # 这里的 x 是输入的特征矩阵
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)

        # 计算注意力分数
        # 通过计算每个查询向量和每个键向量的点积来计算注意力分数
        # 注意：因为增加了batch维度后，不能直接用 T 进行转置，
        # 因为 T 只会转置最后两个维度。
        # 这里我们使用 `transpose(1, 2)` 来转置第二和第三个维度。
        attn_scores = queries @ keys.transpose(1, 2)

        # 因果注意力
        # 添加掩码（mask）来屏蔽某些位置的注意力分数
        # 例如，在自回归模型中，我们可以使用掩码来确保模型只能关注之前的输入，而不能关注未来的输入。
        # 这可以通过将注意力分数中某些位置设置为负无穷大（-inf）来实现，这样在 softmax 计算时，这些位置的权重将变为零。
        # `attn_scores.masked_fill_(mask, value)`: 这是一个原地操作（带下划线），直接作用于原数据，减少不必要的内存拷贝。
        # 用`value`填充`attn_scores`中所有`mask`为True的位置。
        # 这里，我们将这些位置填充为`-torch.inf`，即负无穷。
        mask_bool = self.mask.bool()
        attn_scores.masked_fill_(mask_bool, -torch.inf)
        print("Attn Scores:\n", attn_scores)

        # 计算注意力权重
        # 使用 softmax 函数将注意力分数转换为权重
        # 注意力分数除以键向量的维度的平方根，以防止梯度消失
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        
        # 应用 dropout
        # 在训练过程中，为了防止过拟合，我们可以在注意力权重上应用 dropout，丢掉一部分注意力权重。
        # 过拟合指的是模型在训练数据上表现很好，但在新数据上表现不佳。
        # dropout 是一种正则化技术，可以在训练过程中随机丢弃一部分神经元的输出，
        # 以减少模型对特定神经元的依赖，从而提高模型的泛化能力。
        # dropout 的概率是一个超参数，通常在 0.1 到 0.5 之间。
        # 需要注意：仅在训练过程中使用 dropout，在推理（测试）阶段不使用。
        attn_weights = self.dropout(attn_weights)
        print("Attn Weights:\n", attn_weights)

        # 计算上下文向量
        # 上下文向量是注意力权重和值向量的加权和
        context_vec = attn_weights @ values
        return context_vec

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 叠加多个单头注意力层
        self.heads = nn.ModuleList([
            CausalfAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        # 对每个头进行前向传播，并将结果拼接在一起
        # 在最后一个维度上拼接所有头的输出
        return torch.cat([head(x) for head in self.heads], dim=-1)
    
# 将输入数据扩展为批处理（batch）的形式
# 在实际应用中，我们通常会处理多个输入样本。
# 例如，在自然语言处理任务中，一个批处理可能包含多个句子，每个句子都是一个独立的输入样本。
batch = torch.stack((inputs, inputs), dim=0)
d_in = batch.shape[2] # 单个词元向量的维度，batch 的最第三个维度
d_out = 2
context_length = batch.shape[1] # 词元的数量

# 设置随机种子以保证结果可复现
torch.manual_seed(123)
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
# 在`nn.Module`中，有一个`__call__`方法被重写。
# 当我们像函数一样调用模块实例（例如`sa_v1(inputs)`）时，
# 实际上调用的是`__call__`方法， `__call__`方法内部会调用`forward`方法，
# 同时还会处理一些其他事情（例如钩子、梯度记录等）。
# 神经网络层本质是数学函数：y = f(x)
# f(x) 比 f.forward(x) 更符合数学直觉
context_vecs = mha(batch)
print("Context Vecs:\n", context_vecs)