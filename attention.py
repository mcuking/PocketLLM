import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

# 当我们定义一个PyTorch模型时，通常会继承`torch.nn.Module`类
class SelfAttention_v1(torch.nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        # 生成三个权重矩阵：
        # 查询矩阵Wq：表示“我需要什么信息“
        # 键矩阵Wk：表示“我能提供什么信息“
        # 值矩阵Wv：表示“我的实际价值内容“
        # 假设我们有一个句子：“The cat sat on the mat”
        # 查询空间：让“sat”的查询向量能够询问“谁在坐？”（关注主语）和“坐在哪里？”（关注地点）。
        # 键空间：让“cat”的键向量能够表示“我是主语”，而“mat”的键向量表示“我是地点”。
        # 值空间：当“cat”被关注时，它的值向量提供关于主语的详细信息（如“cat”是动物，单数等）；当“mat”被关注时，它的值向量提供关于地点的信息（如“mat”是物体，在下面等）。
        self.W_query = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = torch.nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = torch.nn.Linear(d_in, d_out, bias=qkv_bias)

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
        attention_scores = queries @ keys.T

        # 计算注意力权重
        # 使用 softmax 函数将注意力分数转换为权重
        # 注意力分数除以键向量的维度的平方根，以防止梯度消失
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)

        # 计算上下文向量
        # 上下文向量是注意力权重和值向量的加权和
        context_vector = attention_weights @ values
        return context_vector
    

d_in = inputs.shape[1]
d_out = 2

# 设置随机种子以保证结果可复现
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
# 在`nn.Module`中，有一个`__call__`方法被重写。
# 当我们像函数一样调用模块实例（例如`sa_v1(inputs)`）时，
# 实际上调用的是`__call__`方法， `__call__`方法内部会调用`forward`方法，
# 同时还会处理一些其他事情（例如钩子、梯度记录等）。
# 神经网络层本质是数学函数：y = f(x)
# f(x) 比 f.forward(x) 更符合数学直觉
context_vector = sa_v1(inputs)
print("Context Vector:\n", context_vector)
