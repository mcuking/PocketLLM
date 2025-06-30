import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

d_in = inputs.shape[1]
d_out = 2

torch.manual_seed(123)  # 设置随机种子以保证结果可复现
# 生成三个权重矩阵：
# 查询矩阵Wq：表示“我需要什么信息“
# 键矩阵Wk：表示“我能提供什么信息“
# 值矩阵Wv：表示“我的实际价值内容“
# 假设我们有一个句子：“The cat sat on the mat”
# 查询空间：让“sat”的查询向量能够询问“谁在坐？”（关注主语）和“坐在哪里？”（关注地点）。
# 键空间：让“cat”的键向量能够表示“我是主语”，而“mat”的键向量表示“我是地点”。
# 值空间：当“cat”被关注时，它的值向量提供关于主语的详细信息（如“cat”是动物，单数等）；当“mat”被关注时，它的值向量提供关于地点的信息（如“mat”是物体，在下面等）。
W_query = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
W_key = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.randn(d_in, d_out), requires_grad=False)
print("W_query:", W_query)

# 计算查询、键和值向量
queries = inputs @ W_query
keys = inputs @ W_key
values = inputs @ W_value
print("queries:", queries)
print("keys:", keys)
print("values:", values)

# 计算注意力分数
# 通过计算每个查询向量和每个键向量的点积来计算注意力分数
attention_scores = queries @ keys.T
print("attention_scores:", attention_scores)

# 计算注意力权重
# 使用 softmax 函数将注意力分数转换为权重
# 注意力分数除以键向量的维度的平方根，以防止梯度消失
# 这里的 d_k 是键向量的维度
d_k = keys.shape[-1]  # 键向量的维度
attention_weights = torch.softmax(attention_scores / d_k**0.5, dim=-1)
print("attention_weights:", attention_weights)

# 计算上下文向量
# 上下文向量是注意力权重和值向量的加权和
context_vector = attention_weights @ values
print("context_vector:", context_vector)