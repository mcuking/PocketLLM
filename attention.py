import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]

attn_scores_2 = torch.empty(inputs.shape[0])
# 这里使用了一个简单的点积注意力机制
# 计算某个输入词元嵌入向量与所有输入词元嵌入向量的点积
# 用来计算注意力分数
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention Weights:", attn_weights_2)
print("Sum of Weights:", attn_weights_2.sum())