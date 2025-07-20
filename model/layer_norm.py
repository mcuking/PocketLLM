import torch
import torch.nn as nn

# 层归一化作用：通过对每一层的输入进行归一化（减去均值，除以标准差）。
# 使得每一层的输出被调整到合适的范围（零均值和单位方差，即均值为 0 方差和为 1），加速训练收敛，减少训练步数；
# 另外有助于梯度的稳定传播，避免梯度消失或爆炸。
# 例如向量 [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000] 在归一化后
# 转化为 [-0.0207,  0.1228, -1.1915,  1.6621,  0.6186, -1.1915]，均值为 0，方差和为 1。
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        """
        初始化层归一化层

        Args:
            emb_dim (int): 输入张量的大小
        """
        super().__init__()
        # 用于数值稳定的小值，防止计算归一化时分母为 0
        self.eps = 1e-12
        # nn.Parameter 将张量标记为模型参数，使其在训练过程中能被优化器自动更新，缩放/偏移因子是可学习的
        self.weight = nn.Parameter(torch.ones(emb_dim))
        self.bias = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # 层归一化的实现通常包括三个步骤：计算均值和方差、归一化、缩放和移位。
       
        # 1.1 计算均值
        # dim=-1 表示沿着最后一个维度计算均值
        # keepdim=True 表示保留原始张量的维度，否则会降维
        mean = x.mean(dim=-1, keepdim=True)

        # 1.2 计算方差
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # 2 归一化
        # 对于单个样本的特征向量 x，归一化后的特征为：x' = (x - μ) / √(σ² + ε)，
        # 其中 μ 和 σ 分别为均值和标准差(即方差的平方根)，ε 是很小的值用来防止分母为0。
        # 然后通常还会进行缩放和平移：y = γ * x' + β（其中γ和β是可学习的缩放/偏移参数）。
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # 3 缩放和移位
        return self.weight * norm_x + self.bias
