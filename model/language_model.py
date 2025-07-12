import torch.nn as nn

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
        self.tok_embed = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_embed = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_embed = nn.Dropout(cfg["drop_rate"])


    
    def forward(self, token_ids):
        return token_ids
