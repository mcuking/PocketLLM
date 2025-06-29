import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 将全部文本进行分词
        token_ids = tokenizer.encode(txt)
        
        # 使用滑动窗口方法将文本划分为长度为max_length的重叠序列
        # stride为步长，决定了每个序列之间的重叠部分
        # 例如，max_length=10, stride=5时，序列为[0:10], [5:15], [10:20]等
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunks = token_ids[i:i + max_length]
            target_chunks = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunks))
            self.target_ids.append(torch.tensor(target_chunks))

    # 返回数据集的总长度
    def __len__(self):
        return len(self.input_ids)
    
    # 返回数据集的指定行的数据
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    

def create_dataloader(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 初始化分词器
    tokenizer = tiktoken.get_encoding("gpt2")
    # 创建数据集
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle, # 是否打乱数据顺序 防止模型过拟合
        drop_last=drop_last, # 是否丢弃最后一个不完整的batch
        num_workers=num_workers # 多线程加载数据的工作线程数
    )
    return dataloader

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

# 设置最大序列长度
max_length = 4 
# 创建数据加载器
dataloader = create_dataloader(
    raw_text,
    batch_size=8,
    max_length=max_length,
    stride=4,
    shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("Inputs shape:\n", inputs.shape)


# GPT-2的词汇表大小
vocab_size = 50257
# 将词元编码为256维向量
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim) # 词元嵌入层
# 获取输入的嵌入向量
token_embeddings = token_embedding_layer(inputs)
print("Inputs embeddings:\n", token_embeddings)
print("Inputs embeddings shape:\n", token_embeddings.shape)


# 获取GPT模型所采用的绝对位置嵌入
# 只需创建一个维度与token_embedded相同的嵌入层即可
context_length = max_length
# 位置嵌入层
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
# 获取位置的嵌入向量，其中torch.arange(max_length)生成0到max_length-1的整数
pos_embeddings = pos_embedding_layer(torch.arange(max_length))
print("Positional embeddings:\n", pos_embeddings)
print("Positional embeddings shape:\n", pos_embeddings.shape)

# 将位置嵌入与token嵌入相加
# 由于token_embedding和pos_embedding的形状相同，可以直接相加
input_embedding = token_embeddings + pos_embeddings
print("Embedded input:\n", input_embedding)
print("Embedded input shape:\n", input_embedding.shape)

