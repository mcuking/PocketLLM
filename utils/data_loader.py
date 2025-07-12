import tiktoken
import torch
from torch.utils.data import Dataset, DataLoader

class LLMDataset(Dataset):
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
    dataset = LLMDataset(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle, # 是否打乱数据顺序 防止模型过拟合
        drop_last=drop_last, # 是否丢弃最后一个不完整的batch
        num_workers=num_workers # 多线程加载数据的工作线程数
    )
    return dataloader
