import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    '''
    用于加载文本数据集的类
    text: 文本数据
    tokenizer: 分词器
    max_length: 每个序列的最大长度
    stride: 滑动窗口的步长 
    '''
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 将全部文本进行分词
        token_ids = tokenizer.encode(text)
        
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
    def __getitem__(self, index):
        return (self.input_ids[index], self.target_ids[index])
