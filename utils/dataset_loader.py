import torch
from torch.utils.data import Dataset
import pandas as pd

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
        return (
            self.input_ids[index],
            self.target_ids[index]
        )

class SpamDataset(Dataset):
    '''
    用于加载垃圾邮件数据集的类
    text: 文本数据
    tokenizer: 分词器
    max_length: 每个序列的最大长度
    pad_token_id: 填充 token 的 id，默认使用 GPT-2 的 token id 50256 即 <|endoftext|> 来填充
    '''
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)

        # 将文本数据编码为 token id
        self.encoded_texts = [
            tokenizer.encode(text)
            for text in self.data["Text"]
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # 如果编码后的文本长度超过 max_length，则截断
            # 比如模型的最大上下文长度是 1024，那么如果文本长度超过 1024，则截断到 1024
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # 如果编码后的文本长度小于 max_length，则填充到 max_length
        # 使用 gpt2 tokenizer 的 token id 50256 即 <|endoftext|> 来填充
        # 确保每个输入张量的大小相同，以便通过 PyTorch DataLoader 加载数据集
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]

    # 返回数据集中最长的序列长度
    def _longest_encoded_length(self):
        return max(
            len(encoded_text) 
            for encoded_text in self.encoded_texts
        )

    # 返回数据集的总长度
    def __len__(self):
        return len(self.encoded_texts)
    
    # 返回数据集的指定行的数据
    def __getitem__(self, index):
        return (
            torch.tensor(self.encoded_texts[index], dtype=torch.long),
            torch.tensor(self.data.iloc[index]["Label"], dtype=torch.long)
        )
