import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    '''
    用于加载文本数据集的类
    data: 字符串格式的文本数据
    tokenizer: 分词器
    max_length: 每个序列的最大长度
    stride: 滑动窗口的步长 
    '''
    def __init__(self, data, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # 将全部文本进行分词
        token_ids = tokenizer.encode(data)
        
        # 使用滑动窗口方法将文本划分为长度为max_length的重叠序列
        # stride为步长，决定了每个序列之间的重叠部分
        # 例如，max_length=10, stride=5时，序列为[0:10], [5:15], [10:20]等
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunks = token_ids[i:i + max_length]
            target_chunks = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunks))
            self.target_ids.append(torch.tensor(target_chunks))

    # 返回数据集的指定行的数据
    def __getitem__(self, index):
        return (
            self.input_ids[index],
            self.target_ids[index]
        )

    # 返回数据集的总长度
    def __len__(self):
        return len(self.input_ids)

class SpamDataset(Dataset):
    '''
    用于加载分类微调数据集的类
    data: pandas DataFrame 格式的数据集
    tokenizer: 分词器
    max_length: 每个序列的最大长度
    pad_token_id: 填充 token 的 id, 默认使用 GPT-2 的 token id 50256 即 <|endoftext|> 来填充
    '''
    def __init__(self, data, tokenizer, max_length=None, pad_token_id=50256):
        self.data = data

        # 将文本数据编码为 token id
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]

        if max_length is None:
            # 获取数据集中最长的序列长度
            self.max_length = max(len(encoded_text) for encoded_text in self.encoded_texts)
        else:
            self.max_length = max_length
            # 如果编码后的文本长度超过 max_length，则截断
            # 比如模型的最大上下文长度是 1024，那么如果文本长度超过 1024，则截断到 1024
            self.encoded_texts = [encoded_text[:self.max_length] for encoded_text in self.encoded_texts]

        # 如果编码后的文本长度小于 max_length，则填充到 max_length
        # 使用 gpt2 tokenizer 的 token id 50256 即 <|endoftext|> 来填充
        # 确保每个输入张量的大小相同，以便通过 PyTorch DataLoader 加载数据集
        self.encoded_texts = [encoded_text + [pad_token_id] * (self.max_length - len(encoded_text)) for encoded_text in self.encoded_texts]

    # 返回数据集的指定行的数据
    def __getitem__(self, index):
        return (
            torch.tensor(self.encoded_texts[index], dtype=torch.long),
            torch.tensor(self.data.iloc[index]["Label"], dtype=torch.long)
        )

    # 返回数据集的总长度
    def __len__(self):
        return len(self.encoded_texts)

class InstructionDataset(Dataset):
    '''
    用于加载指令微调数据集的类

    Args:
        data: 字符串格式的文本数据
        tokenizer: 分词器
    '''
    def __init__(self, data, tokenizer):
        self.data = data

        self.encoded_texts = [tokenizer.encode(self.instruction_to_text(entry)) for entry in self.data]

    # 返回数据集的指定行的数据
    def __getitem__(self, index):
        return self.encoded_texts[index]

    # 返回数据集的总长度
    def __len__(self):
        return len(self.encoded_texts)
    
    # 将指令数据转换为文本格式
    def instruction_to_text(self, entry):
        instruction_text = (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}"
        )
        input_text = f"\n\n### Input:\n{entry['input']}" if entry['input'] else ""
        response_text = f"\n\n### Response:\n{entry['output']}"
        return instruction_text + input_text + response_text

def custom_collate_fn(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device="cpu"):
    '''
    指令微调的自定义批处理函数，用于自定义批次数据的整理方式
    有自定义 collate_fn 的情况下，随机/不随机（shuffle）选择 batch 个索引传入 dataset 里的 __getitem__(self, index) 得到对应的数据，
    将这些数据（样本对）传入 collate_fn 指定函数进行处理
    
    Args:
        batch: 当前批次的数据
        pad_token_id: 填充 token 的 id, 默认使用 GPT-2 的 token id 50256 即 <|endoftext|> 来填充
        ignore_index: 忽略的 id, 在计算损失值时将不计算包含这些 id 的 token
        allowed_max_length: 允许的最大序列长度，如超过则截断
        device: 决定模型在 CPU 还是 GPU 上运行
    '''

    # 获取批次中最长的序列长度
    batch_max_length = max(len(item) for item in batch)

    inputs_list, targets_list = [], []
    for item in batch:
        n = len(item)
        # 1. 输入序列：用 pad_token_id 填充到 batch_max_length
        inputs = item + [pad_token_id] * (batch_max_length - n)

        # 2. 目标序列：偏移+1，用ignore_index填充
        # - 有效部分：item[1:] (下一个token预测)
        # - 填充部分：原始结束位置后，先加一个结尾token，然后再填充为ignore_index
        targets = item[1:] + [pad_token_id] + [ignore_index] * (batch_max_length - n)

        # 3. 统一截断处理
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_tensor = torch.tensor(inputs)
        targets_tensor = torch.tensor(targets)

        inputs_list.append(inputs_tensor)
        targets_list.append(targets_tensor)

    # 将数据转换为张量，并将张量移动到指定设备
    inputs_tensor = torch.stack(inputs_list).to(device)
    targets_tensor = torch.stack(targets_list).to(device)

    return inputs_tensor, targets_tensor
