from argparse import ArgumentParser
import json

import torch
import tiktoken

from model.language_model import LanguageModel

def main(config):
    """
    初始化大模型并执行文本生成

    Arguments:
        --config (str): 模型配置参数文件路径
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    # 将输入数据扩展为批处理（batch）的形式
    # 在实际应用中，我们通常会处理多个输入样本。
    # 例如，在自然语言处理任务中，一个批处理可能包含多个句子，每个句子都是一个独立的输入样本。
    # 此时 batch 的形状为 (batch_size, context_length, d_in)，
    # context_length 是上下文长度（句子中的词元数量），d_in 是输入向量的维度
    batch = torch.stack(batch, dim=0)
    print("batch: \n", batch)

    # 设置随机种子以保证结果可复现
    torch.manual_seed(123)    
    with open(config) as f:
        cfg = json.load(f)
        print("cfg: \n", cfg)

    model = LanguageModel(cfg)

    logits = model(batch)
    print("logits: \n", logits)

if __name__ == "__main__":
    """
    命令行工具

    Arguments:
        --config (str): 模型配置参数文件路径
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
