import sys
import json
from pathlib import Path
from argparse import ArgumentParser
from functools import partial
import tiktoken
import torch
from torch.utils.data import DataLoader
from model.language_model import LanguageModel
from utils.dataset_loader import InstructionDataset, custom_collate_fn
from utils.load_gpt2_weights import load_gpt2_weights_into_model
from utils.model_train import train_model

if __name__ == "__main__":
    """
    初始化模型并进行模型微调

    Args:
        --config (str): 模型配置参数文件路径
        --data_path (str): 用于微调的原始数据文件路径
        --model_path (str): 微调后保存模型权重文件路径
        --gpt2_model_path (str): GPT-2 模型权重文件路径
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gpt2_config_355M.json")
    parser.add_argument("--data_path", type=str, default="data/instruction_finetune/instruction-data.json")
    parser.add_argument("--model_path", type=str, default="instruction_executor.pth")
    parser.add_argument("--gpt2_model_path", type=str, default="pytorch_model.bin")
    args = parser.parse_args()
    config, data_path, model_path, gpt2_model_path = vars(args).values()

    # 如果用于微调的原始数据文件不存在，则提示并退出程序
    if not Path(data_path).exists():
        print(f"用于微调的原始数据文件 {data_path} 不存在")
        sys.exit()

    # 如果用于微调的 GPT-2 模型权重文件不存在，则提示并退出程序
    if not Path(gpt2_model_path).exists():
        print(f"用于微调的 GPT-2 模型权重文件 {gpt2_model_path} 不存在")
        sys.exit()

    with open(config) as f:
        cfg = json.load(f)

    # 如果你有一台支持 CUDA 的 GPU 机器，那么大语言模型将自动在 GPU 上训练且不需要修改代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置随机种子以保证结果可复现
    torch.manual_seed(123)

    # 初始化分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    ##############################
    # 准备数据集
    ##############################

    # 读取原始数据集
    with open(data_path) as f:
        data = json.load(f)

    # 切分训练集和验证集，这里将 90% 的数据作为训练集，10% 的数据作为验证集
    train_ratio = 0.9
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    validate_data = data[split_idx:]

    batch_size = 8 # 设置批次大小
    num_workers = 0 # 设置数据加载器的多线程工作线程数
    
    customized_collate_fn = partial(custom_collate_fn, allowed_max_length=cfg["context_length"], device=device)
    # 准备训练数据集
    train_dataset = InstructionDataset(
        data=train_data,
        tokenizer=tokenizer
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size, # 每个批次的大小
        collate_fn=customized_collate_fn, # 自定义批次数据的整理方式
        shuffle=True, # 是否打乱数据顺序 防止模型过拟合
        drop_last=True, # 是否丢弃最后一个不完整的batch
        num_workers=num_workers, # 多线程加载数据的工作线程数
    )

    validate_dataset = InstructionDataset(
        data=validate_data,
        tokenizer=tokenizer
    )
    validate_loader = DataLoader(
        dataset=validate_dataset,
        batch_size=batch_size, # 每个批次的大小
        collate_fn=customized_collate_fn, # 自定义批次数据的整理方式
        shuffle=False, # 是否打乱数据顺序 防止模型过拟合
        drop_last=False, # 是否丢弃最后一个不完整的batch
        num_workers=num_workers, # 多线程加载数据的工作线程数
    )

    ##############################
    # 初始化模型并加载 GPT-2 预训练权重
    ##############################

    model = LanguageModel(cfg)
    model.to(device)

    # 加载了 GPT-2 预训练权重，需要使用 load_gpt2_weights_into_model 将 GPT-2 模型权重加载到自定义模型中
    load_gpt2_weights_into_model(model, gpt2_model_path)
    
    ##############################
    # 微调并保存模型权重
    ##############################

    # 初始化优化器，优化器是用于更新模型权重参数的算法，这里使用 AdamW 算法
    optimizer = torch.optim.AdamW(
        model.parameters(), # .parameters()方法返回模型的所有可训练权重参数
        lr=5e-4, # 学习率，即模型权重参数的梯度下降步长的系数，决定具体变化快慢，即 w = w - lr * dL/dw
        weight_decay=0.1 # 权重衰减，即模型权重参数的 L2 正则化系数
    )

    # 执行模型微调
    train_model(model, train_loader, validate_loader, optimizer, device, num_epochs=2, eval_freq=5, eval_iter=5, is_classification=False)

    # 保存模型权重
    torch.save(model.state_dict(), model_path)
