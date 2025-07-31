import sys
import json
from pathlib import Path
from argparse import ArgumentParser
import pandas as pd
import tiktoken
import torch
from torch.utils.data import DataLoader
from model.language_model import LanguageModel
from utils.dataset_loader import SpamDataset
from utils.load_gpt2_weights import load_gpt2_weights_into_model
from utils.model_train import train_model

def adapt_model_for_classification(model, cfg):
    """修改模型用于分类任务"""
    # 由于模型经过了预训练，因此不需要微调所有层。基于神经网络的语言模型中，较低层通常捕捉了通用的语言结构和语义，适用于广泛的任务和数据集。
    # 而最后几层更侧重捕捉特定任务的特征，适用于特定任务的数据集。
    # 这里我们将除最后一层输出层外的所有层设置为冻结，只进行最后一层的微调。
    # 冻结模型，将所有层设为不可训练
    for param in model.parameters():
        param.requires_grad = False
    
    # 评论分类任务，因此我们只需替换最后的输出层即可，
    # 该层原本是将输入映射为 50257 维的向量，即词汇表的大小，
    # 现在将其输出层作用改为映射为 2 维的向量，即 0/1 两类的分类器
    # 新的输出层的 requires_grad 默认为 True，意味着该层是模型中唯一在训练过程中会被更新的层
    num_classes = 2
    model.out_head = torch.nn.Linear(cfg["emb_dim"], num_classes)

    # 为了更好的训练效果，额外将最后一个 Transformer 块和最后层归一化设置为可训练
    for params in model.trf_blocks[-1].parameters():
        params.requires_grad = True

    for params in model.final_norm.parameters():
        params.requires_grad = True


if __name__ == "__main__":
    """
    初始化模型并进行微调

    Args:
        --config (str): 模型配置参数文件路径
        --data_path (str): 用于微调的原始数据文件路径
        --model_path (str): 微调后保存模型权重文件路径
        --gpt2_model_path (str): GPT-2 模型权重文件路径
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gpt2_config_355M.json")
    parser.add_argument("--data_path", type=str, default="data/class_finetune/SMSSpamCollection.csv")
    parser.add_argument("--model_path", type=str, default="review_classifier.pth")
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
    df = pd.read_csv(data_path, sep="\t", header=None, names=["Label", "Text"])
    num_spam = df[df["Label"] == "spam"].shape[0] # 计算 "spam" 实例的数量
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123) # 随机抽样 "ham" 实例，使其数量与 "spam" 实例相等
    df = pd.concat([ham_subset, df[df["Label"] == "spam"]]) # 合并 "ham" 子集和所有 "spam" 实例
    df["Label"] = df["Label"].map({"ham": 0, "spam": 1}) # 将 "Label" 列中的 "ham" 和 "spam" 转换为 0/1 的标签
    df = df.sample(frac=1, random_state=123).reset_index(drop=True) # 将整个 DataFrame 打乱顺序

    # 切分训练集和验证集，这里将 90% 的数据作为训练集，10% 的数据作为验证集
    train_ratio = 0.9
    split_idx = int(len(df) * train_ratio)
    train_data = df[:split_idx]
    validate_data = df[split_idx:]

    batch_size = 8 # 设置批次大小
    num_workers = 0 # 设置数据加载器的多线程工作线程数
    
    # 准备训练数据集
    train_dataset = SpamDataset(
        data=train_data,
        tokenizer=tokenizer,
        max_length=None
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size, # 每个批次的大小
        shuffle=True, # 是否打乱数据顺序 防止模型过拟合
        num_workers=num_workers, # 多线程加载数据的工作线程数
        drop_last=True, # 是否丢弃最后一个不完整的batch
    )

    validate_dataset = SpamDataset(
        data=validate_data,
        tokenizer=tokenizer,
        max_length=train_dataset.max_length
    )
    validate_loader = DataLoader(
        dataset=validate_dataset,
        batch_size=batch_size, # 每个批次的大小
        shuffle=False, # 是否打乱数据顺序 防止模型过拟合
        num_workers=num_workers, # 多线程加载数据的工作线程数
        drop_last=False, # 是否丢弃最后一个不完整的batch
    )

    ##############################
    # 初始化模型并加载 GPT-2 预训练权重
    ##############################

    model = LanguageModel(cfg)
    model.to(device)

    # 加载了 GPT-2 预训练权重，需要使用 load_gpt2_weights_into_model 将 GPT-2 模型权重加载到自定义模型中
    load_gpt2_weights_into_model(model, gpt2_model_path)

    ##############################
    # 修改模型用于分类任务
    ##############################

    adapt_model_for_classification(model, cfg)

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
    train_model(model, train_loader, validate_loader, optimizer, device, num_epochs=5, eval_freq=50, eval_iter=5, is_classification=True)

    # 保存模型权重
    torch.save(model.state_dict(), model_path)
