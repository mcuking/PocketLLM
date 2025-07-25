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
from utils.metrics import calc_loss_batch, calc_loss_loader
from utils.dataset_preprocess import create_balanced_dataset, random_split

def train_classifier(model, train_loader, validate_loader, optimizer, device, num_epochs, eval_freq, eval_iter):
    '''
    模型分类微调

    Args:
        model: 语言模型
        train_loader: 训练数据集
        validate_loader: 验证数据集
        optimizer: 优化器
        device: 决定训练模型在 CPU 还是 GPU 上运行
        num_epochs: 训练轮次
        eval_freq: 每隔多少个批次打印一次训练集和验证集损失
        eval_iter: 计算数据集损失时使用的批次数
    '''
    global_step = -1
    task_type = "classification"  # 设置任务类型为分类任务

    # 遍历训练轮次，一轮就是完整地遍历一次训练数据集
    for epoch in range(num_epochs):
        # 切换模型为训练模式
        model.train()

        # 在每个训练轮次中遍历批次，批次数量由训练集的大小除以每个批次的大小确定
        for input_batch, target_batch in train_loader:
            # 重置上一个批次迭代中的得到的损失梯度
            optimizer.zero_grad()

            # 计算当前批次的损失
            loss = calc_loss_batch(input_batch, target_batch, model, device, task_type=task_type)

            # 进行反向传播来计算损失梯度
            loss.backward()

            # 使用损失梯度更新模型权重
            optimizer.step()

            global_step += 1
            
            # 打印训练集/验证集损失
            if global_step % eval_freq == 0:
                # 计算模型在训练数据集和验证数据集的损失
                train_loss = calc_loss_loader(train_loader, model, device, eval_iter, task_type=task_type)
                validate_loss = calc_loss_loader(validate_loader, model, device, eval_iter, task_type=task_type)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Validate loss {validate_loss:.3f}")
                
def class_finetune(input_text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    """
    使用模型进行二元分类

    Args:
        input_text: 用户输入的文本
        model: 语言模型
        tokenizer: 分词器
        device: 决定模型在 CPU 还是 GPU 上运行
        max_length: 输入文本的最大长度，根据你的模型不同会限制最大能接受的输入长度
        pad_token_id: 填充 token 的 id, 默认使用 GPT-2 的 token id 50256 即 <|endoftext|> 来填充
    """
    # 将用户输入的文本转换为 token id
    # unsqueeze 方法用于在张量维度上增加一个维度，这里在第一个维度上增加一个维度，使得输入的形状为 (1, num_tokens)
    # 这里使用 unsqueeze 方法是因为模型要求输入的形状为 (batch_size, num_tokens)
    input_ids = tokenizer.encode(input_text, allowed_special={'<|endoftext|>'})

    # 将当前文本截断至支持的长度。如果大模型仅支持5个词元，如果输入文本长度为10，则只有最后5个词元会被用于输入文本
    input_ids = input_ids[-max_length:]

    # 如果输入文本长度小于 max_length，则使用 pad_token_id 填充到 max_length
    input_ids += [pad_token_id] * (max_length - len(input_ids))

    # 增加 batch 维度
    input_tersor = torch.tensor(input_ids, device=device).unsqueeze(0)

    # 使用模型生成文本，输入输出均为 token id
    with torch.no_grad():
        logits = model(input_tersor)[:, -1, :]

    # 将 logits 转换为概率分布，并使用了 argmax 方法找出概率最大的类别标签
    predicted_label = torch.argmax(logits, dim=-1).item()

    # 返回预测的类别标签，0 表示 "not spam"，1 表示 "spam"
    return "spam" if predicted_label == 1 else "not spam"

def main(config, data_path, model_path, gpt2_model_path, eval_mode):
    """
    初始化大模型并进行模型分类微调

    Args:
        --config (str): 模型配置参数文件路径
        --data_path (str): 用于分类微调的原始数据文件路径
        --model_path (str): 分类微调后保存模型权重文件路径
        --gpt2_model_path (str): GPT-2 模型权重文件路径
        --eval_mode (bool): 是否以评估模式运行模型
    """
    # 如果模型权重文件不存在，则提示并退出程序
    if eval_mode and not Path(model_path).exists():
        print(f"模型权重文件 {model_path} 不存在，请先进行模型分类微调")
        sys.exit()

    # 如果用于分类微调的原始数据文件不存在，则提示并退出程序
    if not eval_mode and not Path(data_path).exists():
        print(f"用于分类微调的原始数据文件 {data_path} 不存在")
        sys.exit()

    # 如果用于分类微调的 GPT-2 模型权重文件不存在，则提示并退出程序
    if not eval_mode and not Path(gpt2_model_path).exists():
        print(f"用于分类微调的 GPT-2 模型权重文件 {gpt2_model_path} 不存在")
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
    if not eval_mode:
        # 读取原始数据集
        df = pd.read_csv(data_path, sep="\t", header=None, names=["Label", "Text"])
        balanced_df = create_balanced_dataset(df) # 创建平衡数据集，使得 "spam" 和 "ham" 的数量相等
        balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1}) # 将 "Label" 列中的 "ham" 和 "spam" 转换为 0/1 的标签

        # 切分训练集和验证集，这里将 90% 的数据作为训练集，10% 的数据作为验证集
        train_ratio = 0.9
        train_df, validation_df = random_split(balanced_df, train_ratio)
        train_df.to_csv("train.csv", index=None)
        validation_df.to_csv("validation.csv", index=None)

        batch_size = 8 # 设置批次大小
        num_workers = 0 # 设置数据加载器的多线程工作线程数
        
        train_dataset = SpamDataset(
            csv_file="train.csv",
            max_length=None,
            tokenizer=tokenizer
        )
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            drop_last=True,
        )

        validate_dataset = SpamDataset(
            csv_file="validation.csv",
            max_length=train_dataset.max_length,
            tokenizer=tokenizer
        )
        validate_loader = DataLoader(
            dataset=validate_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            drop_last=False,
        )

    ##############################
    # 初始化模型并加载 GPT-2 预训练权重
    ##############################
    model = LanguageModel(cfg)
    model.to(device)

    if not eval_mode:
        # 加载了 GPT-2 预训练权重，需要使用 load_gpt2_weights_into_model 将 GPT-2 模型权重加载到自定义模型中
        load_gpt2_weights_into_model(model, gpt2_model_path)

    ##############################
    # 修改模型以适应分类微调
    ##############################

    # 由于模型经过了预训练，因此不需要微调所有层。基于神经网络的语言模型中，较低层通常捕捉了通用的语言结构和语义，适用于广泛的任务和数据集。
    # 而最后几层更侧重捕捉特定任务的特征，适用于特定任务的数据集。
    # 这里我们将除最后一层输出层外的所有层设置为冻结，只进行最后一层的微调。
    # 冻结模型，将所有层设为不可训练
    for param in model.parameters():
        param.requires_grad = False
    
    # 二元分类任务，因此我们只需替换最后的输出层即可，
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

    ##############################
    # 评估模式，则直接加载模型权重并进行分类推理
    ##############################
    # 如果 eval_mode 为 True，则切换为评估模式
    if eval_mode:
        # 加载之前训练好的模型权重参数，weights_only=True 表示只加载模型参数，不加载优化器等状态信息
        model.load_state_dict(torch.load(model_path, weights_only=True))

        # 切换为推断模式，将禁用 dropout 等只在训练时使用的功能
        model.eval()

        print("开始对话（输入'exit'退出）")
        while True:
            input_text = input("用户: ")
            if input_text.lower() == 'exit':
                break

            # 调用 class_finetune 函数进行分类微调
            label = class_finetune(input_text, model, tokenizer, device, max_length=cfg["context_length"])
            
            print(f"模型: {label}\n")
        return
    
    ##############################
    # 分类微调
    ##############################
    # 初始化优化器，优化器是用于更新模型权重参数的算法，这里使用 AdamW 算法
    optimizer = torch.optim.AdamW(
        model.parameters(), # .parameters()方法返回模型的所有可训练权重参数
        lr=5e-4, # 学习率，即模型权重参数的梯度下降步长的系数，决定具体变化快慢，即 w = w - lr * dL/dw
        weight_decay=0.1 # 权重衰减，即模型权重参数的 L2 正则化系数
    )

    # 执行模型分类微调
    train_classifier(model, train_loader, validate_loader, optimizer, device, num_epochs=5, eval_freq=50, eval_iter=5)

    # 保存模型权重
    torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gpt2_config_355M.json")
    parser.add_argument("--data_path", type=str, default="data/class_finetune/SMSSpamCollection.csv")
    parser.add_argument("--model_path", type=str, default="review_classifier.pth")
    parser.add_argument("--gpt2_model_path", type=str, default="pytorch_model.bin")
    parser.add_argument("--eval_mode", type=bool, default=False)
    args = parser.parse_args()
    main(args.config, args.data_path, args.model_path, args.gpt2_model_path, args.eval_mode)
