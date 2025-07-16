import json
from argparse import ArgumentParser
import torch
from model.language_model import LanguageModel
from utils.data_loader import create_dataloader
from utils.train_utils import calc_loss_batch

def train_model(model, train_loader, optimizer, device, num_epochs):
    '''
    用于大语言模型训练
    Args:
        model: 语言模型
        train_loader: 训练数据集
        optimizer: 优化器
        device: 决定训练模型在 CPU 还是 GPU 上运行
        num_epochs: 训练轮次
    '''
    # 遍历训练轮次，一轮就是完整地遍历一次训练数据集
    for _ in range(num_epochs):
        # 切换模型为训练模式
        model.train()

        # 在每个训练轮次中遍历批次，批次数量由训练集的大小除以每个批次的大小确定
        for input_batch, target_batch in train_loader:
            # 重置上一个批次迭代中的得到的损失梯度
            optimizer.zero_grad()

            # 计算当前批次的损失
            loss = calc_loss_batch(input_batch, target_batch, model, device)

            # 进行反向传播来计算损失梯度
            loss.backward()

            # 使用损失梯度更新模型权重
            optimizer.step()

def main(config):
    """
    初始化大模型并进行模型训练

    Arguments:
        --config (str): 模型配置参数文件路径
    """
    # 准备预训练数据
    file_path = "data/西游记.txt"
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    # 准备训练数据集
    train_loader = create_dataloader(
        text_data,
        batch_size=2,
        max_length=cfg["context_length"],
        stride=cfg["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    # 如果你有一台支持 CUDA 的 GPU 机器，那么大语言模型将自动在 GPU 上训练且不需要修改代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置随机种子以保证结果可复现
    torch.manual_seed(123)
    with open(config) as f:
        cfg = json.load(f)

    # 初始化模型
    model = LanguageModel(cfg)
    model.to(device)

    # 初始化优化器，优化器是用于更新模型权重参数的算法，这里使用 AdamW 算法
    optimizer = torch.optim.AdamW(
        model.parameters(), # .parameters()方法返回模型的所有可训练权重参数
        lr=0.0004, # 学习率，即模型权重参数的更新步长
        weight_decay=0.1 # 权重衰减，即模型权重参数的 L2 正则化系数
    )

    # 开始模型训练
    train_model(model, train_loader, optimizer, device, num_epochs=10)

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
