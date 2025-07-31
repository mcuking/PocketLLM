import sys
import json
from pathlib import Path
from argparse import ArgumentParser
import tiktoken
import torch
from model.language_model import LanguageModel
from utils.model_inference import classify_review
from class_finetune_train import adapt_model_for_classification

if __name__ == "__main__":
    """
    初始化模型并进行微调

    Args:
        --config (str): 模型配置参数文件路径
        --model_path (str): 微调后保存模型权重文件路径
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gpt2_config_355M.json")
    parser.add_argument("--model_path", type=str, default="review_classifier.pth")
    args = parser.parse_args()
    config, model_path = vars(args).values()

    # 如果模型权重文件不存在，则提示并退出程序
    if not Path(model_path).exists():
        print(f"模型权重文件 {model_path} 不存在，请先进行模型微调")
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
    # 初始化模型并加载 GPT-2 预训练权重
    ##############################

    model = LanguageModel(cfg)
    model.to(device)

    ##############################
    # 修改模型用于分类任务
    ##############################

    adapt_model_for_classification(model, cfg)

    # 加载之前训练好的模型权重参数，weights_only=True 表示只加载模型参数，不加载优化器等状态信息
    model.load_state_dict(torch.load(model_path, weights_only=True))

    # 切换为推理模式，将禁用 dropout 等只在训练时使用的功能
    model.eval()

    print("开始对话（输入'exit'退出）")
    while True:
        input_text = input("用户: ")
        if input_text.lower() == 'exit':
            break

        # 使用模型进行分类评论
        label = classify_review(
            input_text=input_text, 
            model=model, 
            tokenizer=tokenizer,
            device=device,
            max_length=cfg["context_length"]
        )

        print(f"模型: {label}\n")
