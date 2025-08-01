import sys
import json
from pathlib import Path
from argparse import ArgumentParser
import tiktoken
import torch
from model.language_model import LanguageModel
from utils.load_gpt2_weights import load_gpt2_weights_into_model
from utils.model_inference import generate_text

if __name__ == "__main__":
    """
    初始化大模型并执行文本生成

    Args:
        --config (str): 模型配置参数文件路径
        --model_path (str): 模型权重文件路径
        --max_new_tokens (int): 新生成的 token 最大数量
        --temperature (float): 温度，用于控制生成文本的随机性，值越大越随机，值越小越确定
        --top_k (int): top-k 采样，只从概率最高的 k 个 token 中采样，值越大越随机，值越小越确定
    """
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gpt2_config_124M.json")
    parser.add_argument("--model_path", type=str, default="model.pth")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=None)
    args = parser.parse_args()

    config, model_path, max_new_tokens, temperature, top_k = vars(args).values()
    
    # 如果模型权重文件不存在，则提示并退出程序
    if not Path(model_path).exists():
        print(f"模型权重文件 {model_path} 不存在，请先训练模型")
        sys.exit()

    with open(config) as f:
        cfg = json.load(f)

    # 设置随机种子以保证结果可复现
    torch.manual_seed(123)

    # 加载编码器，默认为 gpt2
    tokenizer = tiktoken.get_encoding("gpt2")

    ##############################
    # 初始化模型并加载模型权重文件
    ##############################

    # 创建模型
    model = LanguageModel(cfg)

    if model_path.endswith("pytorch_model.bin"):
        # 如果权重文件名为 pytorch_model.bin，表明是加载了 GPT-2 预训练权重，
        # 需要使用 load_gpt2_weights_into_model 将 GPT-2 模型权重加载到自定义模型中
        load_gpt2_weights_into_model(model, model_path)
    else:
        # 加载之前训练好的模型权重参数，weights_only=True 表示只加载模型参数，不加载优化器等状态信息
        model.load_state_dict(torch.load(model_path, weights_only=True))

    # 切换为推理模式，将禁用 dropout 等只在训练时使用的功能
    model.eval()

    ##############################
    # 交互式对话
    ##############################

    print("开始对话（输入'exit'退出）\n")
    while True:
        input_text = input("用户: ")
        if input_text.lower() == '':
            print("输入不能为空！")
            continue
        if input_text.lower() == 'exit':
            break

        # 使用模型生成文本
        output_text = generate_text(
            input_text=input_text,
            model=model,
            tokenizer=tokenizer,
            context_length=cfg["context_length"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )

        print(f"模型: {output_text}\n")
