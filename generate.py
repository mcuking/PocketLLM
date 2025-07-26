import sys
import json
from pathlib import Path
from argparse import ArgumentParser
import tiktoken
import torch
from model.language_model import LanguageModel
from utils.text_tokenizer import text_to_token_ids, token_ids_to_text
from utils.load_gpt2_weights import load_gpt2_weights_into_model

def generate_text(model, context_length, token_ids, max_new_tokens, temperature, top_k):
    """
    使用模型生成文本
    
    Args:
        model: 语言模型
        context_length: 最大上下文长度
        token_ids: 输入文本的 token ids 张量，形状为 (batch_size, num_tokens)
        max_new_tokens: 新生成的 token 最大数量
        temperature: 温度，用于控制生成文本的随机性，值越大越随机，值越小越确定
        top_k: top-k 采样，只从概率最高的 k 个 token 中采样，值越大越随机，值越小越确定
    """
    for _ in range(max_new_tokens):
        # 将当前文本截断至支持的长度。如果大模型仅支持5个词元，如果输入文本长度为10，则只有最后5个词元会被用于输入文本
        real_token_ids = token_ids[:, -context_length:]

        # no_grad 用于禁用梯度计算，只有在训练时才需要计算梯度来减小损失函数，禁用后可以加速计算减少内存占用
        with torch.no_grad():
            logits = model(real_token_ids)
        
        # 因为模型会为每个 token 生成一个 logits，而我们只需要最后一个 token 的 logits，所以需要将维度减少一个维度，
        # 使得形状从 (batch_size, num_tokens, vocab_size) 变为 (batch_size, vocab_size)
        logits = logits[:, -1, :]

        if top_k is not None:
            # 先取出概率最高的 top_k 个 token
            top_logits, _ = torch.topk(logits, top_k)
            # 然后取这些 token 中最小的概率值
            min_value = top_logits[:, -1]
            # 将小于这个概率值的 token 的概率置为负无穷，下面 softmax/argmax 时这些 token 的概率会变为 0
            logits = torch.where(logits < min_value, -torch.inf, logits)

        if temperature > 0.0:
            # 可以理解为当 temperature 越小，被除之后之前的概率差距会越大，越容易生成确定性的文本
            # 而当 temperature 越大，被除之后之前的概率差距会越小，越容易生成随机性的文本
            logits = logits / temperature
            # 将 logits 分数转换为概率分布，不会改变输入顺序
            probabilities = torch.softmax(logits, dim=-1)
            # 使用 torch.multinomial 方法从概率分布中采样，返回形状为 (batch_size, 1) 的张量，表示每个样本的 token id
            next_token_id = torch.multinomial(probabilities, num_samples=1)
        else:
            # 当禁用温度缩放时，就采用贪心解码，即选择概率最大的 token id 作为下一个 token
            # argmax 方法用于返回张量中每个元素的最大值的索引，
            # 这里返回的是概率最大的词元的 token id，形状为 (batch_size, 1)
            next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

        # 将新生成的 token id 添加到文本末尾，继续下一个循环，生成下一个 token
        token_ids = torch.cat((token_ids, next_token_id), dim=-1)
    return token_ids

def main(config, model_path, max_new_tokens, temperature, top_k):
    """
    初始化大模型并执行文本生成

    Args:
        --config (str): 模型配置参数文件路径
        --model_path (str): 模型权重文件路径
        --max_new_tokens (int): 新生成的 token 最大数量
        --temperature (float): 温度，用于控制生成文本的随机性，值越大越随机，值越小越确定
        --top_k (int): top-k 采样，只从概率最高的 k 个 token 中采样，值越大越随机，值越小越确定
    """
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
    # 初始化模型
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

    # 切换为推断模式，将禁用 dropout 等只在训练时使用的功能
    model.eval()

    ##############################
    # 交互式对话
    ##############################
    print("开始对话（输入'exit'退出）")
    while True:
        user_input = input("用户: ")
        if user_input.lower() == 'exit':
            break

        # 将用户输入的文本转换为 token id
        # unsqueeze 方法用于在张量维度上增加一个维度，这里在第一个维度上增加一个维度，使得输入的形状为 (1, num_tokens)
        # 这里使用 unsqueeze 方法是因为模型要求输入的形状为 (batch_size, num_tokens)
        token_ids = text_to_token_ids(user_input, tokenizer)

        # 使用模型生成文本，输入输出均为 token id
        output_ids = generate_text(
            model=model,
            context_length=cfg["context_length"],
            token_ids=token_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )

        # 将 token id 转换为文本并打印
        # squeeze 方法用于在张量维度上减少一个维度，这里在第一个维度上减少一个维度，使得输出的形状为 (num_tokens)
        # 这里使用 squeeze 方法是因为模型输出的形状为 (batch_size, num_tokens)，而我们只需要一个文本，因此需要减少一个维度
        output_text = token_ids_to_text(output_ids, tokenizer)
        print(f"模型: {output_text}\n")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gpt2_config_124M.json")
    parser.add_argument("--model_path", type=str, default="model.pth")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_k", type=int, default=None)
    args = parser.parse_args()
    main(args.config, args.model_path, args.max_new_tokens, args.temperature, args.top_k)
