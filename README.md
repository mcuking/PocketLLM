# PocketLLM

从零搭建一个大语言模型，完成架构实现/预训练/微调等环节，以达到深入理解模型内部工作机制的目的。

## 项目结构

```
PocketLLM/
│
├── configs/                  # 模型配置文件目录
│   └── ...
│
├── data/                     # 预训练数据
│   └── ...
│
├── model/                    # 模型定义
│   ├── __init__.py
│   ├── attention.py          # 多头注意力
│   ├── feed_forward.py       # 前馈神经网络
│   ├── language_model.py     # 语言模型
│   └── layer_norm.py         # 层归一化
│   └── transform_block.py    # Transformer 块
│
├── utils/                    # 工具方法
│   ├── __init__.py
│   ├── data_loader.py        # 数据加载器
│   ├── text_utils.py         # 文本和 token id 互转方法
│   └── train_utils.py        # 模型预训练/微调用到的公共方法，如损失计算等
│
├── generate.py               # 文本生成程序
├── pretrain.py               # 预训练程序
├── README.md                 # 项目说明
└── requirements.txt          # 依赖列表
```

## 安装

```bash
# 安装 uv，可参考官方文档：
https://github.com/astral-sh/uv?tab=readme-ov-file#installation

# 克隆仓库
git clone https://github.com/mcuking/PocketLLM.git
cd PocketLLM

# 创建虚拟环境
uv venv --python 3.12

# 激活虚拟环境
source .venv/bin/activate # Unix/macOS 系统
# Windows 系统使用：
# .venv\Scripts\activate

# 安装依赖
uv pip install -r requirements.txt
```

## 预训练

### 使用示例

```bash
python pretrain.py --config configs/gpt2_config_124M.json --data_path data/西游记.txt --model_path model.pth
```

### 参数
| 参数 | 说明 | 是否必填 | 默认值 |
| --- | --- | --- | --- |
| `config` | 模型配置文件路径 | 否 | `configs/gpt2_config_124M.json` |
| `data_path` | 用于预训练的原始数据文件路径 | 否 | `data/西游记.txt` |
| `model_path` | 预训练后保存模型权重文件路径 | 否 | `model.pth` |

**注意：** 

因为需要在个人电脑上进行预训练，所以默认使用 gpt2-124M 模型配置进行预训练，如果电脑配置较低。还可以将默认模型配置中的 context_length 调成 256。

另外仓库中提供的 `data/西游记.txt` 是数据较大，训练时间较长，可以选择其他较小的数据文件进行预训练。

### 输出示例

```bash
Ep 1 (Step 000000): Train loss 9.375, Validate loss 9.433
Ep 1 (Step 000100): Train loss 3.053, Validate loss 3.502
Ep 1 (Step 000200): Train loss 3.238, Validate loss 3.181
...
```

## 文本生成

### 使用示例

```bash
python generate.py --config configs/gpt2_config_124M.json --model_path model.pth --max_new_tokens 30 --temperature 0.9 --top_k 5
```

### 参数
| 参数 | 说明 | 是否必填 | 默认值 |
| --- | --- | --- | --- |
| `config` | 模型配置文件路径 | 否 | `configs/gpt2_config_124M.json` |
| `model_path` | 模型权重文件路径 | 否 | `model.pth` |
| `max_new_tokens` | 新生成的 token 最大数量 | 否 | 30 |
| `temperature` | 温度，用于控制生成文本的随机性，值越大越随机，值越小越确定 | 否 | 0.0 |
| `top_k` | top-k 采样，只从概率最高的 k 个 token 中采样，值越大越随机，值越小越确定 | 否 | None |

**注意：** 

因为文本生成时使用的是预训练好的模型，所以需要使用和预训练时相同的模型配置文件。

### 输出示例

```bash
用户: 你好 
模型: 你好nexus Archangel pantady paragraph students
```

## 分类微调

待完成

## 指令微调

待完成

## 参考资料

- [《从零构建大模型》](https://book.douban.com/subject/37305124/) 
- [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)
