# PocketLLM

从零搭建一个大语言模型，完成架构实现/预训练/微调等环节，以达到深入理解模型内部工作机制的目的。

## 项目结构

```
PocketLLM/
│
├── configs/                       # 模型配置文件目录
│   └── ...
│
├── data/                          # 用于预训练或微调的数据
│   └── ...
│
├── model/                         # 模型定义
│   ├── __init__.py
│   ├── attention.py               # 多头注意力
│   ├── feed_forward.py            # 前馈神经网络
│   ├── language_model.py          # 语言模型
│   └── layer_norm.py              # 层归一化
│   └── transform_block.py         # Transformer 块
│
├── utils/                         # 工具方法
│   ├── __init__.py
│   ├── dataset_loader.py          # 多种加载数据集的类
│   ├── load_gpt2_weights.py       # 将 GPT-2 模型权重加载到自定义模型的方法
│   ├── metrics.py                 # 模型评估指标计算方法，例如计算损失值或预测准确率等
│   ├── model_inference.py         # 模型推理方法，例如生成文本/分类评论/执行指令等
│   └── model_train.py             # 模型预训练/微调方法
│
├── class_finetune_inference.py    # 分类微调推理程序
├── class_finetune_train.py        # 分类微调训练程序
├── generate.py                    # 文本生成程序
├── pretrain.py                    # 预训练程序
├── README.md                      # 项目说明
└── requirements.txt               # 依赖列表
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
python pretrain.py --config configs/gpt2_config_124M.json --data_path data/pretrain/西游记.txt --model_path model.pth
```

#### 参数
| 参数 | 说明 | 是否必填 | 默认值 |
| --- | --- | --- | --- |
| `config` | 模型配置参数文件路径 | 否 | `configs/gpt2_config_124M.json` |
| `data_path` | 用于预训练的原始数据文件路径 | 否 | `data/pretrain/西游记.txt` |
| `model_path` | 预训练后保存模型权重文件路径 | 否 | `model.pth` |

**注意：** 

因为需要在个人电脑上进行预训练，所以默认使用 gpt2-124M 模型配置进行预训练，如果电脑配置较低。还可以将默认模型配置中的上下文长度 context_length 调成 256。

另外仓库中提供的预训练数据 `data/pretrain/西游记.txt` 选择的是小说《西游记》最后两回内容。如果仍觉得训练时间较长，可以选择其他更小的数据集进行预训练。

## 文本生成

### 使用示例

```bash
python generate.py --config configs/gpt2_config_124M.json --model_path model.pth --max_new_tokens 30 --temperature 0.9 --top_k 5
```

#### 参数
| 参数 | 说明 | 是否必填 | 默认值 |
| --- | --- | --- | --- |
| `config` | 模型配置参数文件路径 | 否 | `configs/gpt2_config_124M.json` |
| `model_path` | 模型权重文件路径 | 否 | `model.pth` |
| `max_new_tokens` | 新生成的 token 最大数量 | 否 | `50` |
| `temperature` | 温度，用于控制生成文本的随机性，值越大越随机，值越小越确定 | 否 | `0.0` |
| `top_k` | top-k 采样，只从概率最高的 k 个 token 中采样，值越大越随机，值越小越确定 | 否 | `None` |

**注意：** 因为文本生成时使用的是预训练好的模型，所以需要使用和预训练时相同的模型配置文件。

效果如下：

```bash
用户: 师徒方登岸整理，
模型: 师徒方登岸整理，下颈左边，沙崪福寺大小僧人，看见几株松树一项，一脚踏著老鼌者却叫道：“老鼋，好生走稳
```

### 使用 GPT-2 medium 模型权重

由于上面预训练使用的数据集有限，所以生成的文本质量不高，如果想生成质量更高的文本，可以使用 GPT-2 medium 模型权重文件进行文本生成。而 GPT-2 是使用 TersorFlow 训练的，如果直接加载 OpenAI 官网提供的权重文件到本项目的模型中会报错，需要转换成 PyTorch 格式才能使用。

建议直接从 [https://huggingface.co/gpt2](https://huggingface.co/gpt2) 下载已经转为 PyTorch 格式的权重文件 `pytorch_model.bin` 到项目根目录即可，地址如下：

[https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin](https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin)

**注意：** 下载后需要将文件重命名为 `pytorch_model.bin`

然后使用以下命令进行文本生成：

```bash
python generate.py --config configs/gpt2_config_355M.json --model_path pytorch_model.bin
```

程序中会调用 `load_gpt2_weights_into_model` 方法将权重文件加载到模型中，该方法主要解决 GPT-2 和本项目模型的参数命名规范不同的问题。

## 分类微调

### 微调训练

本节主要是在 GPT-2 medium 预训练模型基础上进行分类微调，使得模型能够将消息分类为正常/垃圾。因此需要提前将 GPT-2 medium 预训练模型权重文件下载到本地，具体操作可参考文本生成那节内容。

预训练权重文件下载后执行下面的命令即可开启分类微调训练过程。

```bash
python class_finetune_train.py --config configs/gpt2_config_355M.json --data_path data/class_finetune/SMSSpamCollection.csv --model_path review_classifier.pth --gpt2_model_path pytorch_model.bin
```

#### 参数
| 参数 | 说明 | 是否必填 | 默认值 |
| --- | --- | --- | --- |
| `config` | 模型配置参数文件路径 | 否 | `configs/gpt2_config_355M.json` |
| `data_path` | 用于分类微调的原始数据文件路径 | 否 | `data/class_finetune/SMSSpamCollection.csv` |
| `model_path` | 分类微调后保存模型权重文件路径 | 否 | `review_classifier.pth` |
| `gpt2_model_path` | GPT-2 模型权重文件路径| 否 | `pytorch_model.bin` |

### 分类消息

当完成分类微调训练后，就可以使用该模型对消息进行分类了。运行命令如下：

```bash
python class_finetune_inference.py --config configs/gpt2_config_355M.json --model_path review_classifier.pth
```

效果如下：

```bash
用户: You are a winner you have been specially selected to recieve $1000 cash.
模型: spam
用户: Hey, just wanted to check if we are still on for dinner tonight?
模型: not spam
```

## 指令微调

### 微调训练

本节主要是在 GPT-2 medium 预训练模型基础上进行指令微调，使得模型能够执行用户输入的指令。因此需要提前将 GPT-2 medium 预训练模型权重文件下载到本地，具体操作可参考文本生成那节内容。

预训练权重文件下载后执行下面的命令即可开启指令微调训练过程。

```bash
python instruction_finetune_train.py --config configs/gpt2_config_355M.json --data_path data/instruction_finetune/instruction-data.json --model_path instruction_executor.pth --gpt2_model_path pytorch_model.bin
```

#### 参数
| 参数 | 说明 | 是否必填 | 默认值 |
| --- | --- | --- | --- |
| `config` | 模型配置参数文件路径 | 否 | `configs/gpt2_config_355M.json` |
| `data_path` | 用于指令微调的原始数据文件路径 | 否 | `data/instruction_finetune/instruction-data.json` |
| `model_path` | 指令微调后保存模型权重文件路径 | 否 | `instruction_executor.pth` |
| `gpt2_model_path` | GPT-2 模型权重文件路径| 否 | `pytorch_model.bin` |

### 执行用户指令

当完成指令微调训练后，就可以使用该模型执行用户指令了。运行命令如下：

```bash
python instruction_finetune_inference.py --config configs/gpt2_config_355M.json --model_path instruction_executor.pth
```

效果如下：

```bash
任务指令: Combine the two sentences into a single coherent sentence.
任务输入: She did not attend the meeting. She was ill.
模型: She did not attend the meeting because she was ill.

任务指令: What is the opposite of 'retain'?
任务输入: 
模型: The opposite of 'retain' is 'release'.
```

## 参考资料

- [《从零构建大模型》](https://book.douban.com/subject/37305124/) 
- [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)
