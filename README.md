# PocketLLM

从零搭建一个大语言模型，完成大模型架构实现/预训练/微调等环节，以达到深入理解模型内部工作机制的目的。

## 项目结构

```
PocketLLM/
│
├── configs/                  # 大模型配置文件目录
│   ├── gpt2_config_1.5B.json # GPT2-1.5B 配置文件
│   ├── gpt2_config_124M.json # GPT2-124M 配置文件
│   ├── gpt2_config_355M.json # GPT2-355M 配置文件
│   └── gpt2_config_774M.json # GPT2-774M 配置文件
│
├── model/                    # 模型定义
│   ├── __init__.py
│   ├── language_model.py     # 大语言模型
│   ├── attention.py          # 多头注意力机制
│   ├── feed_forward.py       # 前馈神经网络
│   └── layer_norm.py         # 层归一化
│
├── utils/                    # 工具脚本
│   ├── __init__.py
│   └── data_processor.py     # 数据预处理
│
├── generate.py               # 文本生成程序
├── README.md                 # 项目说明
└── requirements.txt          # 依赖列表
```

## 安装

```bash
# 克隆仓库
git clone https://github.com/mcuking/PocketLLM.git
cd LLMs-from-scratch

## 确保 python 版本为 3.11，例如 macOS 系统
brew install python@3.11 # macOS

# 创建虚拟环境
uv venv --python=python3.11

# 激活虚拟环境
source .venv/bin/activate # macOS

# 安装依赖
uv pip install -r requirements.txt
```

## 使用示例

### 预训练

待完成

### 文本生成

```bash
# 初始化大模型并开始生成文本 
python generate.py --config ./configs/gpt2_config_124M.json
```

## 参考资料

- [《从零构建大模型》](https://book.douban.com/subject/37305124/) 
- [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)
