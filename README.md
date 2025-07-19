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

## 使用示例

### 预训练

```bash
python pretrain.py --config ./configs/gpt2_config_124M.json
```

### 文本生成

```bash
python generate.py --config ./configs/gpt2_config_124M.json
```

对话内容如下：

```bash
>用户: 你好 
>模型: 你好nexus Archangel pantady paragraph students
```

### 微调

待完成

## 参考资料

- [《从零构建大模型》](https://book.douban.com/subject/37305124/) 
- [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)
- [DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)
