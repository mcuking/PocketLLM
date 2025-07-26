import torch

def load_gpt2_weights_into_model(model, model_path):
    """
    将 GPT-2 模型权重加载到自定义模型中

    Args:
        model: 自定义模型实例
        model_path: GPT-2 模型路径
    """
    def assign(left, right):
        if left.shape != right.shape:
            raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
        # 直接复制数据到现有张量
        left.data.copy_(right.clone().detach())
    
    # 加载 GPT-2 模型权重
    gpt2_model = torch.load(model_path)
    
    # 1. 处理词嵌入层
    assign(model.tok_emb.weight, gpt2_model["wte.weight"])
    
    # 2. 处理位置嵌入层
    assign(model.pos_emb.weight, gpt2_model["wpe.weight"])
    
    # 3. 处理 Transformer 块
    for layer_idx in range(len(model.trf_blocks)):
        # 3.1 第一个层归一化
        assign(model.trf_blocks[layer_idx].norm1.weight, gpt2_model[f"h.{layer_idx}.ln_1.weight"])
        assign(model.trf_blocks[layer_idx].norm1.bias, gpt2_model[f"h.{layer_idx}.ln_1.bias"])
        
        # 3.2 多头自注意力层
        # 拆分 QKV 权重
        qkv_weight = gpt2_model[f"h.{layer_idx}.attn.c_attn.weight"]
        q_weight, k_weight, v_weight = torch.split(qkv_weight, qkv_weight.size(1)//3, dim=-1)
        assign(model.trf_blocks[layer_idx].attn.W_query.weight, q_weight.T)
        assign(model.trf_blocks[layer_idx].attn.W_key.weight, k_weight.T)
        assign(model.trf_blocks[layer_idx].attn.W_value.weight, v_weight.T)
        
        # 拆分 QKV 偏置
        qkv_bias = gpt2_model[f"h.{layer_idx}.attn.c_attn.bias"]
        q_bias, k_bias, v_bias = torch.split(qkv_bias, qkv_bias.size(0)//3, dim=-1)
        assign(model.trf_blocks[layer_idx].attn.W_query.bias, q_bias)
        assign(model.trf_blocks[layer_idx].attn.W_key.bias, k_bias)
        assign(model.trf_blocks[layer_idx].attn.W_value.bias, v_bias)
        
        # 输出投影层
        assign(model.trf_blocks[layer_idx].attn.out_proj.weight, gpt2_model[f"h.{layer_idx}.attn.c_proj.weight"].T)
        assign(model.trf_blocks[layer_idx].attn.out_proj.bias, gpt2_model[f"h.{layer_idx}.attn.c_proj.bias"])

        # 3.3 第二个层归一化
        assign(model.trf_blocks[layer_idx].norm2.weight, gpt2_model[f"h.{layer_idx}.ln_2.weight"])
        assign(model.trf_blocks[layer_idx].norm2.bias, gpt2_model[f"h.{layer_idx}.ln_2.bias"])

        # 3.4 前馈神经网络
        assign(model.trf_blocks[layer_idx].ffn.layers[0].weight, gpt2_model[f"h.{layer_idx}.mlp.c_fc.weight"].T)
        assign(model.trf_blocks[layer_idx].ffn.layers[0].bias, gpt2_model[f"h.{layer_idx}.mlp.c_fc.bias"])
        assign(model.trf_blocks[layer_idx].ffn.layers[2].weight, gpt2_model[f"h.{layer_idx}.mlp.c_proj.weight"].T)
        assign(model.trf_blocks[layer_idx].ffn.layers[2].bias, gpt2_model[f"h.{layer_idx}.mlp.c_proj.bias"])

    # 4. 处理最后的层归一化
    assign(model.final_norm.weight, gpt2_model["ln_f.weight"])
    assign(model.final_norm.bias, gpt2_model["ln_f.bias"])

    # 5. 处理输出层（权重绑定）
    # GPT-2 模型在其输出层中复用了词元嵌入层的权重，以减少参数总数，这一概念称为权重绑定。
    # 因此这里我们只需将词元嵌入层的权重复制到输出层即可。
    assign(model.out_head.weight, gpt2_model["wte.weight"])

    return model
