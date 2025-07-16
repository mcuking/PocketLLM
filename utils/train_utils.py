import torch

def calc_loss_batch(input_batch, target_batch, model, device):
    """
    计算给定批次的交叉熵损失（负平均对数概率）
    :param input_batch: 输入 token id 的 batch
    :param target_batch: 目标 token id 的 batch
    :param model: 语言模型
    :param device: 设备
    :return: 损失值
    """
    # device 可以将数据转移到 GPU 上
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)

    # 计算损失 loss 过程：
    # target_batch 的形状为 (batch_size, num_tokens)
    # 例如
    # [[3626, 6100, 345 ], # [" effort moves you", 
    # [1107, 588, 11311]] # " really like chocolate"]
    #
    # 第 1 步，通过模型计算 logits，形状为 (batch_size, num_tokens, vocab_size)
    # 例如
    # [[[ 0.3613, 0.4222, -0.0711, ..., 0.3483, 0.4661, -0.2838],
    # [-0.1792, -0.5660, -0.9485, ..., 0.0477, 0.5181, -0.3168],
    # [ 0.7120, 0.0332, 0.1085, ..., 0.1018, -0.4327, -0.2553]],
    #
    # [[-0.2564, 0.0900, 0.0335, ..., 0.2659, 0.4454, -0.6806],
    # [ 0.1230, 0.3653, -0.2074, ..., 0.7705, 0.2710, 0.2246],
    # [ 1.0558, 1.0318, -0.2800, ..., 0.6936, 0.3205, -0.3178]]]
    #
    # 第 2 步，用 softmax 将 logits 转换为概率分布，得到 probabilities，形状仍为 (batch_size, num_tokens, vocab_size)
    # 例如
    # [[[0.00009, 0.00002, 0.00003, ..., 0.00006, 0.00008, 0.00004],
    # [0.00009, 0.00001, 0.00003, ..., 0.00005, 0.00008, 0.00004],
    # [0.00005, 0.00002, 0.00003, ..., 0.00007, 0.00008, 0.00004]],
    #
    # [[0.00009, 0.00002, 0.00003, ..., 0.00006, 0.00008, 0.00004],
    # [0.00009, 0.00001, 0.00003, ..., 0.00005, 0.00008, 0.00004],
    # [0.00005, 0.00002, 0.00003, ..., 0.00007, 0.00008, 0.00004]]]
    #
    # 第 3 步，根据 target_batch 中目标 token 的 id，从 probabilities 最后一个维度中取出对应目标 token 的在本次实际预测的概率，
    # 得到 target_probabilities，形状仍为 (batch_size, num_tokens)
    # 例如
    # [[0.009,
    # 0.00002,
    # 0.00003],
    # [0.00006,
    # 0.00008,
    # 0.00004]]
    # 到这里就可以计算损失了，目的就是让目标 token 的预测概率尽量接近 1。
    # 但是为了后面更好计算偏导数，以便进行梯度下降来更新参数使损失函数值减小，
    # 第 4 步，将这些概率对数化（这样更加可微方便求导），
    # 即 torch.log(torch.cat((target_probabilities_1, target_probabilities_2...))) 
    # 这里将不同 batch 的概率对数化（底数为自然数 e）后拼接成一个向量，得到对数概率 target_probabilities_log，
    # 例如
    # [-9.21034025,
    # -10.99938,
    # -11.50390625,
    # -10.99945,
    # -10.999999,
    # -11.503905]
    # 第 5 步，求平均对数概率，例如得到 -10.7940
    # 因为 ln1=0，所以之前是为了让概率尽量接近 1，取完对数后就是让平均对数概率尽量接近 0。
    # 第 6 步，在深度学习中，通常的做法不是将平均对数概率升至 0，而是将负平均对数概率降至 0。负平均对数概率就是平均对数概率乘以-1，得到 10.7940
    # 在深度学习中，将-10.7940 这个负值转换为 10.7940 的术语称为交叉熵损失。
    # 在实践中，“交叉熵”和“负平均对数概率”这两个术语是相关的，且经常可以互换使用。
    #
    # pytorch 中有一个内置的 cross_entropy 函数，该函数可以为我们处理上述所有步骤，因此我们只需要调用该函数即可。
    loss = torch.nn.functional.cross_entropy(
        # 抹平第 0 维即 batch_size 维，将 logits 的形状从 (batch_size, num_tokens, vocab_size) 转换为 (batch_size * num_tokens, vocab_size)
        logits.flatten(0, 1),
        # 抹平第 0 维即 batch_size 维，将 target_batch 的形状从 (batch_size, num_tokens) 转换为 (batch_size * num_tokens)
        target_batch.flatten(0),
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    计算给定数据集的交叉熵损失（负平均对数概率），就是将数据集的每个 batch 的损失加起来，然后除以 batch 数量，求平均值。
    :param data_loader: 数据集
    :param model: 语言模型
    :param device: 决定模型在 CPU 还是 GPU 上运行
    :param num_batches: 计算损失时使用的批次数
    :return: 损失值
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # 如果传入了 num_batches，需要确保 num_batches 不大于数据集的 batch 数量
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            # 计算每个 batch 的损失 loss
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # 将数据集的每个 batch 的损失 loss 加起来
            total_loss += loss.item()
        else:
            break
    # 将数据集中所有 batch 的损失 loss 平均值
    return total_loss / num_batches

