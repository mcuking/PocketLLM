from .metrics import calc_loss_batch, calc_loss_loader

def train_model(model, train_loader, validate_loader, optimizer, device, num_epochs, eval_freq, eval_iter, is_classification=False):
    '''
    模型预训练/微调方法

    Args:
        model: 语言模型
        train_loader: 训练数据集
        validate_loader: 验证数据集
        optimizer: 优化器
        device: 决定训练模型在 CPU 还是 GPU 上运行
        num_epochs: 训练轮次
        eval_freq: 每隔多少个批次打印一次训练集和验证集损失
        eval_iter: 计算数据集损失时使用的批次数
        is_classification: 是否为分类任务，如果是，那么只取每个 batch 中的最后一个 token 的 logits 计算损失
    '''
    global_step = -1

    # 遍历训练轮次，一轮就是完整地遍历一次训练数据集
    for epoch in range(num_epochs):
        # 切换模型为训练模式
        model.train()

        # 在每个训练轮次中遍历批次，批次数量由训练集的大小除以每个批次的大小确定
        for input_batch, target_batch in train_loader:
            # 重置上一个批次迭代中的得到的损失梯度
            optimizer.zero_grad()

            # 计算当前批次的损失
            loss = calc_loss_batch(input_batch, target_batch, model, device, is_classification=is_classification)

            # 进行反向传播来计算损失梯度
            loss.backward()

            # 使用损失梯度更新模型权重
            optimizer.step()

            global_step += 1
            
            # 打印训练集/验证集损失
            if global_step % eval_freq == 0:
                # 计算模型在训练数据集和验证数据集的损失
                train_loss = calc_loss_loader(train_loader, model, device, eval_iter, is_classification=is_classification)
                validate_loss = calc_loss_loader(validate_loader, model, device, eval_iter, is_classification=is_classification)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Validate loss {validate_loss:.3f}")
