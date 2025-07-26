import pandas as pd

def create_balanced_dataset(df):
    '''
    创建一个平衡数据集，使得 "spam" 和 "ham" 的数量相等

    Args:
        df (DataFrame): 原始数据集
    '''
    # 计算 "spam" 实例的数量
    num_spam = df[df["Label"] == "spam"].shape[0]
    # 随机抽样 "ham" 实例，使其数量与 "spam" 实例相等
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)
    # 合并 "ham" 子集和所有 "spam" 实例
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])
    return balanced_df

def random_split(df, train_ratio):
    '''
    将原始数据集按比例随机划分为训练集和验证集

    Args:
        df (DataFrame): 原始数据集
        train_ratio (float): 训练集占总数据集的比例
    '''
    # 将整个 DataFrame 打乱顺序
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    # 计算训练集的结束索引
    train_end = int(len(df) * train_ratio)
    # 拆分训练集和验证集
    train_df = df[:train_end]
    validation_df = df[train_end:]
    return train_df, validation_df
