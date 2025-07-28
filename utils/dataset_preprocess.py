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
