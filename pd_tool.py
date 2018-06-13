# split_train_test函数与get_part_data函数
import pandas as pd
import math


def split_train_test(df_data, size=0.8):
    """
        分割训练集和测试集
    """
    # 为保证每个类中的数据能在训练集中和测试集中的比例相同，所以需要依次对每个类进行处理
    df_train = pd.DataFrame()
    df_test = pd.DataFrame()

    labels = df_data['group'].unique().tolist()
    for label in labels:
        # 找出group的记录
        df_w_label = df_data[df_data['group'] == label]
        # 重新设置索引，保证每个类的记录是从0开始索引，方便之后的拆分
        df_w_label = df_w_label.reset_index()

        # 默认按80%训练集，20%测试集分割
        # 这里为了简化操作，取前80%放到训练集中，后20%放到测试集中
        # 当然也可以随机拆分80%，20%（尝试实现下DataFrame中的随机拆分）

        # 该类数据的行数
        n_lines = df_w_label.shape[0]
        split_line_no = math.floor(n_lines * size)
        text_df_w_label_train = df_w_label.iloc[:split_line_no, :]
        text_df_w_label_test = df_w_label.iloc[split_line_no:, :]

        # 放入整体训练集，测试集中
        df_train = df_train.append(text_df_w_label_train)
        df_test = df_test.append(text_df_w_label_test)

    df_train = df_train.reset_index()
    df_test = df_test.reset_index()
    return df_train, df_test


def get_part_data(df_data, percent=1):
    """
        从df_data中按percent选取部分数据
    """
    df_result = pd.DataFrame()
    grouped = df_data.groupby('group')
    for group_name, group in grouped:
        n_group_size = group.shape[0]
        n_part_size = math.floor(n_group_size * percent)
        part_df = group.iloc[:n_part_size, :]
        df_result = df_result.append(part_df)

    return df_result
# Copyright WengerChan
