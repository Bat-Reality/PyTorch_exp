import torch


def read_file(path):
    """
    读取文本文件进行预处理
    :param path: 文件路径
    :return: 分词后的数组
    """
    if 'training_label' in path:
        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]
            x = [line[2:] for line in lines]
            y = [line[0] for line in lines]
        return x, y
    elif 'training_nolabel' in path:
        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
        return x
    else:
        with open(path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            x = ["".join(line[1:].strip('\n').split(",")) for line in lines[1:]]
            x = [item.split(' ') for item in x]
        return x


def evaluate(outputs, labels):
    """
    分析结果
    :param outputs: 模型的输出
    :param labels: 数据集的标签
    :return: 正确的数目
    """
    outputs[outputs >= 0.5] = 1
    outputs[outputs < 0.5] = 0
    corrects = torch.sum(torch.eq(outputs, labels)).item()

    return corrects
