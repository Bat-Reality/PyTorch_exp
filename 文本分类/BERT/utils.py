import pandas as pd
import torch


def read_file(path='./data/simplifyweibo_4_moods.csv'):
    # 读取csv文件
    df = pd.read_csv(path)

    # sen_len = 0
    # for line in df['review']:
    #     sen_len += len(line)
    # print(sen_len / len(df))  # 69.48

    return df['review'].to_list(), df['label'].to_list()


def pre_processing(tokenizer, sentence, max_len=126):
    """
    预处理: 统一句子长度
    将每一句转成数字（大于126做截断，小于126做PADDING，加上首尾两个标识，长度总共等于128）
    :param tokenizer: Bert分词器
    :param sentence: 待分词句子
    :param max_len: 句子最大长度
    :return: 分词后的列表
    """
    # 直接截断
    # 编码时: 开头添加[LCS]->101, 结尾添加[SEP]->102, 未知的字或单词变为[UNK]->100
    tokens = tokenizer.encode(sentence[:max_len])
    # 补齐（pad的索引号就是0）
    if len(tokens) < max_len + 2:
        tokens.extend([0] * (max_len + 2 - len(tokens)))

    # sen = 'Bert文本分类模型常见做法为将Bert最后一层输出的第一个token位置（CLS位置）当作句子的表示，后接全连接层进行分类。用标记数据微调BERT参数。'
    # print(tokenizer.tokenize(sen))
    # print(tokenizer.encode(sen))
    # print(tokenizer.convert_ids_to_tokens(tokenizer.encode(sen)))

    return tokens


def attention_masks(tokens_list):
    # 在一个文本中，如果是PAD符号则是0，否则就是1
    masks = []
    for tokens in tokens_list:
        mask = [float(token > 0) for token in tokens]
        masks.append(mask)

    return masks


def evaluate(outputs, labels):
    correct = torch.eq(torch.max(outputs, dim=1)[1], labels.flatten()).float()
    acc = correct.sum().item() / len(labels)

    return acc
