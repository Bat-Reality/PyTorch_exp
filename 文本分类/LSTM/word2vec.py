from gensim.models import word2vec
import os
from utils import read_file


def train_word2vec(x):
    """
    LineSentence(inp)：格式简单：一句话=一行; 单词已经过预处理并被空格分隔。
    size：是每个词的向量维度；
    window：是词向量训练时的上下文扫描窗口大小，窗口为5就是考虑前5个词和后5个词；
    min-count：设置最低频率，默认是5，如果一个词语在文档中出现的次数小于5，那么就会丢弃；
    sg ({0, 1}, optional) – 模型的训练算法: 1: skip-gram; 0: CBOW
    iter (int, optional) – 迭代次数，默认为5
    :param x: 处理好的数据集
    :return: 训练好的模型
    """
    return word2vec.Word2Vec(x, size=300, window=5, min_count=5, sg=1, iter=10)


if __name__ == '__main__':
    data_dir = './data'

    print("loading training data ...")
    train_x, y = read_file(os.path.join(data_dir, 'training_label.txt'))
    train_no_label = read_file(os.path.join(data_dir, 'training_nolabel.txt'))
    print("loading test data...")
    test_data = read_file(os.path.join(data_dir, 'testing_data.txt'))

    print("training text data and transforming to vectors by skip-gram...")
    model = train_word2vec(train_x + train_no_label + test_data)
    print("saving model...")
    model.save(os.path.join(data_dir, 'word2vec.model'))
