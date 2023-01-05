本项目配套讲解博客：[文本分类（LSTM+PyTorch）](https://blog.csdn.net/Bat_Reality/article/details/128509050?spm=1001.2014.3001.5502)

### 一、传统方法的基本步骤
1. 预处理：首先进行分词，然后是除去停用词；
2. 将文本表示成向量，常用的就是文本表示向量空间模型；
3. 进行特征选择，这里的特征就是词语，去掉一些对于分类帮助不大的特征。常用的特征选择的方法是词频过滤，互信息，信息增益，卡方检验等；
4. 接下来就是构造分类器，在文本分类中常用的分类器一般是SVM，朴素贝叶斯等；
5. 训练分类器，后面只要来一个文本，进行文本表示和特征选择后，就可以得到文本的类别。

### 二、深度学习与传统方法的区别
传统做法主要问题的文本表示是高纬度高稀疏的，特征表达能力很弱；此外需要人工进行特征工程，成本很高。
利用深度学习做文本分类，首先还是要进行分词，这是做中文语料必不可少的一步，这里的分词使用的jieba分词。 词语的表示不用one-hot编码，而是使用词向量(word embedding)，现在最常用的词向量的分布式表示就是word2vec，这样的分布式表示，既降低了维度，也体现了语义信息。使用深度学习进行文本分类，不需要进行特征选择这一步，因为深度学习具有自动学习特征的能力。

### 三、LSTM文本分类
先将句子进行分词，然后将每个词语表示为词向量，再将词向量按顺序送进LSTM，最后LSTM的输出就是这段话的表示，而且能够包含句子的时序信息。
![LSTM文本分类流程图](https://img-blog.csdnimg.cn/d7618d3cb1ce47459b3250a494c30973.png)
#####  1. 处理数据集
```python
# utils.py

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
```
### 2. word2vec词向量模型
依据原始数据提取了data与labels之后，还需要依据文本生成词向量，即建立单词与向量的一一对应关系，同时建立词汇之间的语义联系。
```python
# word2vec.py

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
```
### 3. 数据预处理
数据预处理就是按照词向量对样本中的数据生成句子的矩阵，因此这里需要统一句子的长度，生成数据集并实现数据集的封装。实现代码由两部分构成，首先生成对应矩阵，然后封装dataset便于使用dataloader加载数据。
```python
# pre_processing.py
import torch
from gensim.models import Word2Vec


class DataProcess:
    def __init__(self, sentences, sen_len, w2v_path="./data/word2vec.model"):
        self.sentences = sentences  # 句子列表
        self.sen_len = sen_len      # 句子的最大长度
        self.w2v_path = w2v_path    # word2vec模型路径
        self.index2word = []        # 实现index到word转换
        self.word2index = {}        # 实现word到index转换
        self.embedding_matrix = []

        # load word2vec.model
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size

    def make_embedding(self):
        # 为model里面的单词构造word2index, index2word 和 embedding
        for i, word in enumerate(self.embedding.wv.vocab):
            print('get word #{}'.format(i+1), end='\r')
            self.word2index[word] = len(self.word2index)
            self.index2word.append(word)
            self.embedding_matrix.append(self.embedding[word])
        self.embedding_matrix = torch.tensor(self.embedding_matrix)

        # 將"<PAD>"和"<UNK>"加进embedding里面
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))

        return self.embedding_matrix

    def add_embedding(self, word):
        # 将新词添加进embedding中
        vector = torch.empty(1, self.embedding_dim)
        torch.nn.init.uniform_(vector)
        self.word2index[word] = len(self.word2index)
        self.index2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def sentence_word2idx(self):
        sentence_list = []
        for i, sentence in enumerate(self.sentences):
            # 将句子中的单词表示成index
            sentence_index = []
            for word in sentence:
                if word in self.word2index.keys():
                    # 如果单词在字典中则直接读取index
                    sentence_index.append(self.word2index[word])
                else:
                    # 否则赋予<UNK>
                    sentence_index.append(self.word2index["<UNK>"])

            # 统一句子长度
            sentence_index = self.pad_sequence(sentence_index)
            sentence_list.append(sentence_index)

        return torch.LongTensor(sentence_list)

    def pad_sequence(self, sentence):
        # 统一句子长度
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2index["<PAD>"])
        assert len(sentence) == self.sen_len

        return sentence

    def labels2tensor(self, y):
        y = [int(label) for label in y]

        return torch.LongTensor(y)
```

### 4. 封装数据集
```python
# data_loader.py
from torch.utils import data


class TwitterDataset(data.Dataset):
    def __init__(self, x, y):
        self.data = x
        self.label = y

    def __getitem__(self, index):
        if self.label is None:
            return self.data[index]

        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)
```
### 5. 构造模型
模型包括三个部分：embedding层、LSTM层、全连接层。

embedding层可以理解为：一个保存了固定字典和大小的简单查找表。这个模块常用来保存词嵌入和用下标检索它们。模块的输入是一个下标的列表，输出是对应的词嵌入。
LSTM层：
> 1. input_size: 输入特征维数，即每一行输入元素的个数
> 2. hidden_size: 隐藏层状态的维数，即隐藏层节点的个数，这个和单层感知器的结构是类似的。
> 3. num_layers: LSTM 堆叠的层数，默认值是1层，如果设置为2，第二个LSTM接收第一个LSTM的计算结果。
> 4. batch_first: 输入输出的第一维是否为 batch_size，默认值 False。
> 5. dropout: 默认值0。是否在除最后一个 RNN 层外的其他 RNN 层后面加 dropout 层。
> 6. bidirectional: 是否是双向 RNN，默认为：false，若为 true，则：num_directions=2，否则为1。

全连接层一般适用于分类。

```python
# model.py
import torch


class LSTMModel(torch.nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, requires_grad=True):
        super(LSTMModel, self).__init__()
        # 这是一个保存了固定字典和大小的简单查找表。这个模块常用来保存词嵌入和用下标检索它们。模块的输入是一个下标的列表，输出是对应的词嵌入。
        # embedding: (嵌入字典的大小, 每个嵌入向量的维度)
        self.embedding = torch.nn.Embedding(embedding.size(0), embedding.size(1))
        # 将一个不可训练的类型为Tensor的参数转化为可训练的类型为parameter的参数，并将这个参数绑定到module里面，成为module中可训练的参数。
        self.embedding.weight = torch.nn.Parameter(embedding, requires_grad=requires_grad)

        self.LSTM = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, inputs):
        inputs = self.embedding(inputs)
        x, _ = self.LSTM(inputs, None)
        # x.shape = (batch_size, seq_len, hidden_size)
        # 取用 LSTM 最后一个的 hidden state
        x = x[:, -1, :]
        x = self.classifier(x)

        return x
```

### 6. 训练模型
模型训练由两部分组成：train、validate
```python
# train.py
import torch
from torch import nn
import torch.optim as optim
from utils import evaluate
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(train_loader, model, criterion, optimizer, epoch):
    # 將 model 的模式设定为 train，这样 optimizer 就可以更新 model 的参数
    model.train()

    train_len = len(train_loader)
    total_loss, total_acc = 0, 0

    for i, (inputs, labels) in enumerate(train_loader):
        # 1. 放到GPU上
        inputs = inputs.to(device, dtype=torch.long)
        labels = labels.to(device, dtype=torch.float)  # 类型为float
        # 2. 清空梯度
        optimizer.zero_grad()
        # 3. 计算输出
        outputs = model(inputs)
        outputs = outputs.squeeze()  # 去掉最外面的 dimension
        # 4. 计算损失
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        # 5.预测结果
        correct = evaluate(outputs, labels)
        total_acc += (correct / batch_size)
        # 6. 反向传播
        loss.backward()
        # 7. 更新梯度
        optimizer.step()
        print('[ Epoch{}: {}/{} ] loss:{:.3f} acc:{:.3f} '.
              format(epoch + 1, i + 1, batch_size, loss.item(), correct * 100 / batch_size), end='\r')
    print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(total_loss / train_len, total_acc / train_len * 100))


def validate(val_loader, model, criterion):
    model.eval()  # 將 model 的模式设定为 eval，固定model的参数

    val_len = len(val_loader)

    with torch.no_grad():
        total_loss, total_acc = 0, 0
        for i, (inputs, labels) in enumerate(val_loader):
            # 1. 放到GPU上
            inputs = inputs.to(device, dtype=torch.long)
            labels = labels.to(device, dtype=torch.float)
            # 2. 计算输出
            outputs = model(inputs)
            outputs = outputs.squeeze()
            # 3. 计算损失
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            # 4. 预测结果
            correct = evaluate(outputs, labels)
            total_acc += (correct / batch_size)
        print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(total_loss / val_len, total_acc / val_len * 100))
    print('-----------------------------------------------')

    return total_acc / val_len * 100
```

### 7. 主函数
通过主函数将前面各部分组织起来

```python
# train.py
# 各数据集的路径
path_prefix = './data'
train_with_label = os.path.join(path_prefix, 'training_label.txt')
train_no_label = os.path.join(path_prefix, 'training_nolabel.txt')

# word2vec模型文件路径
w2v_path = os.path.join(path_prefix, 'word2vec.model')

# 定义句子长度、是否固定 embedding、batch 大小、定义训练次数 epoch、learning rate 的值、model 的保存路径
requires_grad = False
sen_len = 20
model_dir = os.path.join(path_prefix, 'model/')
batch_size = 32
epochs = 10
lr = 0.001


def main():
    # load data
    data_x, data_y = read_file(train_with_label)

    # data pre_processing
    preprocess = DataPreprocess(data_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding()
    data_x = preprocess.sentence_word2idx()
    data_y = preprocess.labels2tensor(data_y)

    # split data
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3, random_state=5)
    # 构造Dataset
    train_dataset = TwitterDataset(x_train, y_train)
    val_dataset = TwitterDataset(x_train, y_train)

    # preparing the training loader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    print('Training loader prepared.')
    # preparing the validation loader
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    print('Validation loader prepared.')

    # load model
    model = LSTMModel(
        embedding,
        embedding_dim=300,
        hidden_dim=128,
        num_layers=1,
        dropout=0.5,
        requires_grad=requires_grad
    ).to(device)

    # 返回model中的参数的总数目
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\nstart training, parameter total:{}, trainable:{}\n'.format(total, trainable))

    # loss function is binary cross entropy loss, 常见的二分类损失函数
    criterion = nn.BCELoss()
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.

    # run epochs
    for epoch in range(epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        total_acc = validate(val_loader, model, criterion)

        if total_acc > best_acc:
            # 如果 validation 的结果好于之前所有的结果，就把当下的模型保存
            best_acc = total_acc
            # torch.save(model, "{}/val_acc_{:.3f}.model".format(model_dir,total_acc/v_batch*100))
            torch.save(model, "{}/ckpt.model".format(model_dir))
            print('saving model with acc {:.3f}'.format(total_acc / len(val_loader) * 100))
```

### 8. 预测数据
在存储训练好的模型后，我们对无标签的测试集进行预测，并上传到Kaggle上。

```python
# test.py
import torch
from torch.utils.data import DataLoader
import os
import pandas as pd
from data_loader import TwitterDataset
from utils import read_file
from pre_processing import DataPreprocess
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# 各数据集的路径
path_prefix = './data'
test_data = os.path.join(path_prefix, 'testing_data.txt')

# word2vec模型文件路径
w2v_path = os.path.join(path_prefix, 'word2vec.model')

# 定义句子长度、是否固定 embedding、batch 大小、定义训练次数 epoch、learning rate 的值、model 的保存路径
requires_grad = False
sen_len = 20
model_dir = './data'
batch_size = 256

# 读取测试文件
test_x = read_file(test_data)
# 测试集预处理
prepocess = DataPreprocess(test_x, sen_len, w2v_path=w2v_path)
embedding = prepocess.make_embedding()
data_x = prepocess.sentence_word2idx()
# 构造Dataset
test_dataset = TwitterDataset(data_x, None)
# 获取dataloader
test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
print('Testing loader prepared.')

# load model
model = torch.load(os.path.join(model_dir, 'ckpt.model'))

# 模型预测
model.eval()
ret_output = []
with torch.no_grad():
    for i, inputs in enumerate(test_loader):
        # 1. 放到GPU上
        inputs = inputs.to(device, dtype=torch.long)
        # 2. 计算输出
        outputs = model(inputs)
        outputs = outputs.squeeze()
        # 3. 预测结果
        outputs[outputs >= 0.5] = 1
        outputs[outputs < 0.5] = 0
        ret_output += outputs.int().tolist()

# 写到.csv文件中
tmp = pd.DataFrame({"id": [str(i) for i in range(len(test_x))], "label": ret_output})
print("Saving csv ...")
tmp.to_csv(os.path.join(path_prefix, 'predict.csv'), index=False)
print("Predicting finished.")
```

### 9. 完整代码
[文本分类（LSTM+PyTorch）](https://github.com/Bat-Reality/PyTorch_exp/tree/main/%E6%96%87%E6%9C%AC%E5%88%86%E7%B1%BB/LSTM)
[数据集下载链接](https://www.kaggle.com/competitions/ml2020spring-hw4/data)

### 参考文献
[LSTM文本分类实战](https://maimai.cn/article/detail?fid=252357321&efid=ht0LWs1-WErzQBIzCWJwMA)
[LSTM算法实现文本分类（PyTorch）](https://www.modb.pro/db/410499)
[Pytorch实战__LSTM做文本分类](https://blog.csdn.net/hello_JeremyWang/article/details/121071281)
[ChineseNlpCorpus](https://github.com/SophonPlus/ChineseNlpCorpus)
