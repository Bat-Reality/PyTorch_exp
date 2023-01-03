import torch
from gensim.models import Word2Vec


class DataPreprocess:
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
