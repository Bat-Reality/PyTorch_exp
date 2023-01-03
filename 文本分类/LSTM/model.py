import torch


class LSTMModel(torch.nn.Module):
    def __init__(self, embedding, embedding_dim, hidden_dim, num_layers, dropout=0.5, requires_grad=False):
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
        x, _ = self.LSTM(inputs)
        # x.shape = (batch_size, seq_len, hidden_size)
        # 取用 LSTM 最后一个的 hidden state
        x = x[:, -1, :]
        x = self.classifier(x)

        return x
