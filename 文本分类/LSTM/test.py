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
