import torch
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader


def get_loader(inputs, masks, labels, train=True, batch_size=128):
    # DataLoader, DataSet, Sampler之间的关系: https://zhuanlan.zhihu.com/p/76893455

    inputs = torch.tensor(inputs)[:100000]
    masks = torch.tensor(masks)[:100000]
    labels = torch.tensor(labels)[:100000]

    # 通过TensorDataset将inputs、masks和labels进行打包
    data = TensorDataset(inputs, masks, labels)
    # 定义sampler
    sampler = RandomSampler(data) if train else SequentialSampler(data)
    # 返回data_laoder
    loader = DataLoader(data, batch_size=batch_size, shuffle=False, sampler=sampler)

    return loader
