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
