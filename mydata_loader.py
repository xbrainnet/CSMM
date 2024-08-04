import torch.utils.data as data


class Mydataset(data.Dataset):

    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.length = data.shape[0]

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return self.length
