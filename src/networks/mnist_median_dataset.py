import torch
from torch.utils.data import Dataset


class MNISTMedianDataset(Dataset):

    def __init__(self, images, transform=None):
        self.len = images.shape[0]
        self.images = images
        self.means = [torch.median(images[i].float()) for i in range(len(images))]
        self.transform = transform

    def __getitem__(self, index):
        return self.images[index], self.means[index]

    def __len__(self):
        return self.len
