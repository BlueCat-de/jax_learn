import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader, Dataset
import ipdb

# class MyDataset(Dataset):
#     def __init__(self, dataset):
#         self.dataset = dataset
        

#     def __len__(self):
#         return len(self.dataset)
    
#     def __getitem__(self, index):
#         # ipdb.set_trace()
#         self.t = 1
#         imgs, labels =  self.dataset[index]
#         batch = {
#             'image': imgs,
#             'label': labels
#         }
#         print(1)
#         return 0

def get_datasets(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将图像转换为PyTorch张量
        transforms.Normalize((0.5,), (0.5,))  # 标准化图像数据
    ])

    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
