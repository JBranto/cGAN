import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from torchvision.utils import make_grid

from torchvision.datasets import MNIST

class MNISTDataHandler(pl.LightningDataModule):
    def __init__(self, transforms, batch_size: int = 28, workers: int = 2, root: str = "./"):
        super().__init__()
        
        self.ROOT = root 
        self.WORKERS = workers
        self.transform = transforms
        self.batch_size = batch_size

        self.trainDataLoader = None
        self.testDataLoader = None
        self.validDataLoader = None

        mnist_full = MNIST(self.ROOT+'/datasets/data/', train=True, transform=self.transform, download=True)
        self.mnist_train, self.mnist_val = torch.utils.data.random_split(mnist_full, [55000, 5000])

        mnist_test = MNIST(self.ROOT+'/datasets/data/', train=False, transform=self.transform, download=True)
        self.mnist_testA, self.mnist_testB = torch.utils.data.random_split(mnist_test, [5000, 5000])

    def getTrainDataLoader(self):
        if self.trainDataLoader == None:
            self.trainDataLoader = DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True, num_workers=self.WORKERS, drop_last=True)

        return self.trainDataLoader

    def getTestDataLoader(self):
        if self.testDataLoader == None:
            self.testDataLoader = DataLoader(self.mnist_testA, batch_size=self.batch_size, shuffle=True, num_workers=self.WORKERS, drop_last=True)

        return self.testDataLoader
    
    def getValidDataLoader(self):
        if self.validDataLoader == None:
            self.validDataLoader = DataLoader(self.mnist_val, batch_size=self.batch_size, shuffle=True, num_workers=self.WORKERS, drop_last=True)

        return self.validDataLoader
    

    def getUniqueLabels(self):
        dataloader = self.getTrainDataloader()

        dataiter = iter(dataloader)
        _, labels = next(dataiter)
        return torch.unique(labels)
    
    def getLabel(self, label):
        return label

    def displaySamples(self, stage: str  = 'train'):
        dataloader = self.getTrainDataLoader()

        dataiter = iter(dataloader)
        images, labels = next(dataiter)

        img_size = images.shape[2]

        grid_images = make_grid(images, nrow=16, padding=0, normalize=True)

        plt.figure(figsize=(16, self.batch_size/16))
        plt.axis("off")
        plt.title(f"Sample images from {stage} dataset")

        # Creating labels for each image
        label_strings = [label.item() for label in labels]

        # Adding labels to the grid
        for i in range(len(label_strings)):
            plt.text((i % 16) * (img_size + 2) + img_size / 2, (i // 16) * (img_size + 2), label_strings[i], color='red', fontsize=18, ha='center', va='top')
        plt.imshow(grid_images.permute(1, 2, 0)) 
        plt.show()

    