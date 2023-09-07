import torch
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from torchvision import datasets

class LETTERSDataHandler(pl.LightningDataModule):
    def __init__(self, transforms, batch_size: int = 28, workers: int = 2, root: str = "./"):
        super().__init__()
        self.labels_map = {
            0: "a",1: "b",2: "c",3: "d",4: "e",5: "f",6: "g",7: "h",8: "i",9: "j",
            10: "l",11: "m",12: "n",13: "o",14: "p",15: "q",16: "r",17: "s",18: "t",
            19: "u",20: "v",21: "x",22: "y",23: "z"
        }
        
        self.ROOT = root 
        self.WORKERS = workers
        self.transform = transforms
        self.batch_size = batch_size

        self.trainDataLoader = None
        self.testDataLoader = None
        self.validDataLoader = None

    def getTrainDataLoader(self):
        trn_path = self.ROOT+'/datasets/data/letters/train'
        dataPath = datasets.ImageFolder(trn_path, transform=self.transform)

        if self.trainDataLoader == None:
            self.trainDataLoader = DataLoader(dataPath, batch_size=self.batch_size, shuffle=True, num_workers=self.WORKERS, drop_last=True)

        return self.trainDataLoader

    def getTestDataLoader(self):
        trn_path = self.ROOT+'/datasets/data/letters/test'
        dataPath = datasets.ImageFolder(trn_path, transform=self.transform)

        if self.testDataLoader == None:
            self.testDataLoader = DataLoader(dataPath, batch_size=self.batch_size, shuffle=True, num_workers=self.WORKERS, drop_last=True)

        return self.testDataLoader
    
    def getValidDataLoader(self):
        trn_path = self.ROOT+'/datasets/data/letters/validation'
        dataPath = datasets.ImageFolder(trn_path, transform=self.transform)

        if self.validDataLoader == None:
            self.validDataLoader = DataLoader(dataPath, batch_size=self.batch_size, shuffle=True, num_workers=self.WORKERS, drop_last=True)

        return self.validDataLoader

    def getUniqueLabels(self):
        dataloader = self.getTrainDataLoader()

        dataiter = iter(dataloader)
        _, labels = next(dataiter)
        return torch.unique(labels)
     
    def getLabel(self, label):
        return self.labels_map[label[0].tolist()]
    
    def getLabelIndex(self, letter):
        return next(key for key, value in self.labels_map.items() if value == letter)
    
    def displayImage(self,img,label):
        plt.imshow(img[0].permute(1, 2, 0),cmap='gray')
        plt.show()
        print(f"Label: {self.getLabel(label)}")

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
        label_strings = [self.labels_map[label.item()] for label in labels]

        # Adding labels to the grid
        for i in range(len(label_strings)):
            plt.text((i % 16) * (img_size + 2) + img_size / 2, (i // 16) * (img_size + 2), label_strings[i], color='red', fontsize=18, ha='center', va='top')


        plt.imshow(grid_images.permute(1, 2, 0)) 
        plt.show()

        