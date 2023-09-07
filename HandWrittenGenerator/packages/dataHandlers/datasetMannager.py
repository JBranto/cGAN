import os

import packages.dataHandlers.datasets.LETTERS as ltt
import packages.dataHandlers.datasets.MNIST as mn
import packages.dataHandlers.datasets.FASHION as fa

class datasetMannager():
    def __init__(self, TRANSFORMS, BATCH_SIZE: int, DATASET_NAME: str):
        super(datasetMannager, self).__init__()
        self.TRANSFORMS = TRANSFORMS
        self.BATCH_SIZE = BATCH_SIZE

        currentPath = os.getcwd().split('/')
        self.ROOT = '/'.join(currentPath[:currentPath.index('CDCGAN')+1]) + '/packages/dataHandlers'

        if DATASET_NAME == "MNIST":
            self.data_module = mn.MNISTDataHandler(TRANSFORMS, BATCH_SIZE, workers=2, root=self.ROOT) 
        elif DATASET_NAME == "FASHION":
            self.data_module = fa.FASHIONDataHandler(TRANSFORMS, BATCH_SIZE, workers=2, root=self.ROOT)
        elif DATASET_NAME == "LETTERS":
            self.data_module = ltt.LETTERSDataHandler(TRANSFORMS, BATCH_SIZE, workers = 2, root = self.ROOT)
        else:
            ValueError(f"Please select valid dataset, {DATASET_NAME} is not supported")

    def getDataModule(self):
        return self.data_module

        
        