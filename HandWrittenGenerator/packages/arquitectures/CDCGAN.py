import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import HTML
from IPython.display import clear_output
import matplotlib.animation as animation
import os

from packages.arquitectures.networks.Discriminator import DiscriminatorMannager
from packages.arquitectures.networks.Generator import GeneratorMannager

def weights_init(net):
    classname = net.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(net.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(net.weight.data, 1.0, 0.02)
        nn.init.constant_(net.bias.data, 0)

def getOptimizers(params, Disc = None, Gen = None):
        optimizerG, optimizerD = None, None

        optimizer = params['optimizer']
        lr = params["learning_rate"]

        if optimizer == "sgd":
            momentun = params["momentun"]
            optimizerD = torch.optim.SGD(Disc.parameters(), lr=lr, momentum=momentun)
            optimizerG = torch.optim.SGD(Gen.parameters(), lr=lr, momentum=momentun)
        elif optimizer == "adam":
            betas_min = params["betas_min"]
            betas_max = params["betas_max"]
            optimizerD = torch.optim.Adam(Disc.parameters(), lr=lr, betas=(betas_min, betas_max))
            optimizerG = torch.optim.Adam(Gen.parameters(), lr=lr, betas=(betas_min, betas_max))

        return optimizerD, optimizerG
    

class CDCGAN(nn.Module):
    def __init__(self, isDebugMode:  bool = False, root: str = '..'):
        super(CDCGAN, self).__init__()  
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.device("mps") else "cpu")     
        self.Discriminator, self.Generator = None, None
        self.optimizerD, self.optimizerG  = None, None
        self.onehot, self.fill = None, None
        self.isDebugMode = isDebugMode
        self.ROOT = root

    def setDataLoader(self, dataloader = None, name = None):
        self.DATASET_NAME = name
        self.DATALOADER = dataloader

    def setImageParams(self, batchSize, channels, size, labels, labels_count):
        self.BATCH_SIZE = batchSize
        self.IMG_CHANNELS = channels
        self.IMG_SIZE = size
        self.LABELS = labels
        self.LABELS_COUNT = labels_count
    
    def setFixedSpace(self, z_size, fixed_z_size, fixed_y_size):
        self.Z_SIZE = z_size
        self.FIXED_X = torch.randn(fixed_z_size, z_size).to(self.DEVICE) 
        self.FIXED_Y = torch.tensor(self.LABELS.tolist()*fixed_y_size).type(torch.LongTensor)

    def setupModels(self, params):
        if self.DATALOADER is None: self.errors(1)

        # Initialize models and weights
        self.Discriminator = DiscriminatorMannager(self.IMG_SIZE, self.IMG_CHANNELS, self.LABELS_COUNT, self.isDebugMode).get().to(self.DEVICE)
        self.Generator = GeneratorMannager(self.Z_SIZE, self.LABELS_COUNT, self.IMG_SIZE, self.isDebugMode).get().to(self.DEVICE)
        self.Discriminator.apply(weights_init) 
        self.Generator.apply(weights_init)

        #Optimizers:
        self.optimizerD, self.optimizerG = getOptimizers(params, self.Discriminator, self.Generator)

        # set Loss Function
        self.criterion = nn.BCELoss()
        
        # support vectors
        self.ONES = torch.ones((self.BATCH_SIZE,1)).to(self.DEVICE)
        self.CEROS = torch.zeros((self.BATCH_SIZE,1)).to(self.DEVICE)
        self.ONEHOT = torch.zeros(self.LABELS_COUNT, self.LABELS_COUNT).scatter_(1, self.LABELS.view(self.LABELS_COUNT,1), 1) 
        self.FILL  = torch.zeros([self.LABELS_COUNT, self.LABELS_COUNT, self.IMG_SIZE, self.IMG_SIZE])
        for i in range(self.LABELS_COUNT): 
            self.FILL[i, i, :, :] = 1
    
    def trainStepDis(self, real_image, real_label):      # Update D network: maximize log(D(x)) + log(1 - D(G(z)))
		#1. Make prediction base on real img
        real_x = real_image.to(self.DEVICE)
        real_y = self.reshapeLabel(real_label) 				  
        real_pred = self.Discriminator(real_x, real_y)
        real_d_loss = self.criterion(real_pred, self.ONES)      

		#2. Make prediction base on fake img
        self.fake_image, fake_label = self.generateImage()
        fake_x = self.fake_image.detach()
        self.fake_y = self.reshapeLabel(fake_label)          
        fake_pred = self.Discriminator(fake_x, self.fake_y)     	
        fake_d_loss = self.criterion(fake_pred, self.CEROS)    
		
        Dis_loss = real_d_loss + fake_d_loss                    
        
        dis_total_loss = Dis_loss.item()									
        dis_real_loss = real_pred.mean().item()

        self.Discriminator.zero_grad()	                        # zero accumalted grads
        Dis_loss.backward()				                        # do backward pass
        self.optimizerD.step()			                        # update discriminator model

        return dis_total_loss, dis_real_loss
    
    def trainStepGen(self):         # Update G network: maximize log(D(G(z)))
        z_out = self.Discriminator(self.fake_image, self.fake_y)	# As we done one step of discriminator, again calculate D(G(z))
        G_loss = self.criterion(z_out, self.ONES)		# loss log(D(G(z)))

        gen_loss = G_loss
        dis_z_loss = z_out.mean().item()

        self.Generator.zero_grad()						# zero accumalted grads
        G_loss.backward()								# do backward pass
        self.optimizerG.step()							# update generator model
        return dis_z_loss, gen_loss

    #Aux funtions
    def oneHotEncode(self, data):
        # use onehot encoding method to encode categorical data
        # notice that this kind of encoding is used to prevent model asuntions about the order of the categories, 
        # it creates a binary table 
        return self.ONEHOT[data].to(self.DEVICE)
   
    def reshapeLabel(self, label): 
        # The categorical data must have a specific form before been feed to the models
        return self.FILL[label].to(self.DEVICE)
    
    def errors(self, num):
        if num == 1:
            print("There is not dataloader defined")
        elif num == 2:
            print("There is no name define for dataloader")
        elif num == 3:
            print("No models found")

    def generateImage(self, letter_index = None):
        # create latent vector z from normal distribution
        z = torch.randn(self.BATCH_SIZE, self.Z_SIZE).to(self.DEVICE) 
        
        fake_label = None
        if letter_index == None:                        				  
            # create random y_labels for generator
            fake_label = (torch.rand(self.BATCH_SIZE, 1)*self.LABELS_COUNT).type(torch.LongTensor).squeeze() 
        else: 
            fake_label = torch.cat((torch.tensor([letter_index]), torch.randint(0, self.LABELS_COUNT, (self.BATCH_SIZE - 1,))))

        fake_image = self.Generator(z, self.oneHotEncode(fake_label))									
        return fake_image, fake_label
    
   
   
    def createSamplesTable(self, fixed_y_size, epoch, totalEpoch, explorationName, experimentName):
        self.Generator.eval()
        with torch.no_grad():
            path =  f"{self.ROOT}/results/{explorationName}/"
            if not os.path.exists(path): os.mkdir(path)

            path = f"{path}{experimentName}/"
            if not os.path.exists(path): os.mkdir(path)

            path =  f"{path}/Images/"
            if not os.path.exists(path): os.mkdir(path)
            
            newLabel = self.oneHotEncode(self.FIXED_Y)
            newImage = self.Generator(self.FIXED_X, newLabel).cpu()
            torchvision.utils.save_image(newImage, path+f"epoch_{totalEpoch-epoch+1}.jpg", nrow=fixed_y_size, padding=0, normalize=True)

        self.Generator.train()
        return newImage


    def saveModel(self,  explorationName, experimentName, modelName):
        path =  f"{self.ROOT}/results/{explorationName}/"
        if not os.path.exists(path): os.mkdir(path)

        path = f"{path}{experimentName}/"
        if not os.path.exists(path): os.mkdir(path)

        path =  f"{path}/models/"
        if not os.path.exists(path): os.mkdir(path)

        torch.save({
            'Dis_dict': self.Discriminator.state_dict(),
            'Gen_dict': self.Generator.state_dict(),
            'Dis_optim_state': self.optimizerD.state_dict(),
            'Gen_optim_state': self.optimizerG.state_dict(),
        }, f"{path}{modelName}.pt")

    def loadModel(self, explorationName, experimentName, modelName = 'GAN'):
        # setupModels must be called first
        PATH = f"{self.ROOT}/results/{explorationName}/{experimentName}/models/{modelName}.pt"
        checkpoint = torch.load(PATH, map_location=torch.device('cpu'))

        self.Discriminator.load_state_dict(checkpoint['Dis_dict'])
        self.Generator.load_state_dict(checkpoint['Gen_dict'])
        self.optimizerD.load_state_dict(checkpoint['Dis_optim_state'])
        self.optimizerG.load_state_dict(checkpoint['Gen_optim_state'])

    def train(self, NUM_EPOCH=20, KSTEPS=1, SMAPLESTEPS=1, explorationName='GAN', experimentName='_testTraining'):
        step = 0
        D_losses, G_losses = [], []
        Dx_values, DGz_values = [], []

        for epoch in range(NUM_EPOCH):
            dis_total_loss, dis_real_loss = 0, 0
            gen_loss, dis_z_loss = 0, 0
            epoch_D_losses, epoch_G_losses = [], []
            epoch_Dx, epoch_DGz = [], []

            for real_image, real_label in self.DATALOADER:
                step += 1
                
                dis_total_loss, dis_real_loss = self.trainStepDis(real_image, real_label)
                epoch_D_losses.append(dis_total_loss)
                epoch_Dx.append(dis_real_loss)

                if step % KSTEPS == 0:
                    dis_z_loss, gen_loss = self.trainStepGen()
                    epoch_DGz.append(dis_z_loss)
                    epoch_G_losses.append(gen_loss)

            else:
                D_losses.append(sum(epoch_D_losses)/len(epoch_D_losses))
                G_losses.append(sum(epoch_G_losses)/len(epoch_G_losses))
                Dx_values.append(sum(epoch_Dx)/len(epoch_Dx))
                DGz_values.append(sum(epoch_DGz)/len(epoch_DGz))
                
                print(f" Epoch: {epoch+1}/{NUM_EPOCH} |" 
                    + f" D_loss = {D_losses[-1]:.3f}, G_loss = {G_losses[-1]:.3f} |"
                    + f" D(real) = {Dx_values[-1]:.3f}, D(fake) = {DGz_values[-1]:.3f}")

                if(epoch % SMAPLESTEPS == 0): 
                    self.createSamplesTable(self.LABELS_COUNT, epoch, NUM_EPOCH, explorationName, experimentName)

        return D_losses, G_losses, Dx_values, DGz_values