import numpy as np
import pandas as pd
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import models
import math
import time

class SiameseNetworkDataset(Dataset):
    
        def __init__(self,imageFolderDataset,transform=None,should_invert=True):
            self.imageFolderDataset = imageFolderDataset    
            self.transform = transform
            self.should_invert = should_invert
        
        def __getitem__(self,index):
            img0_tuple = random.choice(self.imageFolderDataset.imgs)
            #we need to make sure approx 50% of images are in the same class
            should_get_same_class = random.randint(0,1)
            
            if should_get_same_class:
                check_label='same'
                while True:
                    #keep looping till the same class image is found
                    img1_tuple = random.choice(self.imageFolderDataset.imgs) 
                    if img0_tuple[1] == img1_tuple[1]:
                        break
            else:
                check_label='different'
                img1_tuple = random.choice(self.imageFolderDataset.imgs)

            img0 = Image.open(img0_tuple[0])
            img1 = Image.open(img1_tuple[0])
            img0 = img0.convert("L")
            img1 = img1.convert("L")       
                         

            if self.transform is not None:
                img0 = self.transform(img0)
                img1 = self.transform(img1)  

            return img0, img1 , torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])],dtype=np.float32))
    
        def __len__(self):
            return len(self.imageFolderDataset.imgs)



class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()          

        self.conv = nn.Sequential(            
            nn.Conv2d(1, 64, 3),             
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3),             
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), 

            nn.Conv2d(128, 256, 3),             
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, 3),             
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
            
            
        )
        self.liner = nn.Sequential(
            nn.Linear(512*1*1, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Linear(128, 64)
        )   
       
        self.flatten=nn.Flatten()
        
    def forward_one(self, x):        
        x = self.conv(x)        
        x=self.flatten(x)        
        x = self.liner(x)
        
        return x
    

    def forward(self, x1, x2):     
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)

        return out1,out2
        




class Config():
        training_dir = "./fer2013/train/"
        testing_dir = "./fer2013/validation/"
        train_batch_size = 256
        test_batch_size = 256

        train_number_epochs = 5
        test_number_epochs = 5
      
        learning_rate = 0.0005
        workers=4


def accuracy(output, target):
    Y = target.byte()   # a Tensor of 0s and 1s
    pred_y = output >= 0.5  # a Tensor of 0s and 1s
    num_correct = (Y==pred_y).sum()  # a Tensor
    acc = (num_correct.item()/ len(target))  # scalar
    return acc 


def fit(model, optim, criterion, train_loader, val_loader, num_epochs):
    train_losses = []
    val_losses = []
    correct= []
    probs = []
    device = torch.device("cuda:0")
   
    for i in range(0,num_epochs):
        start_epoch = time()
        model.train()
        
        tlosses = []
        vlosses = []
        for input1, input2, labels in train_loader:
            
            input1 = input1.to(device)
            input2 = input2.to(device)
            labels = labels.to(device).float()
            optim.zero_grad()
            out1,out2 = model(input1, input2)
            labels[labels==0]=-1
            loss = criterion(out1,out2,labels)
            
            
            loss.backward()
            optim.step()
            l = loss.cpu().data.numpy()
            
            tlosses.append(l)
            print("Epoch {}\n batch training loss {}\n".format(i,l))
            
        
        model.eval()
       
        acc = []
        correct= []
        with torch.no_grad():
            for input1, input2, labels in val_loader:  
                input1 = input1.to(device)
                input2 = input2.to(device)
                labels = labels.to(device).float()
                
                pred1, pred2 = model(input1, input2)
                labels[labels==0]=-1
                loss = criterion(pred1, pred2, labels)
                l = loss.cpu().data.numpy()
                
                vlosses.append(l)

                pdist = nn.PairwiseDistance()
                pred = pdist(pred1, pred2)
                print(pred)
                prob = torch.sigmoid(pred)
                
                print(prob)
                print(labels.cpu())
                probs.append(prob.cpu().data.numpy())
                acc.append(accuracy(prob.cpu(), labels.cpu()))
                correct.append(labels.cpu().data.numpy())
                print("Epoch {}\n batch validation loss {}\n".format(i,l))
                
        duration = time() - start_epoch
        tloss = np.array(tlosses).mean()
        vloss = np.array(vlosses).mean()
        graphs=[[i, tloss], [i, vloss]]
        train_losses.append(tloss)
        val_losses.append(vloss)

        print(acc)
        acc = np.array(acc).mean()
        print("____________________________________________")
        print('validation_loss: {}\n'.format(vloss))
        print('training_loss: {}\n'.format(tloss))
        print('accuracy: {}\n'.format(acc))
        print('epochs {}\n'.format(i))    
        print("____________________________________________")
    make_graphs(train_losses,val_losses)

   

def make_graphs(t,v):
    plt.figure(figsize=(10, 5))
    
    y1=range(0,len(t))
    y2=range(0,len(v))

    plt.plot(y1,t,label='train') 
    plt.plot(y2,v,label='val') 

    plt.xlabel('x - axis')

    plt.ylabel('y - axis')
    plt.legend()
    plt.show()


if __name__ == "__main__":



    seed=0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.cuda.empty_cache()

    folder_train = dset.ImageFolder(root=Config.training_dir)

    folder_test = dset.ImageFolder(root=Config.testing_dir)

    dataset_train = SiameseNetworkDataset(imageFolderDataset=folder_train,
                                        transform=transforms.Compose([
                                        transforms.Grayscale(num_output_channels=1), transforms.Resize((48,48)),
                                                                      transforms.ToTensor(), 
                                                                      transforms.Normalize((0.5081), (0.2552), inplace=True)
                                                                      ])
                                       ,should_invert=False)


    dataset_test = SiameseNetworkDataset(imageFolderDataset=folder_test,
                                        transform=transforms.Compose([
                                        transforms.Grayscale(num_output_channels=1), transforms.Resize((48,48)),
                                                                      transforms.ToTensor(), 
                                                                      transforms.Normalize((0.5081), (0.2552), inplace=True)
                                                                      ])
                                       ,should_invert=False)



    train_dataloader = DataLoader(dataset_train,
                        shuffle=True,     
                        num_workers=Config.workers,
                        batch_size=Config.train_batch_size)

    test_dataloader = DataLoader(dataset_test,
                        shuffle=True,                         
                        num_workers=Config.workers,
                        batch_size=Config.test_batch_size)

    net = SiameseNetwork().cuda()
   
    criterion = nn.MarginRankingLoss()
    optimizer = optim.Adam(net.parameters(),lr = Config.learning_rate)
   
    fit(net, optimizer, criterion, train_dataloader, test_dataloader, Config.train_number_epochs)
    PATH = "cosian.pt"
    torch.save(net.state_dict(), PATH)