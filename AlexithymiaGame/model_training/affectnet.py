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
from time import time 
import torch
import scipy.misc as m
import os
import csv
import numpy as np
from torch.utils import data
import cv2
from PIL import ImageFile
import random




#function to not include batches if batch has None
def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


#custom dataset for Affectnet
class SiameseAffectnetDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.lenth=len(self.img_labels)

    def __len__(self):        
        return self.lenth

    def __getitem__(self, idx):        
        label = None
        img1 = None
        img2 = None
        idx1=idx
        idx2=None
        img_path1 = os.path.join(self.img_dir, self.img_labels.iloc[idx1, 0])
        img_path2 = None
        label1 = self.img_labels.iloc[idx1, 6]
        label2 = None

        #we need to make sure approx 50% of images are in the same class
        #to affect every second pair in the set
        if idx % 2 != 0:            
            idx2=random.randint(0,self.lenth-1)
            label2=self.img_labels.iloc[idx2, 6]            
            img_path2 = os.path.join(self.img_dir, self.img_labels.iloc[idx2, 0])
            if label1 != label2:
                label=1.0
            else:
                label=0.0

        else:
            label=0.0
            idx2=random.randint(0,self.lenth-1)
            label2=self.img_labels.iloc[idx2, 6]

            #making sure we get pair of images of the same label
            #keep looping till the same class image is found
            while label1 != label2:
                idx2=random.randint(0,self.lenth-1)
                label2=self.img_labels.iloc[idx2, 6]
            img_path2 = os.path.join(self.img_dir, self.img_labels.iloc[idx2, 0])

        #try catch for the corrupted images
        try:
            img1 = Image.open(img_path1)
            img2 = Image.open(img_path2)
        except:
            print("Corrupion\n")            
            return None    
        

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        if self.target_transform:
            label = self.target_transform(label)
        
        return  img1,img2,label



   
        

class SiameseNetworkABS(nn.Module):
    def __init__(self):
        super(SiameseNetwork_ABS, self).__init__()          

        self.conv = nn.Sequential(            
            nn.Conv2d(3, 64, 3, 1),             
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1),             
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2), 

            nn.Conv2d(128, 256, 3 , 1),             
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, 3 , 1),             
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),            

            nn.Conv2d(256, 512, 3, 1),             
            nn.BatchNorm2d(512),
            nn.ReLU(),            

            nn.Conv2d(512, 512, 3 , 1),             
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.liner = nn.Sequential(
            nn.Linear(512*4*4, 4096),
            nn.BatchNorm1d(4096),
            nn.ReLU(),

            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(),

            nn.Linear(2048, 1024)
        )

        self.out = nn.Linear(1024, 1)
        self.flatten=nn.Flatten()

        
    def forward_one(self, x):        
        x = self.conv(x)        
        x=self.flatten(x)
        x = self.liner(x)        
        return x

    def forward(self, x1, x2):     
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        
        dis = torch.abs(out1 - out2)        
        out = self.out(dis)        

        return out



#function for accuracy calculation
def accuracy(output, target):
    Y = target.byte()   # a Tensor of 0s and 1s
    pred_y = output >= 0.5  # a Tensor of 0s and 1s
    num_correct = (Y==pred_y).sum()  # a Tensor
    acc = (num_correct.item()/ len(target))  # scalar
    return acc 

#Config class for settings
class Config():
        training_dir = "./fer2013/train/"
        testing_dir = "./fer2013/validation/"

        train_batch_size = 32        
        test_batch_size = 32

        train_number_epochs = 5
        test_number_epochs = 5

        image_size=120
        
        learning_rate = 0.0005
        
        workers=4

#network trainer
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
            labels = labels.to(device)
            
            optim.zero_grad()
            out = model(input1, input2)
            
            
            #out=out.squeeze(0)#
            labels = labels.unsqueeze(1)
            
            loss = criterion(out, labels)
            
            loss.backward()
            optim.step()
            l = loss.cpu().data.numpy()
            
            tlosses.append(l)
            print("Epoch {}\n batch training loss {}\n".format(i,l))
            
        
        model.eval()       
        acc = []
        correct = []
        with torch.no_grad():
            for input1, input2, labels in val_loader:  
                input1 = input1.to(device)
                input2 = input2.to(device)
                labels = labels.to(device).float()
                pred = model(input1, input2)
               
                #pred=pred.squeeze(0)#
                labels = labels.unsqueeze(1)
                loss = criterion(pred, labels)
                
                
                l = loss.cpu().data.numpy()
                
                vlosses.append(l)
               
                prob = torch.sigmoid(pred)
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

        
        acc = np.array(acc).mean()
        print("____________________________________________")
        print('validation_loss: {}\n'.format(vloss))
        print('training_loss: {}\n'.format(tloss))
        print('accuracy: {}\n'.format(acc))
        print('epochs {}\n'.format(i))    
        print("____________________________________________")
    make_graphs(train_losses,val_losses)        

#function to generate graphs
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


    #setups for results stability
    seed=0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.empty_cache()

    #image transformations fot the dataset
    transform = transforms.Compose([    
                                transforms.Resize((120,120)),
                                transforms.ToTensor(),
                                transforms.Normalize( [149.35457 / 255., 117.06477 / 255., 102.67609 / 255.], [69.18084 / 255., 61.907074 / 255., 60.435623 / 255.])])

    train_dataset = Siamese_Affectnet_Dataset("D:/Datasets/AffectNet/training.csv",'D:/Datasets/AffectNet/Manually_Annotated_Images/',transform)    
    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=Config.train_batch_size, shuffle=True,
                                                       num_workers=Config.workers, collate_fn=collate_fn)

    val_dataset = Siamese_Affectnet_Dataset("D:/Datasets/AffectNet/validation.csv","D:/Datasets/AffectNet/Manually_Annotated_Images/",transform)   
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=Config.test_batch_size, shuffle=False,
                                                     num_workers=Config.workers, collate_fn=collate_fn)


    net = SiameseNetwork_ABS().cuda()
    
    #loss function
    criterion = torch.nn.BCEWithLogitsLoss()    
    #optimizer
    optimizer = optim.Adam(net.parameters(),lr = Config.learning_rate)  
    
    
    fit(net, optimizer, criterion, train_dataloader, val_dataloader, Config.train_number_epochs)

    #path for model saving
    PATH = "affectnetsiam.pt"


    torch.save(net.state_dict(), PATH)
    
