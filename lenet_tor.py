import torch
import torch.nn as nn
import numpy as np
from dataset import Dataset50Loader
import cv2
class LeNet5(nn.Module):
    def __init__(self) :
        super(LeNet5,self).__init__()
        self.allnn=nn.Sequential(
            nn.Conv2d(3,6,5),   #64->60
            nn.Sigmoid(),
            nn.MaxPool2d(2,stride=2),#60->30
            nn.Conv2d(6,16,5),          #30->26
            nn.Sigmoid(),
            nn.MaxPool2d(2,stride=2),   #26->13
            nn.Conv2d(16,120,5), #13->9
        )
        self.fn=nn.Sequential(
            nn.Linear(120*9*9,84),
            nn.Sigmoid(),
            nn.Linear(84,50),
            nn.Softmax(dim=1),
        )

    def forward(self,x):
        x=self.allnn(x)
        x=x.view(-1,120*9*9)
        x=self.fn(x)
        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")    
model=LeNet5().to(device)
model.load_state_dict(torch.load("tor_model_40.pth"))
optimize=torch.optim.Adam(model.parameters(),lr=0.0001)
lossfc=nn.CrossEntropyLoss().to(device)
train_loader=Dataset50Loader('train2.txt')
val_loader=Dataset50Loader('val.txt')
totalt=len(train_loader.dataset)
totalv=len(val_loader.dataset)
for epoch in range(0,50):
    act=0
    losst=[]
    model.train()
    for batch_idx, (img,data) in enumerate(train_loader):
        out=model(torch.Tensor(img.to(torch.float32).to(device)))
        loss=lossfc(out,data.to(device))
        optimize.zero_grad()
        loss.backward()
        optimize.step()
        losst.append(loss.item())
        out=torch.argmax(out,1)
        for j in range(len(data)):
            if out[j]==data[j]:
                act+=1

        if batch_idx%50==0:
            print(f'epoch={epoch} ,batch={batch_idx},loss={loss}')    
    logger = open('tor_train.txt', 'a')
    logger.write('%d %f %f\n'%(epoch,round(sum(losst)/len(losst),3),round(act/totalt,3)))
    logger.close()   
    
    torch.save(model.state_dict(), "tor_model_%d.pth" %(epoch))

    model.eval()
    acv=0
    lossv=[]
    with torch.no_grad():
        for batch_idx, (img,data) in enumerate(val_loader):
            out=model(torch.Tensor(img.to(torch.float32).to(device)))
            loss=lossfc(out,data.to(device))
            lossv.append(loss.item())
            out=torch.argmax(out,1)
            for j in range(len(data)):
                if out[j]==data[j]:
                    acv+=1
    logger = open('tor_val.txt', 'a')
    logger.write('%d %f %f\n'%(epoch,round(sum(lossv)/len(lossv),3),round(acv/totalv,3)))
    logger.close()  