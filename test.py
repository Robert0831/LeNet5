import torch
import torch.nn as nn
import numpy as np
from dataset import Dataset50Loader
from thop import profile
import pandas as pd
import time
import os
import tensorflow as tf

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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


val_loader=Dataset50Loader('test.txt')
totalv=len(val_loader.dataset)
device = torch.device("cpu")

################ pytorch

# model=LeNet5()
# model.load_state_dict(torch.load("tor_model_49.pth" ,map_location=device))
# ac=0
# times=[]
# for batch_idxv, (img,data) in enumerate(val_loader):
#     model.eval()
#     with torch.no_grad():

#         start=time.time()
#         out=model(torch.Tensor(img.to(torch.float32).to(device)))
#         times.append(time.time() - start)

#         out=torch.argmax(out,1)
#     for j in range(len(data)):
#             if out[j]==data[j]:
#                 ac+=1

# print(f'pytorch test top-1 accuracy:{round(ac/totalv,5)},') 
# print(f'pytorch inference time , batch=1:{round(sum(times)/totalv,5)},')


# test=torch.randn(1,3,64,64)
# fl,pa=profile(model,inputs=test)
# print(f"flops:{fl*2/10**9} G  params:{pa}")




###################   tf  ################


# ac=0
# times=[]
# model_tf = tf.keras.models.load_model("tf_model_49.keras")   

# for batch_idx, (img,data) in enumerate(val_loader):
#     img=np.transpose(img.numpy(),(0,2,3,1)).astype(np.float32)
#     data=data.numpy().astype(np.float32)

#     start=time.time()
#     y_pred=model_tf(img)
#     times.append(time.time() - start)

#     y_pred=np.argmax(y_pred,axis=1)
#     for i in range(len(data)):
#             if y_pred[i]==data[i]:
#                 ac+=1

# print(f'tf test top-1 accuracy:{round(ac/totalv,5)},') 
# print(f'tf inference time , batch=1:{round(sum(times)/totalv,5)},')



# from keras_flops import get_flops
# fl=get_flops(model_tf, batch_size=1)
# pa=model_tf.count_params()
# print(f"flops:{fl/10**9} G  params:{pa}")


#############  tf-static


# ac=0
# times=[]
# model_tf = tf.keras.models.load_model("tf_model_static.keras")   
# @tf.function
# def model_func(x):
#     y_pr=model_tf(x)
#     return y_pr

# for batch_idx, (img,data) in enumerate(val_loader):
#     img=np.transpose(img.numpy(),(0,2,3,1)).astype(np.float32)
#     data=data.numpy().astype(np.float32)

#     start=time.time()
#     y_pred=model_func(img)
#     times.append(time.time() - start)

#     y_pred=np.argmax(y_pred,axis=1)
#     for i in range(len(data)):
#             if y_pred[i]==data[i]:
#                 ac+=1

# print(f'tf-sta test top-1 accuracy:{round(ac/totalv,5)},') 
# print(f'tf-sta inference time , batch=1:{round(sum(times)/totalv,5)},')


########################### post-training quantization
# model=LeNet5()
# model = torch.quantization.quantize_dynamic(
#     model, {torch.nn.Linear,torch.nn.Conv2d}, dtype=torch.qint8
# )
# model.load_state_dict(torch.load("quantized_model.pth" ,map_location=device))
# ac=0
# times=[]
# for batch_idxv, (img,data) in enumerate(val_loader):
#     model.eval()
#     with torch.no_grad():

#         start=time.time()
#         out=model(torch.Tensor(img.to(torch.float32).to(device)))
#         times.append(time.time() - start)

#         out=torch.argmax(out,1)
#     for j in range(len(data)):
#             if out[j]==data[j]:
#                 ac+=1

# print(f'pytorch-ptq test top-1 accuracy:{round(ac/totalv,5)},') 
# print(f'pytorch-ptq inference time , batch=1:{round(sum(times)/totalv,5)},')


# test=torch.randn(1,3,64,64)
# fl,pa=profile(model,inputs=test)
# print(f"flops:{fl*2/10**9} G  params:{pa}")

########################  pruning
model=torch.load("quantized_model_1.pth" ,map_location=device)

ac=0
times=[]
for batch_idxv, (img,data) in enumerate(val_loader):
    with torch.no_grad():

        start=time.time()
        out=model(torch.Tensor(img.to(torch.float32).to(device)))
        times.append(time.time() - start)

        out=torch.argmax(out,1)
    for j in range(len(data)):
            if out[j]==data[j]:
                ac+=1

print(f'pytorch-pruning test top-1 accuracy:{round(ac/totalv,5)},') 
print(f'pytorch-pruning inference time , batch=1:{round(sum(times)/totalv,5)},')

test=torch.randn(1,3,64,64)
fl,pa=profile(model,inputs=test)
print(f"flops:{fl*2/10**9} G  params:{pa}")
