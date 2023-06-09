import pickle
import random
import numpy as np
import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod
import os
import cv2
import pandas as pd
import time
def load():
    with open("mnist.pkl",'rb') as f:
        mnist = pickle.load(f)
    return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

def MakeOneHot(Y, D_out):
    N = Y.shape[0]
    Z = np.zeros((N, D_out))
    Z[np.arange(N), Y] = 1
    return Z

def draw_losses(losses):
    t = np.arange(len(losses))
    plt.plot(t, losses)
    plt.show()

def get_batch(X, Y, batch_size):
    N = len(X)
    i = random.randint(1, N-batch_size)
    return X[i:i+batch_size], Y[i:i+batch_size]

class FC():
    """
    Fully connected layer
    """
    def __init__(self, D_in, D_out):
        #print("Build FC")
        self.cache = None
        #self.W = {'val': np.random.randn(D_in, D_out), 'grad': 0}
        self.W = {'val': np.random.normal(0.0, np.sqrt(2/D_in), (D_in,D_out)), 'grad': 0}
        self.b = {'val': np.random.randn(D_out), 'grad': 0}

    def _forward(self, X):
        #print("FC: _forward")
        out = np.dot(X, self.W['val']) + self.b['val']
        self.cache = X
        return out

    def _backward(self, dout):
        #print("FC: _backward")
        X = self.cache
        dX = np.dot(dout, self.W['val'].T).reshape(X.shape)
        self.W['grad'] = np.dot(X.reshape(X.shape[0], np.prod(X.shape[1:])).T, dout)
        self.b['grad'] = np.sum(dout, axis=0)
        #self._update_params()
        return dX

    def _update_params(self, lr=0.001):
        # Update the parameters
        self.W['val'] -= lr*self.W['grad']
        self.b['val'] -= lr*self.b['grad']

class ReLU():
    """
    ReLU activation layer
    """
    def __init__(self):
        #print("Build ReLU")
        self.cache = None

    def _forward(self, X):
        #print("ReLU: _forward")
        out = np.maximum(0, X)
        self.cache = X
        return out

    def _backward(self, dout):
        #print("ReLU: _backward")
        X = self.cache
        dX = np.array(dout, copy=True)
        dX[X <= 0] = 0
        return dX
class tanh():
    """
    tanh activation layer
    """
    def __init__(self):
        self.cache = X

    def _forward(self, X):
        self.cache = X
        return np.tanh(X)

    def _backward(self, X):
        X = self.cache
        dX = dout*(1 - np.tanh(X)**2)
        return dX
class Sigmoid():
    """
    Sigmoid activation layer
    """
    def __init__(self):
        self.cache = None

    def _forward(self, X):
        self.cache = X
        return 1 / (1 + np.exp(-X))

    def _backward(self, dout):
        X = self.cache
        X=1 / (1 + np.exp(-X))
        dX = dout*X*(1-X)
        return dX


class Softmax():
    """
    Softmax activation layer
    """
    def __init__(self):
        #print("Build Softmax")
        self.cache = None

    def _forward(self, X):
        #print("Softmax: _forward")
        maxes = np.amax(X, axis=1)
        maxes = maxes.reshape(maxes.shape[0], 1)
        Y = np.exp(X - maxes)
        Z = Y / np.sum(Y, axis=1).reshape(Y.shape[0], 1)
        self.cache = (X, Y, Z)
        return Z # distribution

    def _backward(self, dout):
        X, Y, Z = self.cache
        dZ = np.zeros(X.shape)
        dY = np.zeros(X.shape)
        dX = np.zeros(X.shape)
        N = X.shape[0]
        for n in range(N):
            i = np.argmax(Z[n])
            dZ[n,:] = np.diag(Z[n]) - np.outer(Z[n],Z[n])
            M = np.zeros((N,N))
            M[:,i] = 1
            dY[n,:] = np.eye(N) - M
        dX = np.dot(dout,dZ)
        dX = np.dot(dX,dY)
        return dX


class Conv():
    """
    Conv layer
    """
    def __init__(self, Cin, Cout, F, stride=1, padding=0, bias=True):
        self.Cin = Cin
        self.Cout = Cout
        self.F = F
        self.S = stride
        #self.W = {'val': np.random.randn(Cout, Cin, F, F), 'grad': 0}
        self.W = {'val': np.random.normal(0.0,np.sqrt(2/Cin),(Cout,Cin,F,F)), 'grad': 0} # Xavier Initialization
        self.b = {'val': np.random.randn(Cout), 'grad': 0}
        self.cache = None
        self.pad = padding

    def _forward(self, X):
        X = np.pad(X, ((0,0),(0,0),(self.pad,self.pad),(self.pad,self.pad)), 'constant')
        (N, Cin, H, W) = X.shape
        H_ = H - self.F + 1
        W_ = W - self.F + 1
        Y = np.zeros((N, self.Cout, H_, W_))

        for n in range(N):
            for c in range(self.Cout):
                for h in range(H_):
                    for w in range(W_):
                        Y[n, c, h, w] = np.sum(X[n, :, h:h+self.F, w:w+self.F] * self.W['val'][c, :, :, :]) + self.b['val'][c]

        self.cache = X
        return Y

    def _backward(self, dout):
        # dout (N,Cout,H_,W_)
        # W (Cout, Cin, F, F)
        X = self.cache
        (N, Cin, H, W) = X.shape
        H_ = H - self.F + 1
        W_ = W - self.F + 1
        W_rot = np.rot90(np.rot90(self.W['val']))

        dX = np.zeros(X.shape)
        dW = np.zeros(self.W['val'].shape)
        db = np.zeros(self.b['val'].shape)

        # dW
        for co in range(self.Cout):
            for ci in range(Cin):
                for h in range(self.F):
                    for w in range(self.F):
                        dW[co, ci, h, w] = np.sum(X[:,ci,h:h+H_,w:w+W_] * dout[:,co,:,:])

        # db
        for co in range(self.Cout):
            db[co] = np.sum(dout[:,co,:,:])

        dout_pad = np.pad(dout, ((0,0),(0,0),(self.F,self.F),(self.F,self.F)), 'constant')
        #print("dout_pad.shape: " + str(dout_pad.shape))
        # dX
        for n in range(N):
            for ci in range(Cin):
                for h in range(H):
                    for w in range(W):
                        #print("self.F.shape: %s", self.F)
                        #print("%s, W_rot[:,ci,:,:].shape: %s, dout_pad[n,:,h:h+self.F,w:w+self.F].shape: %s" % ((n,ci,h,w),W_rot[:,ci,:,:].shape, dout_pad[n,:,h:h+self.F,w:w+self.F].shape))
                        dX[n, ci, h, w] = np.sum(W_rot[:,ci,:,:] * dout_pad[n, :, h:h+self.F,w:w+self.F])

        return dX

class MaxPool():
    def __init__(self, F, stride):
        self.F = F
        self.S = stride
        self.cache = None

    def _forward(self, X):
        # X: (N, Cin, H, W): maxpool along 3rd, 4th dim
        (N,Cin,H,W) = X.shape
        F = self.F
        W_ = int(float(W)/F)
        H_ = int(float(H)/F)
        Y = np.zeros((N,Cin,W_,H_))
        M = np.zeros(X.shape) # mask
        for n in range(N):
            for cin in range(Cin):
                for w_ in range(W_):
                    for h_ in range(H_):
                        Y[n,cin,w_,h_] = np.max(X[n,cin,F*w_:F*(w_+1),F*h_:F*(h_+1)])
                        i,j = np.unravel_index(X[n,cin,F*w_:F*(w_+1),F*h_:F*(h_+1)].argmax(), (F,F))
                        M[n,cin,F*w_+i,F*h_+j] = 1
        self.cache = M
        return Y

    def _backward(self, dout):
        M = self.cache
        (N,Cin,H,W) = M.shape
        dout = np.array(dout)
        (a1,a2,a3,a4)=dout.shape
        #print("dout.shape: %s, M.shape: %s" % (dout.shape, M.shape))
        if a3*2!=H:
            dX = np.zeros((N,Cin,H-1,W-1))
            dXX=np.zeros((N,Cin,H,W))
            for n in range(N):
                for c in range(Cin):
                    #print("(n,c): (%s,%s)" % (n,c))
                    dX[n,c,:,:] = dout[n,c,:,:].repeat(2, axis=0).repeat(2, axis=1)
            dXX[:,:,:-1,:-1]=dX
            return dXX*M
        
        else:
            dX = np.zeros(M.shape)
            for n in range(N):
                for c in range(Cin):
                    #print("(n,c): (%s,%s)" % (n,c))
                    dX[n,c,:,:] = dout[n,c,:,:].repeat(2, axis=0).repeat(2, axis=1)
            return dX*M

def NLLLoss(Y_pred, Y_true):
    """
    Negative log likelihood loss
    """
    loss = 0.0
    N = Y_pred.shape[0]
    M = np.sum(Y_pred*Y_true, axis=1)
    for e in M:
        if np.all(e) == 0:
            loss += 500
        else:
            loss += -np.log(e)
    return loss/N

class CrossEntropyLoss():
    def __init__(self):
        pass

    def get(self, Y_pred, Y_true):
        N = Y_pred.shape[0]
        softmax = Softmax()
        prob = softmax._forward(Y_pred)
        #loss = NLLLoss(prob, Y_true)
        Y_serial = np.argmax(Y_true, axis=1)
        dout = prob.copy()
        dout[np.arange(N), Y_serial] -= 1
        #return loss, dout
        return dout

class SoftmaxLoss():
    def __init__(self):
        pass

    def get(self, Y_pred, Y_true):
        N = Y_pred.shape[0]
        loss = NLLLoss(Y_pred, Y_true)
        Y_serial = np.argmax(Y_true, axis=1)
        dout = Y_pred.copy()
        dout[np.arange(N), Y_serial] -= 1
        return loss, dout
class XSigmoid():
    def __init__(self):
        self.cache = None

    def _forward(self, X):
        self.cache = X
        return X / (1 + np.exp(-X))

    def _backward(self, dout):
        X = self.cache
        SX=1 / (1 + np.exp(-X)) 
        SXX=SX*X
        dX = dout*(SXX+SX*(1-SXX))
        return dX

class Net(metaclass=ABCMeta):
    # Neural network super class

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, X):
        pass

    @abstractmethod
    def backward(self, dout):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def set_params(self, params):
        pass

class NLeNet5(Net):


    def __init__(self,weights=''):
        self.conv1 = Conv(3, 6, 3)#64->62
        self.sig1 = XSigmoid()
        self.pool1 = MaxPool(2,2) #62->31
        self.conv2 = Conv(6, 16, 3) #31->29
        self.sig2 = XSigmoid()
        self.pool2 = MaxPool(2,2) #29->14
        self.conv3=  Conv(16, 120,3 ) #14->12
        self.sig3 = XSigmoid()
        self.pool3 = MaxPool(2,2) #12->6
        self.conv4=  Conv(120,50 ,3 ) #6->4


        self.FC1 = FC(50*4*4, 84)
        self.sig4 = XSigmoid()
        self.FC2 = FC(84, 50)
        self.Softmax = Softmax()
        if weights == '':
            pass
        else:
            with open(weights,'rb') as f:
                params = pickle.load(f)
                self.set_params(params)
        self.p2_shape = None

    def forward(self, X):
        h1 = self.conv1._forward(X)
        a1 = self.sig1._forward(h1)
        p1 = self.pool1._forward(a1)
        h2 = self.conv2._forward(p1)
        a2 = self.sig2._forward(h2)
        p2 = self.pool2._forward(a2)
        h3=self.conv3._forward(p2)
        a3=self.sig3._forward(h3)
        p3 = self.pool3._forward(a3)
        h4=self.conv4._forward(p3)

        self.h4_shape = h4.shape
        fl = h4.reshape(X.shape[0],-1) # Flatten
        h5 = self.FC1._forward(fl)
        a5 = self.sig4._forward(h5)
        h6 = self.FC2._forward(a5)
        a6 = self.Softmax._forward(h6)
        return a6

    def backward(self, dout):
        dout = self.FC2._backward(dout)
        dout = self.sig4._backward(dout)
        dout = self.FC1._backward(dout)

        dout = dout.reshape(self.h4_shape) # reshape
        dout=self.conv4._backward(dout)
        dout = self.pool3._backward(dout)
        dout = self.sig3._backward(dout)
        dout=self.conv3._backward(dout)
        dout = self.pool2._backward(dout)
        dout = self.sig2._backward(dout)
        dout = self.conv2._backward(dout)
        dout = self.pool1._backward(dout)
        dout = self.sig1._backward(dout)
        dout = self.conv1._backward(dout)

    def get_params(self):
        return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.conv3.W, self.conv3.b,self.conv4.W, self.conv4.b,self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b]

    def set_params(self, params):
        [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.conv3.W, self.conv3.b,self.conv4.W, self.conv4.b,self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b] = params
class LeNet5(Net):
    # LeNet5

    def __init__(self,weights=''):
        self.conv1 = Conv(3, 6, 5)#64->60
        self.sig1 = Sigmoid()
        self.pool1 = MaxPool(2,2) #60->30
        self.conv2 = Conv(6, 16, 5) #30->26
        self.sig2 = Sigmoid()
        self.pool2 = MaxPool(2,2) #26->13
        self.conv3=  Conv(16, 120, 5) #13->9

        self.FC1 = FC(120*9*9, 84)
        self.sig3 = Sigmoid()
        self.FC2 = FC(84, 50)
        self.Softmax = Softmax()
        if weights == '':
            pass
        else:
            with open(weights,'rb') as f:
                params = pickle.load(f)
                self.set_params(params)
        self.p2_shape = None

    def forward(self, X):
        h1 = self.conv1._forward(X)
        a1 = self.sig1._forward(h1)
        p1 = self.pool1._forward(a1)
        h2 = self.conv2._forward(p1)
        a2 = self.sig2._forward(h2)
        p2 = self.pool2._forward(a2)
        h3=self.conv3._forward(p2)

        self.p3_shape = h3.shape
        fl = h3.reshape(X.shape[0],-1) # Flatten
        h4 = self.FC1._forward(fl)
        a4 = self.sig3._forward(h4)
        h5 = self.FC2._forward(a4)
        a5 = self.Softmax._forward(h5)
        return a5

    def backward(self, dout):
        dout = self.FC2._backward(dout)
        dout = self.sig3._backward(dout)
        dout = self.FC1._backward(dout)

        dout = dout.reshape(self.p3_shape) # reshape
        dout=self.conv3._backward(dout)
        dout = self.pool2._backward(dout)
        dout = self.sig2._backward(dout)
        dout = self.conv2._backward(dout)
        dout = self.pool1._backward(dout)
        dout = self.sig1._backward(dout)
        dout = self.conv1._backward(dout)

    def get_params(self):
        return [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.conv3.W, self.conv3.b,self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b]

    def set_params(self, params):
        [self.conv1.W, self.conv1.b, self.conv2.W, self.conv2.b, self.conv3.W, self.conv3.b,self.FC1.W, self.FC1.b, self.FC2.W, self.FC2.b] = params
class fnn(Net):
    # LeNet5

    def __init__(self, weights=''):
        self.conv1 = Conv(3, 1, 3) #64->62
        self.ReLU1 = ReLU()
        self.pool1 = MaxPool(2,2)  #62->31
        self.FC1 = FC(31*31, 50)
        self.Softmax = Softmax()
        if weights == '':
            pass
        else:
            with open(weights,'rb') as f:
                params = pickle.load(f)
                self.set_params(params)
        self.p1_shape = None

    def forward(self, X):
        h1 = self.conv1._forward(X)
        a1 = self.ReLU1._forward(h1)
        p1 = self.pool1._forward(a1)
        self.p1_shape = p1.shape
        fl = p1.reshape(X.shape[0],-1) # Flatten
        h2 = self.FC1._forward(fl)
        a2 = self.Softmax._forward(h2)
        return a2

    def backward(self, dout):
        #dout = self.Softmax._backward(dout)
        dout = self.FC1._backward(dout)
        dout = dout.reshape(self.p1_shape) # reshape
        dout = self.pool1._backward(dout)
        dout = self.ReLU1._backward(dout)
        dout = self.conv1._backward(dout)

    def get_params(self):
        return [self.conv1.W, self.conv1.b, self.FC1.W, self.FC1.b]

    def set_params(self, params):
        [self.conv1.W, self.conv1.b, self.FC1.W, self.FC1.b] = params

class SGD():
    def __init__(self, params, lr=0.001, reg=0):
        self.parameters = params
        self.lr = lr
        self.reg = reg

    def step(self):
        for param in self.parameters:
            param['val'] -= (self.lr*param['grad'] + self.reg*param['val'])

class SGDMomentum():
    def __init__(self, params, lr=0.001, momentum=0.99, reg=0):
        self.l = len(params)
        self.parameters = params
        self.velocities = []
        for param in self.parameters:
            self.velocities.append(np.zeros(param['val'].shape))
        self.lr = lr
        self.rho = momentum
        self.reg = reg

    def step(self):
        for i in range(self.l):
            self.velocities[i] = self.rho*self.velocities[i] + (1-self.rho)*self.parameters[i]['grad']
            self.parameters[i]['val'] -= (self.lr*self.velocities[i] + self.reg*self.parameters[i]['val'])

def tolabel(y):
    a=np.zeros((1,50))
    a[0][y]=1
    return np.expand_dims(a,0)

# def getimg(set,idx):
#     images=[]
#     for i in range(5):
#         img=set['dir'][int(idx+i)]
#         img = cv2.imread(img)
#         img = cv2.resize(img, (64, 64))
#         img=img/255
#         img=img.transpose(2,0,1)
#         images.append(img)
#     return np.array(images)

def getimg(set,idx):
    images=[]
    img=set['dir'][int(idx)]
    img = cv2.imread(img)
    img = cv2.resize(img, (64, 64))
    img=img/255
    img=img.transpose(2,0,1)
    images.append(img)
    return np.array(images)
# def getlabel(set,idx):
#     labels=[]
#     for i in range(5):
#         labels.append(set['class'][int(idx+i)])
#     return np.array(labels)

def getlabel(set,idx):
    labels=[]
    
    labels.append(set['class'][int(idx)])
    return np.array(labels)


model2=LeNet5('Lanet_weights.pkl')


testpath=os.path.join(os.path.abspath(""),'test.txt')
testset=pd.read_csv(testpath,sep=" ",header=None,names=["dir", "class"])

totalt=len(testset)

score2=0
times=[]

for iv in range(0,len(testset),1):
    imgv=getimg(testset,iv)
    labelv=getlabel(testset,iv)
    labelv=MakeOneHot(labelv,50)
    labelv=np.argmax(labelv,1)


    start=time.time()
    Y_pred2 = model2.forward(imgv)
    times.append(time.time() - start)
    out2=np.argmax(Y_pred2,1)
    for j in range(out2.shape[0]):
        if out2[j]==labelv[j]:
            score2+=1
    print(iv)


print(f'batch=1  infereance time :{round(round(sum(times)/len(times),5))} ;LeNet5 top-1 accuracy:{round(score2/totalt,5)} ') 
