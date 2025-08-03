import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data = pd.read_csv('Neural_Network/NN_Scrach_MNIST/mnist_train.csv')
data = np.array(data)
row, features =data.shape#features = amt of labels + 1,row= no of data points
np.random.shuffle(data)

data_dev=data[0:1000].T
Y_dev=data_dev[0]
X_dev=data_dev[1:features]

data_train = data[1000:row].T 
Y_train = data_train[0]
X_train = data_train[1:features]

X_train = X_train / 255.0
X_dev = X_dev / 255.0


def init_params():
    #randn-0.5-0.5 rand=0-1
    
    W1 = np.random.randn(10, 784) * np.sqrt(2 / 784)
    B1 = np.zeros((10, 1))
    W2 = np.random.randn(10, 10) * np.sqrt(2 / 10)
    B2 = np.zeros((10, 1))

    return W1,B1,W2,B2
    
def ReLU(Z):
    return np.maximum(0,Z)


def Softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # for stability
    return expZ / np.sum(expZ, axis=0, keepdims=True)


    
def fwd_prop(W1,B1,W2,B2,X):
    Z1 = W1.dot(X)+B1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1)+B2
    A2 = Softmax(Z2)
    return Z1,A1,Z2,A2

def one_hot(Y):
    one_hot_Y = np.zeros((10, Y.size))  # 10 classes
    one_hot_Y[Y, np.arange(Y.size)] = 1
    return one_hot_Y

def derivative_relu(Z):
    return Z>0

def back_prop(Z1, A1, Z2, A2,W2, X, Y,):
    m = Y.size
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = (1/m) * dZ2.dot(A1.T)
    dB2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = W2.T.dot(dZ2) * derivative_relu(Z1)
    dW1 = (1/m) * dZ1.dot(X.T)
    dB1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    return dW1, dB1, dW2, dB2


def update_params(W1,B1,W2,B2,dW1,dB1,dW2,dB2,alpha):
    W1=W1-alpha*dW1
    B1=B1-alpha*dB1
    W2=W2-alpha*dW2
    B2=B2-alpha*dB2
    
    return W1,B1,W2,B2

def get_predictions(A):
    return np.argmax(A,0)

def get_accuracy(predictions,Y):
    print(predictions,Y)
    return np.sum(predictions==Y)/Y.size
    


def gradient_descent(X,Y,iterations,alpha):
    W1,B1,W2,B2         =   init_params()
    for i in range(iterations):
        Z1,A1,Z2,A2         =   fwd_prop(W1,B1,W2,B2,X)
        dW1,dB1,dW2,dB2     =   back_prop(Z1,A1,Z2,A2,W2,X,Y)
        W1,B1,W2,B2         =   update_params(W1,B1,W2,B2,dW1,dB1,dW2,dB2,alpha)
        if (i%10==0):
            print("Iteration:\t",i)
            print("Accuracy:\t",get_accuracy(get_predictions(A2),Y))
    return W1,B1,W2,B2


W1,B1,W2,B2 = gradient_descent(X_train,Y_train, 100, 0.1)