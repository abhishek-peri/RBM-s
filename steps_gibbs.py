import numpy as np
import pandas as pd
num_max = 1000
k =20
def sigmoid(vec): 
    return (1.0/(1+np.exp(-vec)))
n= 500
def func(W,b,c):
    h_samples = []
    v_samples = []
    v = np.zeros((784,1))
    for t in range(num_max):
        prob = sigmoid(np.matmul(np.transpose(W), v) + c)
        u = np.random.uniform(low=0, high=1, size=(prob.shape[0], prob.shape[1]))
        h = (prob > u).astype(float)
        h_samples.append(h)
        prob1 = sigmoid(np.matmul(W, np.reshape(h, (n, 1))) + b)
        # for j in range(X_tr.shape[1]):
        # prob1 = sigmoid(np.matmul(W,np.reshape(h,(n,1)))+b) # sigmoid of W_iV_j + c_i
        u1 = np.random.uniform(low=0, high=1, size=(prob1.shape[0], prob1.shape[1]))
        v = (prob1 > u1).astype(float)
        v_samples.append(v)
    h_samples = np.array(h_samples)
    v_samples = np.array(v_samples)
    avg_v = []
    avg_h = []
    for i in range(0, num_max-k):
        add_1 = np.zeros(shape=(784,1))
        add_2 = np.zeros(shape=(784, 1))
        for j in range(i,i+k):
            add_1 = add_1 + v_samples[j]
            add_2 = add_2 + v_samples[j]
        avg_v.append(add_1/k)
        avg_h.append(add_2/k)
    mse_v = []
    mse_h= []

    for i in range(len(avg_v)-1):
        if i%10== 0:
            mse_v.append(np.mean((avg_v[i] - avg_v[i+1])**2))
            mse_h.append(np.mean((avg_h[i] - avg_h[i + 1]) ** 2))

    return np.array(mse_h), np.array(mse_v)

W1= np.random.normal(size = (784,500))
b1 = np.random.normal(size=(784,1))
c1 = np.random.normal(size=(500,1))
m1, m2 = func(W1,b1,c1)
print(m1*100000)
print(m2*100000)    
