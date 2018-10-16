import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

eta = 0.01
n = 300
k=1
iterations = 30
#data normalizing function
def normalize(R):
        mean = np.mean(R)
        range_val = np.amax(R)-np.amin(R)
        R = (R-mean)/float(np.sqrt(np.var(R)))
        return R

#data loading function
def load_data():
	df1 = pd.read_csv('train.csv',header=0)
	M = df1.as_matrix()
	M =M[:,785]
	N = df1.as_matrix()
	N = N[:,1:785].astype(np.float32)
	train_data = np.asarray(N)
	train_labels = np.asarray(M, dtype=np.int32)
	df2 = pd.read_csv('val.csv',header=0)
	m = df2.as_matrix()
	m =m[:,784]
	n = df2.as_matrix()
	n = n[:,0:784].astype(np.float32)
	eval_data = np.asarray(n)
	eval_labels = np.asarray(m, dtype=np.int32)
	return train_data,train_labels,eval_data,eval_labels

#function to compute the sigmoid of a vector
def sigmoid(vec): 
	return (1.0/(1+np.exp(-vec)))

def vec_bin(P):
	Q = (P >= 127)
	return Q.astype(float)

def optimizer():
	X_tr,Y_tr,X_cv,Y_cv = load_data()
	V= np.zeros((X_tr.shape[0],X_tr.shape[1]))
	V = (X_tr >= 127).astype(float)
	#print(len(V[34,:]))
	H = np.zeros((X_tr.shape[0],n)) # n is to be defined	
	W = 0.03*np.random.randn(X_tr.shape[1],n)
	b = np.zeros((X_tr.shape[1],1))
	c = np.zeros((n,1))
	count = int(X_tr.shape[0]/64)
	p = []
	for epoch in range(iterations):
		for e in range(len(X_tr)):
			v = V[e,:]
			h = H[e,:]
			v = np.reshape(v,(X_tr.shape[1],1))
			#print(v.shape)
			h = np.reshape(h,(n,1))
			if(e%10000 == 0):
				print(e)
			for t in range(k):
				prob = sigmoid(np.matmul(np.transpose(W),v)+c)
				u = np.random.uniform(low=0, high=1,size = (prob.shape[0],prob.shape[1]))
				h = (prob > u).astype(float)
				prob1 = sigmoid(np.matmul(W,np.reshape(h,(n,1)))+b)
				#for j in range(X_tr.shape[1]):
				#prob1 = sigmoid(np.matmul(W,np.reshape(h,(n,1)))+b) # sigmoid of W_iV_j + c_i 
				u1 = np.random.uniform(low=0, high=1,size = (prob1.shape[0],prob1.shape[1]))
				v = (prob1 > u1).astype(float)
				#print(W.shape)
				#print(v)
				#print(h.shape)
				#print('done')
					#if u < prob1:
					#	v[j,:] = 1
					#else:
			if (e%count == 0):
				temp = v.reshape((28,28))
				plt.imshow(temp,cmap='gray')
				plt.show()
					#	v[j,:] = 0
			#pri = sigmoid(np.matmul(np.transpose(W),np.transpose(np.asmatrix(V[e,:])))+c)
			#print(pri)
			W = W + eta*(np.transpose(np.matmul(np.reshape(sigmoid(np.matmul(np.transpose(W),np.transpose(np.asmatrix(V[e,:])))+c),(n,1)),(np.asmatrix(V[e,:]))))- np.transpose(np.matmul(np.reshape(sigmoid(np.matmul(np.transpose(W),(v))+c),(n,1)),np.transpose(np.asmatrix(v)))))# GRADIENT_DESCENT W
			#W = W+eta*()
			#print(W.shape)
			#print(W[35,:])
			b = b+eta*(np.transpose(np.asmatrix(V[e,:])) - v)# GRADIENT DESCENT B
			c = c + eta*(((np.reshape(sigmoid(np.matmul(np.transpose(W),np.transpose(np.asmatrix(V[e,:])))+c),(n,1)))- (np.reshape(sigmoid(np.matmul(np.transpose(W),v)+c),(n,1)))))# GRADIENT DESCENT C
	print("done")
	return W,b,c

W1,b1,c1 = optimizer()		
np.save('weights5.npy',W1)
np.save('b5.npy',b1)
np.save('c5.npy',c1)