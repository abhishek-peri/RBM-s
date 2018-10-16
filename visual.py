import pandas as pd
import numpy as np
from ggplot import *
import matplotlib.pyplot as plt
from tsne import bh_sne


import time

from sklearn.manifold import TSNE


def sigmoid(vec): 
	L = []
	for i in range(len(vec)):
    		L.append(1.0/(1+np.exp(-vec[i])))
  	return np.asarray(L)

def load_data():
	df2 = pd.read_csv('test.csv',header=0)
	m = df2.as_matrix()
	m =m[:,785]
	n = df2.as_matrix()
	n = n[:,1:785].astype(np.float32)
	eval_data = np.asarray(n)
	eval_labels = np.asarray(m, dtype=np.int32)
	return eval_data,eval_labels

def load_param():
	W = np.load('500_1_20/weights3.npy')
	b = np.load('500_1_20/b3.npy')
	c = np.load('500_1_20/c3.npy')
	#print(W[35,:])
	return W,b,c

def vec_bin(P):
	Q = (P >= 127)
	return Q.astype(float)

def hidden():
	#call the load_data
	X,y = load_data()
	#convert X to binary
	W1,b1,c1 = load_param()
	#print(b1)
	H = []
	#convert the test data into hidden state
	for i in range(len(X)):
		X[i,:] = vec_bin(X[i,:])
		H.append(sigmoid(np.matmul(W1.T,np.reshape(X[i,:],(784,1)))+c1))
	return np.asarray(H),y
	

h,y = hidden()
#print(h.shape)
#print(y.shape)
h= h.reshape(10000,500)
#rndperm1 = np.random.permutation(df.shape[0])
#df_tsne = df.loc[rndperm1[:n_sne],:].copy()
#print(h[45,:])

X = h
print(X.shape)
# feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
# df = pd.DataFrame(X,columns=feat_cols)
# df['label'] = y_
# df['label'] = df['label'].apply(lambda i: str(i))
# X, y_ = None, None
# print 'Size of the dataframe: {}'.format(df.shape)
# rndperm = np.random.permutation(df.shape[0])
# n_sne = 5000
# time_start = time.time()
# tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
# tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)
# print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)
# df_tsne = df.loc[rndperm[:n_sne],:].copy()
# df_tsne['x-tsne'] = tsne_results[:,0]
# df_tsne['y-tsne'] = tsne_results[:,1]
# chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
#         + geom_point(size=70,alpha=0.1) \
#         + ggtitle("tSNE dimensions colored by digit")
# #	return chart	
# print(chart)
vis_data = bh_sne(X)

# plot the result
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

plt.scatter(vis_x, vis_y, c=y, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()