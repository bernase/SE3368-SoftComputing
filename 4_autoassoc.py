from sklearn.neural_network import MLPRegressor

import numpy as np
import math
import matplotlib.pyplot as plt
import random

x = np.arange(1,20,0.01);
y = [math.sin(i) + 0.2*random.gauss(0,1) for i in x]

n = len(y)
window = 20
nn = n - window + 1
P = np.empty((0,window))

for i in range(0, nn):
  a = y[i : i+window]
  P = np.vstack([P, a])
print(np.shape(P))

model = MLPRegressor(activation='tanh',hidden_layer_sizes=(10,),verbose=True,learning_rate='adaptive',max_iter=1000)
model.fit(P, P)
yy = model.predict(P)

out = np.zeros(len(x))

for i in range(0,yy.shape[0]):
  for j in range(0,window):
    out[i+j] += yy[i][j]

k = 1
sz = len(out)
for i in range(0,sz):
  out[i] = out[i]/k
  
  if k == window and window <= sz - i:
    k = window
  elif window > sz - i:
    k = k - 1 
  else:
    k = k + 1

plt.plot(x,y)
plt.show()

plt.plot(x, out)
plt.show()