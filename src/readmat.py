import h5py
import os
import numpy as np
import matplotlib.pyplot as plt

 
fname='00001.png'
a = h5py.File(os.path.join('./dataset/train/direction/', fname+'.mat'),'r')

print(type(a))  #dict 
x=list(a.keys())

www=a[x[0]].value
www=np.transpose(www,(2,1,0))

plt.figure()
plt.imshow(www)
plt.show()