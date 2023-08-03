
# coding: utf-8

# In[1]:


from skimage import io,transform
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


print('importing train images')
path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/inputs/P4/test2/'
train = []
gender = ['M', 'W']
for gen in gender:
    for i in range(1,10):
        trainlist=[]
        for j in range(1,8):
            img = io.imread(path + gen+"-00"+ str(i)+"-0"+str(j)+ ".bmp")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fft_img = np.fft.fft2(img)
            fft_img = fft_img.ravel()
            trainlist.append(fft_img)
        train.append(np.array(trainlist))
    for i in range(10,51):
        trainlist=[]
        for j in range(1,8):
            img = io.imread(path +gen+ "-0"+ str(i)+"-0"+str(j)+ ".bmp")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            fft_img = np.fft.fft2(img)
            fft_img = fft_img.ravel()
            trainlist.append(fft_img)
        train.append(np.array(trainlist))
    
m, n = img.shape[0], img.shape[1]
#print(len(train))
#print(train[0].shape)


# In[3]:


print('creating filters')
filters = []
for t in range(len(train)):
    
    # X
    X = np.array(train[t])
    X = X.T
    # D and D inverse
    asd = np.abs(X)**2
    diagon = np.sum(asd, axis=1)
    D_diagon = diagon/X.shape[0]
    D_inv_diagon = 1/D_diagon
    D = np.diag(D_diagon)
    D_inv = np.diag(D_inv_diagon)
    # U
    U = np.ones((X.shape[1], 1))
    # X*
    C = np.matrix(X)
    C = C.getH()
    # X* D inverse
    a = np.multiply(C,D_inv_diagon)
    # X* D inverse X
    out = np.zeros((a.shape[0],a.shape[0]), dtype=complex)
    for r in range(a.shape[0]):
        out[r,:] = np.dot(a[r,:],X)
    # (X* D inverse X)inverse
    F=np.linalg.inv(out)
    # (X* D inverse X)inverse U
    Y=np.matmul(F,U)
    # D inverse X
    E = np.multiply(X.T,D_inv_diagon).T
    # H = D inverse X (X* D inverse X)inverse U
    h = np.zeros((E.shape[0],Y.shape[1]), dtype=complex)
    for r in range(E.shape[0]):
        h[r,:] = np.dot(E[r,:],Y)
    
    H=np.reshape(h,(m,n))
    filters.append(H)
    filter_df = pd.DataFrame(H, dtype=complex)
    #filter_df.to_csv('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/outputs/P4/a_mace_filters_csv/hfilter'+str(t)+'.csv',header=False,index=False)
    #plt.imsave('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/outputs/P4/hfilter'+str(t)+'.png',np.abs(H), cmap='gray')
    #plt.imshow(H,cmap='gray')

print('filters are saved!')

