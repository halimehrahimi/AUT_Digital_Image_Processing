
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[2]:


path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW4/'
img = cv2.imread(path + 'inputs/P1/nasir_and_wives.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# In[3]:


for i in range(3):
    histogram = plt.hist(img[:,:,i].ravel(),255,[0,255])
plt.title('Histogram of the Image')
plt.show()


# In[4]:


x = 373
y = 205
len = 50
plt.imshow(img[x:x+len, y:y+len,:])
plt.title('A Uniform Strip of Input Image');
plt.imsave(path+'outputs/P1/b/uniform.jpg', img[x:x+len, y:y+len])
plt.show()
for i in range(3):
    histogram = plt.hist(img[x:x+len, y:y+len, i].ravel(),255,[0,255])
plt.title('Histogram of a Uniform Strip of Input Image')
plt.show()


# In[5]:


ksize=3
padsize = int((ksize-1)/2)
pad_img = cv2.copyMakeBorder(img, *[padsize]*4, cv2.BORDER_DEFAULT)
F_img = np.zeros_like(img)
for r in range(img.shape[0]):
    for c in range(img.shape[1]):
        for i in range(3):
            F_img[r, c, i] = (np.sum(pad_img[r:r+ksize, c:c+ksize, i]**1.3))/np.sum(pad_img[r:r+ksize, c:c+ksize, i]**0.3)
F_img = np.uint8(F_img)
plt.imshow(F_img)
plt.imsave(path+'outputs/P1/b/out2_2.jpg', F_img, cmap='gray')


# In[7]:


for i in range(3):
    histogram = plt.hist(F_img[:,:,i].ravel(), 255,[0,255])
plt.title('Histogram of the Output Image')
plt.show()


# In[6]:


x = 373
y = 205
len = 50
plt.imshow(F_img[x:x+len, y:y+len,:])
plt.title('A Uniform Strip of Output Image');
plt.imsave(path+'outputs/P1/b/out_uniform.jpg', img[x:x+len, y:y+len])
plt.show()
for i in range(3):
    histogram = plt.hist(F_img[x:x+len, y:y+len, i].ravel(),255,[0,255])
plt.title('Histogram of a Uniform Strip of Output Image')
plt.show()

