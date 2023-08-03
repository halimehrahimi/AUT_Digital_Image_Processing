
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# In[2]:


path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW4/'
img = cv2.imread(path + 'inputs/P1/nasir_and_dentist.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[3]:


histogram = plt.hist(img.ravel(),255,[0,255])
plt.title('Histogram of the Image')
plt.show()


# In[4]:


x = 339
y = 512
len = 50
plt.imshow(img[x:x+len, y:y+len], cmap='gray')
plt.title('A Uniform Strip of Input Image');
plt.imsave(path+'outputs/P1/a/uniform.jpg', img[x:x+len, y:y+len], cmap='gray')
plt.show()

histogram = plt.hist(img[x:x+len, y:y+len].ravel(),255,[0,255])
plt.title('Histogram of a Uniform Strip of Input Image')
plt.show()


# In[6]:


filtered_img = gaussian_filter(img, 0.8)
plt.imshow(filtered_img,cmap='gray')
plt.show()
plt.imsave(path+'outputs/P1/a/out1-1.jpg', filtered_img,cmap='gray')


# In[7]:


histogram = plt.hist(filtered_img.ravel(), 255, [0,255])
plt.title('Histogram of Filtered Image')
plt.show()


# In[8]:


ksize=3
padsize = int((ksize-1)/2)
pad_img = cv2.copyMakeBorder(img, *[padsize]*4, cv2.BORDER_DEFAULT)
F_img = np.zeros_like(img)
for r in range(img.shape[0]):
    for c in range(img.shape[1]):
        F_img[r, c] = np.median(pad_img[r:r+ksize, c:c+ksize])
F_img = np.uint8(F_img)
plt.imshow(F_img,'gray')
plt.show()
plt.imsave(path+'outputs/P1/a/out1-2.jpg', F_img, cmap='gray')


# In[9]:


histogram = plt.hist(F_img.ravel(), 255, [0,255])
plt.title('Histogram of Output Image')
plt.show()

