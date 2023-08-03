
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


# # Pic a

# In[2]:


path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW4/'
img = cv2.imread(path + 'inputs/P4/post-mortem_1.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[3]:


mask = np.zeros_like(img)
mask[img>=200]=1
kernel = np.ones((13,13), np.uint8)
mask = cv2.dilate(mask,kernel,iterations=1)
mask[:100,:250] = 1
plt.imshow(mask,'gray')
plt.imsave(path+'outputs/P4/a/mask.jpg', np.uint8(mask), cmap='gray')


# In[4]:


img = cv2.inpaint(img,mask,20,cv2.INPAINT_TELEA)
plt.imshow(img,'gray')
plt.imsave(path+'outputs/P4/a/inpaint_output.jpg', img, cmap='gray')


# In[5]:


img = cv2.GaussianBlur(img,(5,5),0)
plt.imshow(img,'gray')
plt.imsave(path+'outputs/P4/a/output.jpg', img, cmap='gray')

