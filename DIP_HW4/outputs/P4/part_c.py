
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


# # Pic a

# In[2]:


path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW4/'
img = cv2.imread(path + 'inputs/P4/post-mortem_3.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[3]:


mask = np.zeros_like(img)
mask[img>=255]=1
kernel = np.ones((13,13), np.uint8)
mask = cv2.dilate(mask,kernel,iterations=1)
mask[:,210:]=0
plt.imshow(mask,'gray')
plt.imsave(path+'outputs/P4/c/mask.jpg', np.uint8(mask), cmap='gray')


# In[4]:


img = cv2.inpaint(img,mask,25,cv2.INPAINT_TELEA)
plt.imshow(img,'gray')
plt.imsave(path+'outputs/P4/c/inpaint_output.jpg', img, cmap='gray')


# In[5]:


kernel = np.ones((3,3), np.uint8)
new = cv2.erode(np.uint8(img),kernel,iterations=1)
new = cv2.dilate(np.uint8(new),kernel,iterations=1)
plt.figure(figsize=(10,10))
plt.imshow(new,'gray')
plt.imsave(path+'outputs/P4/c/output.jpg', new, cmap='gray')
plt.show()

