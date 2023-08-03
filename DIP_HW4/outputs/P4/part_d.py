
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


# # Pic b

# In[2]:


path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW4/'
img = cv2.imread(path + 'inputs/P4/post-mortem_4.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[3]:


mask = np.zeros_like(img)
mask[img>=255]=1
kernel = np.ones((13,13), np.uint8)
mask = cv2.dilate(mask,kernel,iterations=1)
plt.imshow(mask,'gray')
plt.imsave(path+'outputs/P4/d/mask.jpg', np.uint8(mask), cmap='gray')


# In[4]:


img = cv2.inpaint(img,mask,25,cv2.INPAINT_TELEA)
plt.imshow(img,'gray')
plt.imsave(path+'outputs/P4/d/inpaint_output.jpg', img, cmap='gray')


# In[5]:


fftimg = np.fft.fftshift(np.fft.fft2(img))
plt.imshow(np.log10(np.abs(fftimg)/1000+1), cmap='gray')
plt.imsave(path+'outputs/P4/d/fft.jpg',np.uint8(np.log10(np.abs(fftimg)/1000+1)),cmap='gray')
plt.show()


# In[6]:


filt = np.zeros_like(fftimg)
m = int((1/2)*fftimg.shape[0])
x = [m-70,m+70]
filt[x[0]:x[1],:] = 1
plt.imshow(np.uint8(filt*255),cmap='gray')
plt.title('Mask')
plt.imsave(path+'outputs/P4/d/mask_fft.jpg',np.uint8(filt),cmap='gray')
plt.show()
fft_im = np.multiply(fftimg, filt)
F_img = np.fft.ifft2(np.fft.ifftshift(fft_im))
F_img = np.abs(F_img)
plt.imshow(F_img, cmap='gray')
plt.show()
plt.imsave(path+'outputs/P4/d/output_fft.jpg', F_img, cmap='gray')


# In[7]:


kernel = np.ones((3,3), np.uint8)
new = cv2.dilate(np.uint8(F_img),kernel,iterations=1)
new = cv2.erode(np.uint8(new),kernel,iterations=1)
plt.figure(figsize=(10,10))
plt.imshow(new,'gray')
plt.imsave(path+'outputs/P4/d/output.jpg', new, cmap='gray')
plt.show()

