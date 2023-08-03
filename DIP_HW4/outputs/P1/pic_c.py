
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


# In[2]:


path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW4/'
img = cv2.imread(path + 'inputs/P1/nasir_receiving_pachekhari.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# In[3]:


histogram = plt.hist(img.ravel(),255,[0,255])
plt.title('Histogram of the Image')
plt.show()


# In[4]:


fftimg = np.fft.fftshift(np.fft.fft2(img))
ff = np.log10(np.abs(fftimg)/1000+1)
plt.imshow(ff, cmap='gray')
plt.imsave(path+'outputs/P1/c/out3_fft.jpg',ff,cmap='gray')
plt.show()


# In[5]:


filt = np.zeros_like(fftimg)+1
m = int((1/2)*fftimg.shape[0])
x = [m-80,m+80]
n = int((1/2)*fftimg.shape[1])
y = [n-50,n+50]
filt[:x[0],y[0]:y[1]] = 0
filt[x[1]:,y[0]:y[1]] = 0
plt.imshow(np.uint8(filt*255),cmap='gray')
plt.title('Mask')
plt.imsave(path+'outputs/P1/c/out3_mask.jpg',np.uint8(filt*255),cmap='gray')
plt.show()


# In[6]:


fft_im = np.multiply(fftimg, filt)
F_img = np.fft.ifft2(np.fft.ifftshift(fft_im))
F_img = np.abs(F_img)
plt.imshow(F_img, cmap='gray')
plt.show()
plt.imsave(path+'outputs/P1/c/out3_1.jpg', F_img, cmap='gray')


# In[7]:


histogram = plt.hist(F_img.ravel(),255,[0,255])
plt.title('Histogram of the filtered Image')
plt.show()


# In[8]:


x = 432
y = 737
len = 50
plt.imshow(F_img[x:x+len, y:y+len], cmap='gray')
plt.title('A Uniform Strip of Input Image')
plt.imsave(path+'outputs/P1/c/uniform.jpg', F_img[x:x+len, y:y+len], cmap='gray')
plt.show()

plt.hist(F_img[x:x+len, y:y+len].ravel(),255,[0,255])
plt.title('Histogram of a Uniform Strip of Input Image')
plt.show()


# In[9]:


from scipy.ndimage import gaussian_filter
filtered_img = gaussian_filter(F_img, 1.2);
plt.imshow(filtered_img, cmap='gray')
plt.imsave(path+'outputs/P1/c/out3_2.jpg', filtered_img, cmap='gray')


# In[10]:


histogram = plt.hist(filtered_img.ravel(),255,[0,255])
plt.title('Histogram of the Output Image')
plt.show()

