
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from pypher import pypher


# # Functions

# In[2]:


def deblur_ls(img, single_kernel):
    H=pypher.psf2otf(single_kernel,shape=(img.shape[0], img.shape[1]))
    y = np.fft.fftshift(np.fft.fft2(img))
    a = H.conj()*H
    x_ls_img = H.conj()*y/a
    x_ls_img = np.abs(np.fft.ifft2(np.fft.ifftshift(x_ls_img)))
    return x_ls_img


# In[3]:


def deblur2d_ls_plot(img, kernel):
    plt.figure(figsize=(20,12))
    for i in range(4):
        filtered_img = deblur_ls(img[i], kernel[i])
        plt.subplot(2,2,i+1)
        plt.title(f'kernel {i+1}')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(filtered_img, cmap='gray')
        plt.imsave(path+f'outputs/P5/part_b/donald_1_deblur_ls_kernel{i+1}.png', filtered_img, cmap='gray')
    plt.savefig(path+f'outputs/P5/part_b/donald_1_deblur_ls.png')
    plt.show()


# In[4]:


def deblur3d_ls_plot(img, kernel):
    plt.figure(figsize=(20,12))
    for i in range(4):
        filtered_img = []
        for j in range(3):
            fil = deblur_ls(img[i][:,:,j],kernel[i])
            filtered_img.append(fil/np.max(fil)*255)
        filtered_img = np.dstack(filtered_img)
        filtered_img = cv2.cvtColor(filtered_img.astype('uint8'), cv2.COLOR_BGR2RGB)
        plt.subplot(2,2,i+1)
        plt.title(f'kernel {i+1}')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(filtered_img, cmap='gray')
        plt.imsave(path+f'outputs/P5/part_b/donald_2_deblur_ls_kernel_{i+1}.png', filtered_img, cmap='gray')
    plt.savefig(path+f'outputs/P5/part_b/donald_2_deblur_ls.png')


# # Importing the Data

# In[5]:


path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/'
kernel = []
for i in range(1,5):
    img = cv2.imread(path + f'inputs/P5/kernel_{i}.png')
    kernel.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


# In[6]:


donald_gray = []
for i in range(1,5):
    img = cv2.imread(path + f'outputs/P5/part_a/donald_1_blur_kernel_{i}.png')
    donald_gray.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))


# In[7]:


donald_rgb = []
for i in range(1,5):
    img = cv2.imread(path + f'outputs/P5/part_a/donald_2_blur_kernel_{i}.png')
    donald_rgb.append(img)


# # Donald in the Car Grayscale

# In[8]:


deblur2d_ls_plot(donald_gray,kernel)


# # Donald in the Car Grayscale

# In[9]:


deblur3d_ls_plot(donald_rgb,kernel)

