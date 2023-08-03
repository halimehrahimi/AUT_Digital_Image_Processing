
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# # Part b functions

# In[2]:


def filtered(img, sigma):
    gaussian_filter = np.zeros((img.shape[0],img.shape[1]))
    a = img.shape[0]//2
    b = img.shape[1]//2
    for x in range(-a, a):
        for y in range(-b, b):
            x1 = np.sqrt(2*np.pi*(sigma**2))
            x2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
            gaussian_filter[x+a-1, y+b-1] = x2
    img_fft = np.fft.fftshift(np.fft.fft2(img))
    filtered_fft = img_fft*gaussian_filter
    filtered_img = np.abs(np.fft.ifft2(filtered_fft))
    
    return filtered_img


# In[3]:


def lowpass_filter(img, sigma, name):
    img_filtered = filtered(img, sigma)
    plt.imsave('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/outputs/P3/part_b/'+name+'_lowpass.png',img_filtered,cmap='gray')
    plt.figure(figsize=(8,8))
    plt.imshow(img_filtered, cmap='gray')
    img_fft = np.fft.fftshift(np.fft.fft2(img_filtered))
    plt.imsave('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/outputs/P3/part_b/'+name+'_lowpass_fft.png',
               np.log10(np.abs(img_fft)/1000+1),cmap='gray')
    return img_filtered


# In[4]:


def highpass_filter(img, sigma, name):
    img_filtered = img-filtered(img, sigma)
    plt.imsave('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/outputs/P3/part_b/'+name+'_highpass.png',img_filtered,cmap='gray')
    plt.figure(figsize=(8,8))
    plt.imshow(img_filtered, cmap='gray')
    plt.show()
    img_fft = np.fft.fftshift(np.fft.fft2(img_filtered))
    plt.imsave('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/outputs/P3/part_b/'+name+'_highpass_fft.png',
               np.log10(np.abs(img_fft)/1000+1),cmap='gray')
    return img_filtered


# # Part c function

# In[5]:


def merge(filtered_1, filtered_2, name):
    merged = filtered_1+filtered_2
    plt.imsave('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/outputs/P3/part_c/'+name+'_merged.png',merged,cmap='gray')
    plt.figure(figsize=(8,8))
    plt.imshow(merged, cmap='gray')
    plt.show()
    img_fft = np.fft.fftshift(np.fft.fft2(merged))
    plt.imsave('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/outputs/P3/part_c/'+name+'_merged_fft.png',
               np.log10(np.abs(img_fft)/1000+1),cmap='gray')
    return merged


# # Joe

# In[6]:


path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/outputs/P3/part_a/'
img_1 = cv2.imread(path + 'joe_4_aligned.png')
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

img_2 = cv2.imread(path + 'joe_6_aligned.png')
img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)


# In[7]:


img_1_filtered = lowpass_filter(img_1, 25, 'joe_4')


# In[8]:


img_2_filtered = highpass_filter(img_2, 25, 'joe_6')


# In[9]:


img_merged = merge(img_1_filtered, img_2_filtered, 'joe')


# # Donald

# In[10]:


path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/outputs/P3/part_a/'
img_1 = cv2.imread(path + 'donald_1_aligned.png')
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

img_2 = cv2.imread(path + 'donald_6_aligned.png')
img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)


# In[11]:


img_1_filtered = lowpass_filter(img_1, 25, 'donald_1')


# In[12]:


img_2_filtered = highpass_filter(img_2, 25, 'donald_6')


# In[13]:


img_merged = merge(img_1_filtered, img_2_filtered, 'donald')

