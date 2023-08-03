
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2


# # Part a.

# In[2]:


def global_thresh(img, thresh):
    new = img.copy()
    new[new>=thresh] = 255
    new[new<thresh] = 0
    new = new.astype('uint8')
    return new


# In[54]:


path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW2/inputs/P3/'

quran_1 = cv2.imread(path + 'quran_1.png')
quran_1 = cv2.cvtColor(quran_1, cv2.COLOR_BGR2GRAY)

quran_2 = cv2.imread(path + 'quran_2.png')
quran_2 = cv2.cvtColor(quran_2, cv2.COLOR_BGR2GRAY)

quran_3 = cv2.imread(path + 'quran_3.png')
quran_3 = cv2.cvtColor(quran_3, cv2.COLOR_BGR2GRAY)

quran_4 = cv2.imread(path + 'quran_4.png')
quran_4 = cv2.cvtColor(quran_4, cv2.COLOR_BGR2GRAY)

fig, axs = plt.subplots(2,2, figsize=(10,8))
axs[0,0].imshow(quran_1, cmap='gray')
axs[0,0].set_title('Quran 01')
axs[0,1].imshow(quran_2, cmap='gray')
axs[0,1].set_title('Quran 02')
axs[1,0].imshow(quran_3, cmap='gray')
axs[1,0].set_title('Quran 03')
axs[1,1].imshow(quran_4, cmap='gray')
axs[1,1].set_title('Quran 04')
plt.show()

fig, axs = plt.subplots(2,2, figsize=(10,10))
axs[0,0].hist(quran_1.ravel(),256,[0,255])
axs[0,0].set_title('Quran 01')
axs[0,1].hist(quran_2.ravel(),256,[0,255])
axs[0,1].set_title('Quran 02')
axs[1,0].hist(quran_3.ravel(),256,[0,255])
axs[1,0].set_title('Quran 03')
axs[1,1].hist(quran_4.ravel(),256,[0,255])
axs[1,1].set_title('Quran 04')
plt.show()


# In[49]:


threshold_1 = 125
threshold_2 = 130
threshold_3 = 190
threshold_4 = 70

fig, axs = plt.subplots(2,2, figsize=(20,16))
axs[0,0].imshow(global_thresh(quran_1, threshold_1), cmap='gray')
axs[0,1].imshow(global_thresh(quran_2, threshold_2), cmap='gray')
axs[1,0].imshow(global_thresh(quran_3, threshold_3), cmap='gray')
axs[1,1].imshow(global_thresh(quran_4, threshold_4), cmap='gray')
axs[0,0].set_title('Quran 01')
axs[0,1].set_title('Quran 02')
axs[1,0].set_title('Quran 03')
axs[1,1].set_title('Quran 04')
plt.show()


# # Part b.

# In[19]:


def otsu(img):
    pixel_number = img.shape[0] * img.shape[1]
    mean_weight = 1.0/pixel_number
    his, bins = np.histogram(img, np.array(range(0, 256)))
    final_thresh = -1
    final_value = -1
    for t in bins[1:-1]:
        Wb = np.sum(his[:t]) * mean_weight
        Wf = np.sum(his[t:]) * mean_weight

        mub = np.mean(his[:t])
        muf = np.mean(his[t:])

        value = Wb * Wf * (mub - muf) ** 2

        #print("Wb", Wb, "Wf", Wf)
        #print("t", t, "value", value)

        if value > final_value:
            final_thresh = t
            final_value = value
    final_img = img.copy()
    print("Final Threshold: ", final_thresh)
    final_img[img > final_thresh] = 255
    final_img[img < final_thresh] = 0
    return final_img


# In[50]:


fig, axs = plt.subplots(2,2, figsize=(20,16))
axs[0,0].imshow(otsu(quran_1), cmap='gray')
axs[0,1].imshow(otsu(quran_2), cmap='gray')
axs[1,0].imshow(otsu(quran_3), cmap='gray')
axs[1,1].imshow(otsu(quran_4), cmap='gray')
axs[0,0].set_title('Quran 01')
axs[0,1].set_title('Quran 02')
axs[1,0].set_title('Quran 03')
axs[1,1].set_title('Quran 04')
plt.show()


# # Part c.

# In[22]:


def adaptive_mean_thresh(img, block, const):
    row = np.ceil(img.shape[0]/block).astype(int)
    col = np.ceil(img.shape[1]/block).astype(int)
    new = img.copy()
    for i in range(row):
        for j in range(col):
            
            x = [i*block,(i+1)*block]
            y = [j*block,(j+1)*block]
            
            if x[1]>img.shape[0]:
                x[1]=img.shape[0]
            if y[1]>img.shape[1]:
                y[1]=img.shape[1]
            part = new[x[0]:x[1], y[0]:y[1]]
            thresh = np.mean(part)-const
            part[part >= thresh] = 255
            part[part < thresh] = 0
    return new


# In[51]:


fig, axs = plt.subplots(2,2, figsize=(20,16))
axs[0,0].imshow(adaptive_mean_thresh(quran_1, 16, 105), cmap='gray')
axs[0,1].imshow(adaptive_mean_thresh(quran_2, 16, 110), cmap='gray')
axs[1,0].imshow(adaptive_mean_thresh(quran_3, 16, 90), cmap='gray')
axs[1,1].imshow(adaptive_mean_thresh(quran_4, 16, 40), cmap='gray')
axs[0,0].set_title('Quran 01')
axs[0,1].set_title('Quran 02')
axs[1,0].set_title('Quran 03')
axs[1,1].set_title('Quran 04')
plt.show()


# # Part d.

# In[40]:


def adaptive_gauss_thresh(img, block, const):
    row = np.ceil(img.shape[0]/block).astype(int)
    col = np.ceil(img.shape[1]/block).astype(int)
    new = img.copy()
    gaussian_filter = np.zeros((block,block))
    sigma = block//6
    b = block//2
    for x in range(-b, b):
        for y in range(-b, b):
            x1 = np.sqrt(2*np.pi*(sigma**2))
            x2 = np.exp(-(x**2 + y**2)/(2* sigma**2))
            gaussian_filter[x+b-1, y+b-1] = (1/x1)*x2
    
    for i in range(row):
        for j in range(col):
            
            x = [i*block,(i+1)*block]
            y = [j*block,(j+1)*block]
            
            if x[1]>img.shape[0]:
                x[1]=img.shape[0]
            if y[1]>img.shape[1]:
                y[1]=img.shape[1]
                
            gauss_part = gaussian_filter[0:(x[1]-x[0]),0:(y[1]-y[0])]
            part = new[x[0]:x[1], y[0]:y[1]]
            
            thresh = np.sum(np.multiply(part, gauss_part))/np.sum(gauss_part)-const
            part[part >= thresh] = 255
            part[part < thresh] = 0
    return new


# In[52]:


fig, axs = plt.subplots(2,2, figsize=(20,16))
axs[0,0].imshow(adaptive_gauss_thresh(quran_1, 16, 95), cmap='gray')
axs[0,1].imshow(adaptive_gauss_thresh(quran_2, 16, 100), cmap='gray')
axs[1,0].imshow(adaptive_gauss_thresh(quran_3, 16, 80), cmap='gray')
axs[1,1].imshow(adaptive_gauss_thresh(quran_4, 16, 30), cmap='gray')
axs[0,0].set_title('Quran 01')
axs[0,1].set_title('Quran 02')
axs[1,0].set_title('Quran 03')
axs[1,1].set_title('Quran 04')
plt.show()

