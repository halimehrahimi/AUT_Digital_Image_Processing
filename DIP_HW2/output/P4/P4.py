
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt


# # Part a.

# In[2]:


def bitplane_slice(img, plot=False):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bit_slices = []
    for k in range(8):
        bit_slices.append((gray // 2 ** k) % 2)
    if plot:
        fig = plt.figure(figsize = (20,5))
        fig.suptitle('bitplane_slice')
        for i in range(8):
            plt.subplot(2, 4, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(bit_slices[i],cmap='gray')
            plt.xlabel('k = '+str(i))
        plt.show()
    return np.array(bit_slices)


# In[3]:


path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW2/inputs/P4/'
highway_1 = cv2.imread(path + 'highway_1.png')
bit_highway_1 = bitplane_slice(highway_1, True)
highway_2 = cv2.imread(path + 'highway_2.png')
bit_highway_2 = bitplane_slice(highway_2, True)
pavement_1 = cv2.imread(path + 'pavement_1.png')
bit_pavement_1 = bitplane_slice(pavement_1, True)
pavement_2 = cv2.imread(path + 'pavement_2.png')
bit_pavement_2 = bitplane_slice(pavement_2, True)


# # Part b.

# In[4]:


def xor(img1, img2, plot=False):
    bit_slices_img1 = bitplane_slice(img1)
    bit_slices_img2 = bitplane_slice(img2)
    bit_slices_xor = np.logical_xor(bit_slices_img1, bit_slices_img2)
    if plot:
        fig = plt.figure(figsize = (20,5))
        fig.suptitle('xor')
        for i in range(8):
            plt.subplot(2, 4, i + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(bit_slices_xor[i], cmap='gray')
            plt.xlabel('k = '+str(i))
        plt.show()
    return bit_slices_xor


# In[5]:


xor_highway = xor(highway_1, highway_2, True)
xor_pavement = xor(pavement_1, pavement_2, True)


# # Part c.

# In[19]:


def moving_regions(img1, img2, plot = False):
    bit_slices_xor = xor(img1, img2)
    result = np.sum(np.array([bit_slices_xor[k] * 2 ** k for k in range(4,8)]),axis=0)

    if plot:
        plt.title('moving region')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(result, cmap='gray')
        plt.show()
    return result


# In[20]:


mr_highway = moving_regions(highway_1, highway_2, True)
mr_pavement = moving_regions(pavement_1, pavement_2, True)


# # Part d.

# In[21]:


def enhancement(img):
    #median filtering to blur the details
    box = cv2.medianBlur(img.astype('float32'), 5)
    #thresholding will create more contrast
    _, trh = cv2.threshold(box.astype('float32'), 40, 255, cv2.THRESH_BINARY)
    plt.title('enhanced')
    plt.xticks([])
    plt.yticks([])
    plt.imshow(trh, cmap="gray")
    plt.show()


# In[22]:


enhanced_highway = enhancement(mr_highway)
enhanced_pavement = enhancement(mr_pavement)

