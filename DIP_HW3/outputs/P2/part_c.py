
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2


# In[2]:


path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/'
childhood = cv2.imread(path + 'inputs/P2/donald_childhood.png')


# In[3]:


def convolve2d(image, kernel):
    
    # Flip the kernel
    kernel = np.flipud(np.fliplr(kernel))
    # convolution output
    output = np.zeros_like(image)

    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 4, image.shape[1] + 4, 3))
    image_padded[2:-2, 2:-2] = image

    # Loop over every pixel of the image
    for d in range(3):
        for x in range(image.shape[1]):
            for y in range(image.shape[0]):
                # element-wise multiplication of the kernel and the image
                output[y, x, d] = (kernel * image_padded[y: y+5, x: x+5, d]).sum()

    return output


# In[4]:


kernel = np.ones((5,5),np.float32)/25
dst = convolve2d(childhood,kernel)
plt.figure(figsize=(8,8))
plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.xticks([])
plt.yticks([])
plt.savefig(path + 'outputs/P2/childhood_spatial.jpg')
plt.show()


# In[5]:


graduation = cv2.imread(path + 'inputs/P2/donald_graduation.png')


# In[6]:


dst = convolve2d(graduation,kernel)
plt.figure(figsize=(8,8))
plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
plt.xticks([])
plt.yticks([])
plt.savefig(path + 'outputs/P2/graduation_spatial.jpg')
plt.show()

