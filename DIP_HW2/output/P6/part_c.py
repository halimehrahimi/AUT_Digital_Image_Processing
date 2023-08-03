
# coding: utf-8

# In[1]:


import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2


# In[2]:


def extract_foreground_mask(foregrounds, numFrames, method):
    trshlist = [70,60,40,40]
    for i in range(len(foregrounds)):
        foregrounds[i][foregrounds[i]>trshlist[i]] = 255
        foregrounds[i][foregrounds[i]<=trshlist[i]] = 0

    plt.figure()
    for i in range(len(foregrounds)):
        plt.imsave('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW2/output/P6/foreground mask/foreground_mask_'
                   +str(numFrames[i])+method+'.png',foregrounds[i], cmap='gray')
        plt.subplot(2, 2, i + 1)
        plt.imshow(foregrounds[i], cmap='gray')
    plt.show()
    return foregrounds


# In[3]:


# import foreground images for average
path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW2/output/P6/foregrounds/'
filenames = glob.glob(path+'*average.png')
filenames.sort()
foregrounds_avg = []
for filename in filenames:
    foreground = cv2.imread(filename)
    foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    foregrounds_avg.append(foreground)
foregrounds_avg = np.array(foregrounds_avg)


# In[4]:


# import foreground images for median
filenames = glob.glob(path+'*median.png')
filenames.sort()
foregrounds_med = []
for filename in filenames:
    foreground = cv2.imread(filename)
    foreground = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    foregrounds_med.append(foreground)
foregrounds_med = np.array(foregrounds_med)


# In[5]:


extract_foreground_mask(foregrounds_avg,[2,5,10,20], method='average')
extract_foreground_mask(foregrounds_med, [2,5,10,20], method='median')

