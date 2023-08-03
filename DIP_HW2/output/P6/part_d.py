
# coding: utf-8

# In[18]:


import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2


# In[19]:


def extract_foreground_players(test, masks, numFrames, method):
    imgs = []
    for i in range(len(masks)):
        img =np.copy(test)
        img[masks[i] == 0]= 0
        imgs.append(img)

    plt.figure()
    for i in range(len(imgs)):
        plt.imsave('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW2/output/P6/foreground players/foreground_players_'
                   +str(numFrames[i])+method+'.png',imgs[i])
        plt.subplot(2, 2, i + 1)
        plt.imshow(imgs[i])
    plt.show()
    return imgs


# In[20]:


# import masks for average
path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW2/output/P6/foreground mask/'
filenames = glob.glob(path+'*average.png')
filenames.sort()
mask_avg = []
for filename in filenames:
    mask = cv2.imread(filename)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_avg.append(mask)
mask_avg = np.array(mask_avg)


# In[21]:


# import masks for median
filenames = glob.glob(path+'*median.png')
filenames.sort()
mask_med = []
for filename in filenames:
    mask = cv2.imread(filename)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask_med.append(mask)
mask_med = np.array(mask_med)


# In[22]:


# import test frame
path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW2/inputs/P6/'
test = cv2.imread(path+'test/football_test.png')
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)


# In[23]:


extract_foreground_players(test, mask_avg, [2,5,10,20], 'average')
extract_foreground_players(test, mask_med, [2,5,10,20], 'median')

