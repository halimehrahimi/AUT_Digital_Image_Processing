
# coding: utf-8

# In[1]:


import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2


# In[2]:


def extract_background(imgs, numFrames, method):
    backgrounds = []
    for n in numFrames:
        if method == 'average':
            backgrounds.append(np.average(imgs[:n], axis=0).astype('uint8'))
        else:
            backgrounds.append(np.median(imgs[:n], axis=0).astype('uint8'))
    plt.figure()
    for i in range(len(backgrounds)):
        plt.imsave('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW2/output/P6/backgrounds/background_n'
                   +str(numFrames[i])+method+'.png',backgrounds[i])
        plt.subplot(2, 2, i + 1)
        plt.imshow(backgrounds[i])
    plt.show()
    return backgrounds


# In[3]:


path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW2/inputs/P6/'
filenames = glob.glob(path+'frames/*.png')
filenames.sort()
imgs = []
for filename in filenames:
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    imgs.append(img)
imgs = np.array(imgs)
backgrounds_average = extract_background(imgs,[2,5,10,20],method='average')
backgrounds_median = extract_background(imgs,[2,5,10,20],method='median')

