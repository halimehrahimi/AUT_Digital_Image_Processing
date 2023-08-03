
# coding: utf-8

# In[1]:


import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2


# In[2]:


def extract_foreground(backgrounds, test, numFrames, method):
    foregrounds = []
    for i in range(len(backgrounds)):
        fore = cv2.absdiff(test, backgrounds[i])
        foregrounds.append(fore)

    plt.figure()
    for i in range(len(foregrounds)):
        plt.imsave('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW2/output/P6/foregrounds/foreground_'
                   +str(numFrames[i])+method+'.png',foregrounds[i])
        plt.subplot(2, 2, i + 1)
        plt.imshow(foregrounds[i])
    plt.show()
    return foregrounds


# In[3]:


# import background images for average
path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW2/output/P6/backgrounds/'
filenames = glob.glob(path+'*average.png')
filenames.sort()
backgrounds_avg = []
for filename in filenames:
    background = cv2.imread(filename)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    backgrounds_avg.append(background)
backgrounds_avg = np.array(backgrounds_avg)


# In[4]:


# import background images for median
filenames = glob.glob(path+'*median.png')
filenames.sort()
backgrounds_med = []
for filename in filenames:
    background = cv2.imread(filename)
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    backgrounds_med.append(background)
backgrounds_med = np.array(backgrounds_med)


# In[5]:


# import test frame
path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW2/inputs/P6/'
test = cv2.imread(path+'test/football_test.png')
test = cv2.cvtColor(test, cv2.COLOR_BGR2RGB)
foregrounds_average = extract_foreground(backgrounds_avg,test,[2,5,10,20],'average')
foregrounds_median = extract_foreground(backgrounds_med,test,[2,5,10,20],'median')

