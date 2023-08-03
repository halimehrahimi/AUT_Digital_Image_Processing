
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import itertools
import sys
sys.path.append('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW4/outputs/P3/')
from StitchImages import *


# In[9]:


path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW4/'
images = []
for filename in glob.glob(path + 'inputs/P3/I/*.png'):
    img = cv2.imread(filename)
    images.append(img)

output = stitch_image(images)
output = cv2.cvtColor(np.uint8(output), cv2.COLOR_BGR2RGB)
plt.imshow(output)
plt.imsave(path + 'outputs/P3/I/output.jpg', output)
plt.show()


# In[8]:


path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW4/'
images = []
for filename in glob.glob(path + 'inputs/P3/II/*.png'):
    img = cv2.imread(filename)
    images.append(img)

output = stitch_image(images)
output = cv2.cvtColor(np.uint8(output), cv2.COLOR_BGR2RGB)
plt.imshow(output)
plt.imsave(path + 'outputs/P3/II/output.jpg', output)
plt.show()


# In[11]:


path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW4/'
images = []
for filename in glob.glob(path + 'inputs/P3/III/*.png'):
    img = cv2.imread(filename)
    images.append(img)

output = stitch_image(images)
output = cv2.cvtColor(np.uint8(output), cv2.COLOR_BGR2RGB)
plt.imshow(output)
plt.imsave(path + 'outputs/P3/III/output.jpg', output)
plt.show()

