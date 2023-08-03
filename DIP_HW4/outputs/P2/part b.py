
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import pandas as pd

# In[ ]:


images = []

for filename in glob.glob('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW4/inputs/P2/*.png'):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)
"""
plt.imshow(images[1])
points_m = plt.ginput(15)
points_m = np.array(points_m)
plt.imshow(images[0])
points_n = plt.ginput(15)
points_n = np.array(points_n)

data_m = pd.DataFrame(points_m)
data_n = pd.DataFrame(points_n)

data_m.to_csv('points_m.csv',index=False,header=None)
data_n.to_csv('points_n.csv',index=False,header=None)
"""

points_m = np.array(pd.read_csv('points_m.csv', header=None))
points_n = np.array(pd.read_csv('points_n.csv', header=None))

points_m = points_m.astype(int)
points_n = points_n.astype(int)

fig = plt.figure()
plt.subplot(1,2,1)
plt.imshow(images[1])
plt.scatter(points_m[:,0],points_m[:,1])
plt.subplot(1,2,2)
plt.imshow(images[0],)
plt.scatter(points_n[:,0],points_n[:,1])
plt.show()
