
# coding: utf-8

# In[ ]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import pandas as pd
from scipy.spatial import Delaunay


# In[ ]:


images = []

for filename in glob.glob('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW4/inputs/P2/*.png'):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)

points_m = np.array(pd.read_csv('points_m.csv', header=None))
points_n = np.array(pd.read_csv('points_n.csv', header=None))


points_m = points_m.astype(int)
points_n = points_n.astype(int)

tri_m = Delaunay(points_m)
tri_n = Delaunay(points_n)

fig = plt.figure()
plt.subplot(1,2,1)
plt.imshow(images[1])
plt.triplot(points_m[:, 0], points_m[:, 1], tri_m.simplices)
plt.subplot(1,2,2)
plt.imshow(images[0])
plt.triplot(points_n[:, 0], points_n[:, 1], tri_n.simplices)
plt.show()
