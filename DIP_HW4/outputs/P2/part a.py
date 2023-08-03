
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob


# In[5]:


images = []
fig = plt.figure(figsize=(10,8))
i=1
for filename in glob.glob('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW4/inputs/P2/*.png'):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)
    plt.subplot(1,2,i)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    i+=1
plt.show()


# In[22]:


alpha = np.linspace(start=0.05,stop=0.95,num=10)
print(alpha)
fig = plt.figure(figsize=(10,10))
for a in range(len(alpha)):
    output = np.uint8((1-alpha[a])*images[1]+alpha[a]*images[0])
    plt.subplot(4,3,a+1)
    plt.title('alpha = '+str(alpha[a]))
    plt.imshow(output)#, cmap='gray'
    plt.xticks([])
    plt.yticks([])
plt.show()

