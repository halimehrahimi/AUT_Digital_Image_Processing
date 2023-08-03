
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2


# In[2]:


def load_pic(path):
    
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


# # a.

# In[3]:


print('-----------------------------')
print('a.')
patha = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW1/inputs/P1/color_saturation_illusion.png'
pica = load_pic(patha)
plt.imshow(pica)
plt.title('a - not processed')
plt.show()


# In[4]:


pica[pica!=pica[300,300]]=0


# In[5]:


plt.imshow(pica)
plt.title('a - processed')
plt.show()


# # b.

# In[6]:


print('-----------------------------')
print('b.')
pathb = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW1/inputs/P1/gradient_optical_illusion.png'
picb = load_pic(pathb)
plt.imshow(picb)
plt.title('b - not processed')
plt.show()


# In[7]:


picb[picb!=picb[350,500]]=0


# In[8]:


plt.imshow(picb)
plt.title('b - processed')
plt.show()


# # c.

# In[9]:


print('-----------------------------')
print('c.')
pathc = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW1/inputs/P1/ebbinghaus_illusion.png'
picc = load_pic(pathc)
plt.imshow(picc)
plt.title('c - not processed')
plt.show()


# In[10]:


picc[picc!=picc[200,550]]=0


# In[11]:


plt.imshow(picc)
plt.title('c - processed')
plt.show()


# In[34]:


counts1 = np.where(picc[:,:350,:].reshape(-1,3)==picc[200,550])[0].size
print('Circle on the left pixel counts: ', counts1)
counts2 = np.where(picc[:,350:,:].reshape(-1,3)==picc[200,550])[0].size
print('Circle on the right pixel counts: ', counts2)


# # d.

# In[12]:


print('-----------------------------')
print('d.')
pathd = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW1/inputs/P1/checker_shadow_illusion.png'
picd = load_pic(pathd)
plt.imshow(picd)
plt.title('d - not processed')
plt.show()


# In[13]:


picd[picd!=picd[250,500]]=0


# In[14]:


plt.imshow(picd)
plt.title('d - processed')
plt.show()

