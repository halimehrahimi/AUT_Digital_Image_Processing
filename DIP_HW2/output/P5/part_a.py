
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2


# In[2]:


def hist_eq(img):
    
    #cdf needed to get normalized.. the usual dividing over number of pixels didn't go well
    pdf = (np.histogram(img.ravel(), 256, [0,255])[0])
    cdf = np.cumsum(pdf)
    #Normalize cdf
    nj = (cdf - cdf.min()) * 255
    N = cdf.max() - cdf.min()
    cdf = nj / N
    
    cdf = cdf.astype('uint8')
    new = cdf[img.ravel()]
    new = new.reshape((img.shape[0], img.shape[1]))
    
    return new


# In[3]:


def adaptive_hist_eq(img, tile):
    row = np.ceil(img.shape[0]/tile).astype(int)
    col = np.ceil(img.shape[1]/tile).astype(int)
    new = img.copy()
    for i in range(row):
        for j in range(col):
            
            x = [i*tile,(i+1)*tile]
            y = [j*tile,(j+1)*tile]
            
            if x[1]>new.shape[0]:
                x[1]=new.shape[0]
            if y[1]>new.shape[1]:
                y[1]=new.shape[1]
            new[x[0]:x[1], y[0]:y[1]] = hist_eq(new[x[0]:x[1], y[0]:y[1]])
            
    return new


# In[4]:


def hist_eq_plot(img,i):
    output_a = adaptive_hist_eq(img, 200)
    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])
    plt.imsave('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW2/output/P5/hist_eq_'+str(i)+'.png', output_a,
              cmap = 'gray')
    plt.title('Histogram Equalization')
    plt.imshow(output_a, cmap='gray')
    plt.show()
    return output_a


# In[5]:


path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW2/inputs/P5/'

ghost_1 = cv2.imread(path + 'ghost_1.png')
ghost_1 = cv2.cvtColor(ghost_1, cv2.COLOR_BGR2GRAY)

ghost_2 = cv2.imread(path + 'ghost_2.png')
ghost_2 = cv2.cvtColor(ghost_2, cv2.COLOR_BGR2GRAY)

ghost_3 = cv2.imread(path + 'ghost_3.png')
ghost_3 = cv2.cvtColor(ghost_3, cv2.COLOR_BGR2GRAY)

ghost_4 = cv2.imread(path + 'ghost_4.png')
ghost_4 = cv2.cvtColor(ghost_4, cv2.COLOR_BGR2GRAY)


# In[7]:


hist_eq_plot(ghost_1,1)
hist_eq_plot(ghost_2,2)
hist_eq_plot(ghost_3,3)
hist_eq_plot(ghost_4,4)

