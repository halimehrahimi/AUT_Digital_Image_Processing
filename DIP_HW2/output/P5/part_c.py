
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import cv2


# In[2]:


def contrast_lim_hist_eq(img, thresh):
    pdf = (np.histogram(img.ravel(), 256, [0,255])[0])
    cdf = np.cumsum(pdf)
    nj = (cdf - cdf.min()) * 255
    N = cdf.max() - cdf.min()
    cdf = nj / N

    extra = 0
    for c in range(cdf.size):
        if cdf[c]>thresh:
            extra += (cdf[c]-thresh)
            cdf[c] = thresh
    cdf += (extra/256)
    
    cdf = cdf.astype('uint8')
    new = cdf[img.ravel()]
    new = new.reshape((img.shape[0], img.shape[1]))
    return new


# In[3]:


def clahe(img, tile, thresh):
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
            new[x[0]:x[1], y[0]:y[1]] = contrast_lim_hist_eq(new[x[0]:x[1], y[0]:y[1]], thresh)
            
    return new


# In[4]:


def plot_clahe(img, tile, thresh, i):
    plt.figure(figsize=(10,10))
    output_c = clahe(img, tile, thresh)
    plt.imsave('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW2/output/P5/clahe_'+str(i)+'.png', output_c,
              cmap = 'gray')
    plt.imshow(output_c, cmap='gray')
    plt.show()


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


plot_clahe(ghost_1, 200, 225, 1)
plot_clahe(ghost_2, 200, 225, 2)
plot_clahe(ghost_3, 200, 225, 3)
plot_clahe(ghost_4, 200, 225, 4)

