
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from sklearn.metrics.pairwise import euclidean_distances


# # Functions

# In[ ]:


def load_pic(path):
    
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image


# In[ ]:


def keep_red(img):
    pic = img.reshape(img.shape[0]*img.shape[1],3)
    keep = np.zeros(pic.shape[0])
    keep[np.where(euclidean_distances(pic, np.expand_dims(red, axis=0))<=36)[0]] = 1
    keep = keep.reshape(img.shape[0],img.shape[1])
    numbers = np.where(keep != 0)
    location_red = np.array([numbers[0], numbers[1]]).T
    
    return location_red


# In[ ]:


def keep_white(img):
    pic = img.reshape(img.shape[0]*img.shape[1],3)
    keep = np.zeros(pic.shape[0])
    keep[np.where(euclidean_distances(pic, np.expand_dims(white, axis=0))<=8)[0]] = 1
    keep = keep.reshape(img.shape[0],img.shape[1])
    numbers = np.where(keep != 0)
    location_white = np.array([numbers[0], numbers[1]]).T
    
    return location_white


# In[ ]:


def ncc(a,b):
    try:
        return(np.sum(np.corrcoef(a.ravel(),b.ravel())))
    except:
        return 100000

def ncc_similarity(a, b, location, h, w, lower, upper):
    min_ncc = []
    output = []
    for l in location:
        nccDiff = ncc(a, b[l[0]-h:l[0]+h,l[1]-w:l[1]+w])
        
        if nccDiff > lower and nccDiff < upper:
            min_ncc.append(nccDiff)
            output.append(l)
    return output, min_ncc


# In[ ]:


def find_wally(template, img, percent, lower, upper):
    numbers = np.where(np.all(img != [0,0,0], axis=-1))
    location = np.array([numbers[0], numbers[1]]).T
    
    # resize image if necessary
    scale_percent = percent
    width = int(template.shape[1] * scale_percent / 100)
    height = int(template.shape[0] * scale_percent / 100)
    dim = (width, height)
    template_re = cv2.resize(template, dim, interpolation = cv2.INTER_AREA)
    
    h = math.ceil(height/2)
    w = math.ceil(width/2)
    
    out, minncc = ncc_similarity(template_re, img, location, h, w, lower, upper)
    
    return out,minncc,h,w


# In[ ]:


def wheres_wally(template, img, percent, lower, upper):
    location_red = keep_red(img)
    new = np.zeros(img.shape)
    for l in location_red:
        new[l[0]-40:l[0]+40,l[1]-40:l[1]+40] = img[l[0]-40:l[0]+40,l[1]-40:l[1]+40]
    #fig = plt.figure(figsize=(20,12))
    new = new.astype('uint8')
    #plt.imshow(new)
    #plt.show()
    
    location_white = keep_white(new)
    new2 = np.zeros(img.shape)
    for l in location_white:
        new2[l[0]-40:l[0]+40,l[1]-40:l[1]+40] = new[l[0]-40:l[0]+40,l[1]-40:l[1]+40]
    new2 = new2.astype('uint8')
    fig = plt.figure(figsize=(20,12))
    plt.imshow(new2)
    plt.show()
    
    wallyloc, minncc, wallyh, wallyw = find_wally(template, new2, percent, lower, upper)
    
    for l in wallyloc:
        cv2.rectangle(img, (l[1]-wallyw, l[0]-wallyh), (l[1]+wallyw, l[0]+wallyh), (255, 0, 0), 3)

    fig = plt.figure(figsize = (10,20))
    plt.imshow(img)
    plt.show()
    return wallyloc,minncc


# # Initialization

# In[ ]:


path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW1/inputs/P4/'

wally1 = load_pic(path+'wally_1.jpg')
#plt.imshow(wally1)
red1 = np.array([wally1[100,20]])
white1 = np.array([wally1[90,10]])

wally2 = load_pic(path+'wally_2.jpg')
#plt.imshow(wally2)
red2 = np.array([wally2[100,10]])
white2 = np.array([wally2[90,10]])

wally3 = load_pic(path+'wally_3.jpg')
#plt.imshow(wally3)
red3 = np.array([wally3[120,10]])
white3 = np.array([wally3[95,15]])

wally4 = load_pic(path+'wally_4.jpg')
#plt.imshow(wally4)
red4 = np.array([wally4[95,15]])
white4 = np.array([wally4[105,15]])

wally5 = load_pic(path+'wally_5.jpg')
#plt.imshow(wally5)
red5 = np.array([wally5[78,15]])
white5 = np.array([wally5[59,17]])

where1 = load_pic(path + 'wheres_wally_1.jpg')
#plt.imshow(where1)
where2 = load_pic(path + 'wheres_wally_2.jpg')
#plt.imshow(where1)
where3 = load_pic(path + 'wheres_wally_3.jpg')
#plt.imshow(where1)
where4 = load_pic(path + 'wheres_wally_4.jpg')
#plt.imshow(where1)


# In[ ]:


redmat = np.concatenate((red1,red2,red3,red4,red5), axis=0)
whitemat = np.concatenate((white1,white2,white3,white4,white5), axis=0)
red = redmat.mean(axis=0).astype(int)
white = whitemat.mean(axis=0).astype(int)
colors = np.array([red,white])
print('Red: ', red)
print('White: ', white)


# # Where's Wally image 1

# In[ ]:


wallyloc, minncc = wheres_wally(wally2, where1, 50, 2.47874, 2.47875)


# # Where's Wally image 2

# In[ ]:


wallyloc, minncc = wheres_wally(wally2, where2, 80, 2.23547426, 2.23547427)


# # Where's Wally image 3

# In[ ]:


wallyloc, minncc = wheres_wally(wally2, where3, 90, 2.167502, 2.167503)


# # Where's Wally image 4

# In[ ]:


wallyloc, minncc = wheres_wally(wally2, where4, 80, 2.46356, 2.46357)

