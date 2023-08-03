
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import pandas as pd
from scipy.spatial import Delaunay


# In[2]:


def applyAffineTransform(src, srcTri, dstTri, size) :
    
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform( np.float32(srcTri), np.float32(dstTri) )
    
    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine( src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101 )

    return dst


# In[3]:


def morphTriangle(img1, img2, img, t1, t2, t, alpha) :

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))


    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []


    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]),(t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]),(t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]),(t2[i][1] - r2[1])))


    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype = np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] = img[r[1]:r[1]+r[3], r[0]:r[0]+r[2]] * ( 1 - mask ) + imgRect * mask


# In[4]:


def morphImage(img1, img2, points_m, points_n, alpha):
    
    # Calculate the points for the destination image
    points_i = (1-alpha)*points_m + alpha*points_n
    
    # Compute dalaunay triangles for each set of points
    tri_m = Delaunay(points_m)
    tri_n = Delaunay(points_n)
    tri_i = Delaunay(points_i)
    
    # Creating space for destination image
    imgMorph = np.zeros(img1.shape, dtype = img1.dtype)
    
    # Calculate for each triangle
    for i in range(len(tri_i.simplices)) :
        x,y,z = tri_i.simplices[i]

        x = int(x)
        y = int(y)
        z = int(z)

        t1 = [points_m[x], points_m[y], points_m[z]]
        t2 = [points_n[x], points_n[y], points_n[z]]
        t = [points_i[x], points_i[y], points_i[z]]

        morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)

    return np.uint8(imgMorph)


# # Part d

# In[5]:


print('Part d\n')

images = []
path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW4/'
for filename in glob.glob(path+'inputs/P2/*.png'):
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    images.append(img)

points_m = np.array(pd.read_csv(path+'outputs/P2/points_m.csv', header=None))
points_n = np.array(pd.read_csv(path+'outputs/P2/points_n.csv', header=None))

points_m = points_m.astype(int)
points_n = points_n.astype(int)

alpha = np.linspace(start=0.05,stop=0.95,num=10)
print('alpha: ',alpha)
fig = plt.figure(figsize=(10,10))
for a in range(len(alpha)):
    final_image = morphImage(images[1], images[0], points_m, points_n, alpha[a])
    plt.subplot(4,3,a+1)
    plt.title('alpha = '+str(alpha[a]))
    plt.imshow(final_image)#, cmap='gray'
    plt.xticks([])
    plt.yticks([])
plt.show()


# # Part e

# In[6]:


print('Part e\n')
alpha = np.linspace(start=0,stop=1,num=61)
print('alpha:\n',alpha)


# In[7]:


frameSize = (images[1].shape[1], images[1].shape[0])
out = cv2.VideoWriter(path+'outputs/P2/output_video.avi',cv2.VideoWriter_fourcc(*'DIVX'), 30, frameSize)

for a in range(len(alpha)):
    final_image = morphImage(images[1], images[0], points_m, points_n, alpha[a])
    out.write(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))

out.release()

