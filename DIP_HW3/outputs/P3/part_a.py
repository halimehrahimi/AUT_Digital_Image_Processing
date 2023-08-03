import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.insert(0, 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/inputs/P3')
from align_imgs import *


path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/inputs/P3/Images/'
img_1 = cv2.imread(path + 'donald_1.png')
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/inputs/P3/Images/'
img_2 = cv2.imread(path + 'donald_6.png')
img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

img_1_aligned, img_2_aligned = align_imgs(img_1, img_2)
plt.imsave('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/outputs/P3/donald_1_aligned.png',img_1_aligned,cmap='gray')
plt.imshow(img_1_aligned, cmap='gray')
plt.show()
plt.imsave('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/outputs/P3/donald_6_aligned.png',img_2_aligned,cmap='gray')
plt.imshow(img_2_aligned, cmap='gray')
plt.show()

img_fft = np.fft.fftshift(np.fft.fft2(img_1_aligned))
plt.imsave('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/outputs/P3/donald_1_fft.png',np.log10(np.abs(img_fft)/1000+1),cmap='gray')
img_fft = np.fft.fftshift(np.fft.fft2(img_2_aligned))
plt.imsave('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/outputs/P3/donald_6_fft.png',np.log10(np.abs(img_fft)/1000+1),cmap='gray')
#*******************************

path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/inputs/P3/Images/'
img_1 = cv2.imread(path + 'joe_4.png')
img_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/inputs/P3/Images/'
img_2 = cv2.imread(path + 'joe_6.png')
img_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

img_1_aligned, img_2_aligned = align_imgs(img_1, img_2)
plt.imsave('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/outputs/P3/joe_4_aligned.png',img_1_aligned,cmap='gray')
plt.imshow(img_1_aligned, cmap='gray')
plt.show()
plt.imsave('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/outputs/P3/joe_6_aligned.png',img_2_aligned,cmap='gray')
plt.imshow(img_2_aligned, cmap='gray')
plt.show()

img_fft = np.fft.fftshift(np.fft.fft2(img_1_aligned))
plt.imsave('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/outputs/P3/joe_4_fft.png',np.log10(np.abs(img_fft)/1000+1),cmap='gray')
img_fft = np.fft.fftshift(np.fft.fft2(img_2_aligned))
plt.imsave('F://Uni/992/Digital Image Processing/Homeworks/DIP_HW3/outputs/P3/joe_6_fft.png',np.log10(np.abs(img_fft)/1000+1),cmap='gray')
