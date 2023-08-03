import numpy as np
import matplotlib.pyplot as plt
import cv2

def load_pic(path):
    
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

#Read image
path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW1/inputs/P4/wally_2.jpg'
wally2 = load_pic(path)
plt.imshow(wally2)
plt.show()
