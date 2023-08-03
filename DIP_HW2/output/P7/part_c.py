
# coding: utf-8

# In[1]:


path = 'F://Uni/992/Digital Image Processing/Homeworks/DIP_HW2/'
import sys
sys.path.append(path+'output/P7')
from SeamsResize import *


# In[2]:


def resize(img, percentages, direction, name):
    # number of seams
    seem_nums = (img.shape[1] * percentages).astype(int)

    # find energy_map and plot it
    energy_map = find_energy_map(img)
    plt.figure(figsize=(8,8))
    plt.imshow(energy_map, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.imsave(path+'output/P7/'+name+'_energy_map_'+direction+'.png', energy_map, cmap='gray')
    plt.show()

    # we compute the seems until the largest amount and then start deleting
    seams_list, index_map = find_seams_list(energy_map,direction,np.max(seem_nums))

    for i in range(len(seem_nums)):

        if direction == 'horizontal':
            seams_list_new = seams_list[:seem_nums[i],:]
        else:
            seams_list_new = seams_list[:,:seem_nums[i]]

        new_index_map = index_map.copy()
        new_index_map[new_index_map>seem_nums[i]] = 0
        resized_image,seems_image = remove_seams(img,seams_list_new,direction,new_index_map)
        plt.figure(figsize=(8,8))
        plt.imshow(seems_image)
        plt.xticks([])
        plt.yticks([])
        plt.title(f'percentage = {percentages[i]}, dir={dir}')
        plt.imsave(path+f'output/P7/'+name+'_'+str(percentages[i])+'_'+direction+'_seams.png', seems_image)
        plt.show()

        plt.figure(figsize=(8,8))
        plt.imshow(resized_image.astype('uint8'))
        plt.xticks([])
        plt.yticks([])
        plt.title(f'percentage = {percentages[i]}, dir={dir}')
        plt.imsave(path+f'output/P7/'+name+'_'+str(percentages[i])+'_'+direction+'_resized.png', resized_image.astype('uint8'))
        plt.show()


# In[3]:


luci = cv2.imread(path+'output/P7/Lucifer.jpeg')
luci = cv2.cvtColor(luci, cv2.COLOR_BGR2RGB)

resize(luci, np.array([0.1,0.25,0.5]), 'vertical', 'Lucifer')

