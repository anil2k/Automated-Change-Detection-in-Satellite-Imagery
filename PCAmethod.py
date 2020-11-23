import cv2 
import sklearn
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.decomposition import PCA
import skimage.morphology
import numpy as np
import time

image1_path ="Pre.jpg"
image2_path = "Post.jpg"

# %%

def find_vector_set(diff_image, new_size):
 
    i = 0
    j = 0
    vector_set = np.zeros((int(new_size[0] * new_size[1] / 25),25))
    while i < vector_set.shape[0]:
        while j < new_size[1]:
            k = 0
            while k < new_size[0]:
                block   = diff_image[j:j+5, k:k+5]
                feature = block.ravel()
                vector_set[i, :] = feature
                k = k + 5
            j = j + 5
        i = i + 1
 
    mean_vec   = np.mean(vector_set, axis = 0)
    # Mean normalization
    vector_set = vector_set - mean_vec   
    return vector_set, mean_vec

def find_FVS(EVS, diff_image, mean_vec, new):
 
    i = 2
    feature_vector_set = []
 
    while i < new[1] - 2:
        j = 2
        while j < new[0] - 2:
            block = diff_image[i-2:i+3, j-2:j+3]
            feature = block.flatten()
            feature_vector_set.append(feature)
            j = j+1
        i = i+1
 
    FVS = np.dot(feature_vector_set, EVS)
    FVS = FVS - mean_vec
    print ("[INFO] Feature vector space size", FVS.shape)
    return FVS

def clustering(FVS, components, new):
    kmeans = KMeans(components, verbose = 0)
    kmeans.fit(FVS)
    output = kmeans.predict(FVS)
    count  = Counter(output)
 
    least_index = min(count, key = count.get)
    change_map  = np.reshape(output,(new[1] - 4, new[0] - 4))
    return least_index, change_map
    

# Read Images


image1 = cv2.imread(image1_path)
image2 = cv2.imread(image2_path)




# Resize Images
new_size = np.asarray(image1.shape) /5
new_size = new_size.astype(int) *5
image1 = cv2.resize(image1, (new_size[0],new_size[1])).astype(int)
image2 = cv2.resize(image2, (new_size[0],new_size[1])).astype(int)


# Difference Image

start = time.time()
diff_image = abs(image1 - image2)
cv2.imwrite('difference.jpg', diff_image)
end = time.time()

diff_image=diff_image[:,:,1]





pca = PCA()
vector_set, mean_vec=find_vector_set(diff_image, new_size)
pca.fit(vector_set)
EVS = pca.components_



FVS = find_FVS(EVS, diff_image, mean_vec, new_size)
components = 3




start = time.time()
least_index, change_map = clustering(FVS, components, new_size)
end = time.time()


change_map[change_map == least_index] = 255
change_map[change_map != 255] = 0
change_map = change_map.astype(np.uint8)


cv2.imwrite('ChangeMap.jpg', change_map)


kernel = skimage.morphology.disk(6)
CloseMap = cv2.morphologyEx(change_map, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('CloseMap.jpg', CloseMap)


OpenMap = cv2.morphologyEx(CloseMap, cv2.MORPH_OPEN, kernel)
cv2.imwrite('OpenMap.jpg', OpenMap)
print('finish')