import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread("RGBD_dataset/467.png",cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("RGBD_dataset/472.png",cv2.IMREAD_GRAYSCALE)
depth_img1 = cv2.imread("RGBD_dataset/467_D.png",cv2.IMREAD_GRAYSCALE)
depth_img2 = cv2.imread("RGBD_dataset/472_D.png",cv2.IMREAD_GRAYSCALE)

# We are finding the keypoints and feature descriptors of both the images.
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

# WE are then matching the feature descriptors by using BFMatcher Object.
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)

#img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None, flags=2)

# plt.imshow(img3),plt.show()
# print(len(matches))

# Finding the 3d points of the matches of image 1
p_matrix = np.zeros((3,len(matches)))

# This will give points x and y
for i in range(len(matches)):
    pt_i=kp1[matches[i].queryIdx].pt
    p_matrix[:2,i] = np.asarray(pt_i).transpose()
p_matrix = p_matrix.astype(np.uint16)

# This will give the 3rd coordinate (depth)
for i in range(len(matches)):
    p_matrix[2,i] = depth_img1[p_matrix[1, i], p_matrix[0, i]]

# Finding the 3d points of the matches of image 1
q_matrix = np.zeros((3, len(matches)))

# This will give points x and y
for i in range(len(matches)):
    pt_i = kp2[matches[i].trainIdx].pt
    q_matrix[:2, i] = np.asarray(pt_i).transpose()
q_matrix = q_matrix.astype(np.uint16)

# This will give the 3rd coordinate (depth)
for i in range(len(matches)):
    q_matrix[2, i] = depth_img2[q_matrix[1, i], q_matrix[0, i]]

# Finding the weight matrix using Gaussian of the distance between the matches
weight_matrix = np.zeros((len(matches),len(matches)))
sigma = 1.5 # Sigma value used in the gaussian expression
for i in range(len(matches)):
    #weight_matrix[i,i]= 1/(np.linalg.norm(des1[matches[i].queryIdx]-des2[matches[i].trainIdx])+ 0.0000001)
    #weight_matrix[i,i] = 1/abs(matches[i].distance+0.0000000001)
    weight_matrix[i, i] = np.exp((-np.power((matches[i].distance),2))/2*np.power((sigma),2))

# Finding the weighted mean of p_i's
p_weighted = np.average(p_matrix,axis = 1,weights=np.diag(weight_matrix))
p_weighted = p_weighted.astype(np.uint16)
p_weighted = p_weighted.reshape((3,1))

# Finding the weighted mean of p_i's
q_weighted = np.average(q_matrix,axis = 1,weights=np.diag(weight_matrix))
q_weighted = q_weighted.astype(np.uint16)
q_weighted = q_weighted.reshape((3,1))

# print(p_weighted)
# print(q_weighted)

# Finding X matrix from p_i's and its weighted mean
x_matrix = p_matrix - np.repeat(p_weighted,len(matches),axis = 1)

# Finding Y matrix from q_i's and its weighted mean
y_matrix = q_matrix - np.repeat(q_weighted,len(matches),axis = 1)

# Finding the S matrix by XWY'
s_matrix = np.dot(x_matrix,np.dot(weight_matrix,y_matrix.transpose()))
# print(s_matrix.shape)

# SVD of S matrix
U,s,Vt = np.linalg.svd(s_matrix)

# Finding Optimal R
R_optimal = np.dot(Vt.transpose(), U.transpose())
print(R_optimal)

# Finding Optimal t from the above found Optimal R
t_optimal= q_weighted - np.dot(R_optimal, p_weighted)
print(t_optimal)

# Finding the transformed image using the found Rotaion matrix and the translation vector
new_image = np.zeros((img1.shape[0],img1.shape[1]))

for k in range(img1.shape[0]):
    for l in range(img1.shape[1]):
        p_i = np.array([l,k,depth_img1[k,l]])
        rotated = np.dot(R_optimal,np.reshape(p_i,(3,1))) + t_optimal
        #print(rotated.shape)
        #rotated = abs(rotated).astype(np.uint16)
        #print(rotated)
        if rotated[0]>=0 and rotated[0]<img1.shape[1] and rotated[1]>=0 and rotated[1]<img1.shape[0]:
            rotated = rotated.astype(np.uint16)
            new_image[rotated[1],rotated[0]] = img1[k,l]

new_image = new_image.astype(np.uint8)

#print(new_image)
#cv2.imwrite("467to480.jpg",new_image)
cv2.imshow('image',new_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

