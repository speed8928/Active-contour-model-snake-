import cv2
import sys 
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import data
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import PIL.ImageDraw as ImageDraw
from Watershed import Watershed



if len(sys.argv) > 1:
    file_to_load = sys.argv[1]

img = cv2.imread( file_to_load, cv2.IMREAD_COLOR )
img = rgb2gray(img)
s = np.linspace(0, 2*np.pi, 180)

x = 331 + 20*np.cos(s)
y = 353 + 18*np.sin(s)
init = np.array([x, y]).T


e1x = np.linspace(317, 339, 40)
e1y = np.linspace(343, 343, 40)
edge1 = np.array([e1x,e1y]).T
e2x = np.linspace(339, 339 , 40)
e2y = np.linspace(343, 363 , 40)
edge2 = np.array([e2x,e2y]).T
e3x = np.linspace(339, 320 , 40)
e3y = np.linspace(368, 368 , 40)
edge3 = np.array([e3x,e3y]).T
e4x = np.linspace(320, 317 , 40)
e4y = np.linspace(368, 343 , 40)
edge4 = np.array([e4x,e4y]).T

edge = np.vstack((edge1,edge2))
edge = np.vstack((edge,edge3))
edge = np.vstack((edge,edge4))


building1 = edge + 5
building2 = np.array([edge[:,0],edge[:,1]-25]).T
building3 = np.array([edge[:,0],edge[:,1]-55]).T


snake = active_contour(gaussian(img, sigma=0.5), building1, alpha=0.03, beta=0.3, gamma=0.0001, w_line = -0.1)
snake_b2 = active_contour(gaussian(img, sigma=0.5), building2, alpha=0.03, beta=0.3, gamma=0.0001, w_line = -0.1)
snake_b3 = active_contour(gaussian(img, sigma=0.5), building3, alpha=0.03, beta=0.3, gamma=0.0001, w_line = -0.1)
# alpha : float, optional

#     Snake length shape parameter. Higher values makes snake contract faster.
# beta : float, optional

#     Snake smoothness shape parameter. Higher values makes snake smoother.
# gamma : float, optional

#     Explicit time stepping parameter.

# w_line : float, optional
#     Controls attraction to brightness. Use negative values to attract toward dark regions.



snake2 = active_contour(gaussian(img, 2),snake, alpha=0.015, beta=0.7, gamma=0.001)
snake2_b2 = active_contour(gaussian(img, 2),snake_b2, alpha=0.015, beta=0.7, gamma=0.001)
snake2_b3 = active_contour(gaussian(img, 2),snake_b3, alpha=0.015, beta=0.7, gamma=0.001)

fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
ax.plot(building1[:, 0], building1[:, 1], '-g', lw=1)
ax.plot(building2[:, 0], building2[:, 1], '-g', lw=1)
ax.plot(building3[:, 0], building3[:, 1], '-g', lw=1)
ax.plot(snake[:, 0], snake[:, 1], '--r', lw=2)
ax.plot(snake2[:, 0], snake2[:, 1], '-b', lw=2)
ax.plot(snake_b2[:, 0], snake_b2[:, 1], '--r', lw=2)
ax.plot(snake2_b2[:, 0], snake2_b2[:, 1], '-b', lw=2)
ax.plot(snake_b3[:, 0], snake_b3[:, 1], '--r', lw=2)
ax.plot(snake2_b3[:, 0], snake2_b3[:, 1], '-b', lw=2)

ax.set_xticks([]), ax.set_yticks([])
ax.axis([0, img.shape[1], img.shape[0], 0])
plt.show()


# if len(sys.argv) > 1:
#    file_to_load = sys.argv[1]#
# img = cv2.imread( file_to_load, cv2.IMREAD_COLOR )
# img = rgb2gray(img)
# img = img * 255;#
# ret,thresh3 = cv2.threshold(img,150,200,cv2.THRESH_TRUNC)#
# plt.imshow(thresh3,'gray')#
# plt.show()

# Step 1 :: convert image to gray scale and threshold to certain range and use Gaussian filter out the noise
# Step 2 :: drawing the points that retrieve from openstreet map as building's shape prior
# Step 3 :: use the shape prior to create more data points so it can be deformed
# Step 4 :: fitting those data point to active model(snake)
# Step 5 :: Iterative the deforming shape process until converge or we can set the external contraint to control the deformation.