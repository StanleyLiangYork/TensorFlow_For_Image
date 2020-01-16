import cv2
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt


i = misc.ascent() # get a sample grayscale image
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i)
plt.show()
i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]
print("the image size is: {0} x {1}".format(size_x,size_y))

# set a 3x3 convolution filter
# set CNN filter to detect edges that only passes through sharp edges and straight lines

# filter = [ [0, 1, 0], [1, -4, 1], [0, 1, 0]]
filter = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]]
# filter = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]

# set CNN filter to sharpen
# filter = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]

# perform convolution
# the current pixel's neighbor above it and to the left will be multiplied by the top left item in the filter etc. etc.
# We'll then multiply the result by the weight, and then ensure the result is in the range 0-255

# note to skip the most-left and most-right columns
# set initial weight to 1, use a 3x3 convolution filter
weight = 1
for x in range(1,size_x-1):
    for y in range(1, size_y - 1):
        convolution = 0.0
        convolution = convolution + (i[x - 1, y - 1] * filter[0][0])
        convolution = convolution + (i[x, y - 1] * filter[0][1])
        convolution = convolution + (i[x + 1, y - 1] * filter[0][2])
        convolution = convolution + (i[x - 1, y] * filter[1][0])
        convolution = convolution + (i[x, y] * filter[1][1])
        convolution = convolution + (i[x + 1, y] * filter[1][2])
        convolution = convolution + (i[x - 1, y + 1] * filter[2][0])
        convolution = convolution + (i[x, y + 1] * filter[2][1])
        convolution = convolution + (i[x + 1, y + 1] * filter[2][2])
        convolution = convolution * weight
        if (convolution < 0):
            convolution = 0
        if (convolution > 255):
            convolution = 255
        i_transformed[x,y] = convolution

# Plot the image. Note the size of the axes -- they are 512 by 512
plt.gray()
plt.grid(False)
plt.imshow(i_transformed)
#plt.axis('off')
plt.show()

# do a 2-to-2 pooling
new_x = int(size_x/2)
new_y = int(size_y/2)
newImage = np.zeros((new_x, new_y))
for x in range(0, size_x, 2):
  for y in range(0, size_y, 2):
    pixels = []
    pixels.append(i_transformed[x, y])
    pixels.append(i_transformed[x+1, y])
    pixels.append(i_transformed[x, y+1])
    pixels.append(i_transformed[x+1, y+1])
    newImage[int(x/2),int(y/2)] = max(pixels)

# Plot the image. Note the size of the axes -- now 256 pixels instead of 512
plt.gray()
plt.grid(False)
plt.imshow(newImage)
#plt.axis('off')
plt.show()

