import cv2
import numpy as np

img = cv2.imread('cat.jpg')
cv2.imshow('cat', img)
print(img.shape)


# ROI of face only by defining pixel location
# image[height, width] start from the top left
face_ROI = img[5:123, 80:200]
# show the ROI of face
cv2.imshow('code_face', face_ROI)


# ROI of the eyes only by using selectROI to draw rectangle
(x, y, w, h) = cv2.selectROI('select_eyes', img)
eyes_ROI = img[y:y+h, x:x+w]
cv2.imshow('select_face', eyes_ROI)
cv2.imwrite('eyes.jpg', eyes_ROI)
print("x={0}, y={1}, w={2}, h={3}".format(x,y,w,h))


cv2.waitKey(0)
cv2.destroyAllWindows()