import numpy as np
import cv2

img_original = cv2.imread("./images/img_3.jpg")
img_grayscale = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
img_contrast = cv2.convertScaleAbs(img_grayscale, alpha = 1.25, beta = 0)
ret, img_threshold = cv2.threshold(img_contrast, 0, 255, cv2.THRESH_OTSU)

contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

possiblePlates = []

for contour in contours:
    (x, y, width, height) = cv2.boundingRect(contour)
    area = width * height
    roi = img_threshold[y:y+height, x:x+width]
    if (2000 < area < 6000) and (width >= 2 * height):
        possiblePlates.append(roi)
        cv2.rectangle(img_original, (x,y), (x+width,y+height), (0,255,0), 2)

cv2.imshow("Possible plates", img_original)
cv2.waitKey(0)
cv2.destroyAllWindows()