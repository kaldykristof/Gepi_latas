import cv2
import numpy as np

img_original = cv2.imread("./images/img_1.jpg")
img_grayscale = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
img_contrast = cv2.convertScaleAbs(img_grayscale, alpha = 1.25, beta = 0)
ret, img_threshold = cv2.threshold(img_contrast, 0, 255, cv2.THRESH_OTSU)
#img_canny = cv2.Canny(img_contrast, 500,60,apertureSize=3)

contours, hierarchy = cv2.findContours(img_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

def enhance_image(image, scale):
    scale = scale * 100
    new_width = int(image.shape[1] * scale / 100)
    new_height = int(image.shape[0] * scale / 100)
    new_dimensions = (new_width, new_height)
    resizedImage = cv2.resize(image, new_dimensions) #interpolation = cv2.INTER_LINEAR
    denoised = cv2.fastNlMeansDenoising(resizedImage)
    return denoised

possiblePlates = []

for contour in contours:
    (x, y, width, height) = cv2.boundingRect(contour)
    area = width * height
    roi = img_contrast[y:y+height, x:x+width].copy()
    if (2000 < area < 6000) and (width >= 2 * height):
        possiblePlates.append(enhance_image(roi, 2.5))
        cv2.rectangle(img_original, (x,y), (x+width,y+height), (0,255,0), 2)

for i, plate in enumerate(possiblePlates):
    cv2.imshow("Talalt rendszamtablak ({})".format(i), plate)

cv2.imshow("Eredeti kep", img_original)
cv2.waitKey(0)
cv2.destroyAllWindows()