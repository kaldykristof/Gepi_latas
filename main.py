import cv2

img_original = cv2.imread("./images/img_1.jpg")
img_grayscale = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
img_contrast = cv2.convertScaleAbs(img_grayscale, alpha = 1.25, beta = 0)
ret, img_threshold = cv2.threshold(img_contrast, 0, 255, cv2.THRESH_OTSU)

cv2.imshow("Treshold", img_threshold)
cv2.waitKey(0)
cv2.destroyAllWindows()