import cv2
import numpy as np
from glob import glob
import os

templates = []
characters = glob("./characters/*.png")
for character in characters:
    templates.append(character)

img_original = cv2.imread("././images/img_1.jpg")
mask = np.zeros(img_original.shape[:2], np.uint8)
img_grayscale = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
img_contrast = cv2.convertScaleAbs(img_grayscale, alpha = 1.25, beta = 0)
(_, img_threshold) = cv2.threshold(img_contrast, 0, 255, cv2.THRESH_OTSU)

(contours, _) = cv2.findContours(img_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

possible_plates = []

for contour in contours:
    (x, y, width, height) = cv2.boundingRect(contour)
    area = width * height
    roi = img_contrast[y:y+height, x:x+width]
    if ((2000 < area < 10000) and (width >= height * 2) and (width <= height * 6)):
        possible_plate = cv2.resize(roi, (225,50))
        characters_found = 0
        for template in templates:
            temp = cv2.imread(template, 0)
            (w, h) = temp.shape[::-1]
            res = cv2.matchTemplate(possible_plate, temp, cv2.TM_CCOEFF_NORMED)
            threshold = 0.8
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                if (mask[pt[1] + h//2, pt[0] + w//2] != 255):
                    mask[pt[1]:pt[1]+h, pt[0]:pt[0]+w] = 255
                    characters_found += 1
        if (characters_found >= 3):
            possible_plates.append(possible_plate)
            cv2.rectangle(img_original, (x,y), (x+width,y+height), (0,255,0), 2)

mask = np.zeros(img_original.shape[:2], np.uint8)
rendszam = []

for possible_plate in possible_plates:
    for template in templates:
            #--------template beolvasása--------
            temp = cv2.imread(template, 0)
            (w, h) = temp.shape[::-1]
            #--------matchTemplate--------
            res = cv2.matchTemplate(possible_plate, temp, cv2.TM_CCOEFF_NORMED)
            threshold = 0.9 if (h > 15) else 0.7
            #--------
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                name = str(os.path.basename(template)).split(".")[0]
                top_left = pt
                bottom_right = (pt[0] + w, pt[1] + h)
                col = top_left[0]
                row = top_left[1]
                if (mask[row + h//2, col + w//2] != 255):
                    mask[row:row+h, col:col+w] = 255
                    cv2.rectangle(possible_plate, top_left, bottom_right, (0,0,255), 1)
                    rendszam.append((name, col))
    cv2.imshow("Rendszam", possible_plate)

def sortBySecond(element): 
    return element[1]

rendszam.sort(key = sortBySecond)  

print("Rendszam:")

for betu in rendszam:
    print(betu[0], sep=' ', end='', flush=True)

cv2.imshow("Eredeti kep", img_original)
cv2.waitKey(0)
cv2.destroyAllWindows()