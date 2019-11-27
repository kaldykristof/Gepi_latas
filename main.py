import cv2
import numpy as np
import glob
from os import path

template_characters = []
characters = glob.glob("./characters/*.png")
for character in characters:
    character_image = cv2.imread(character, 0)
    template_characters.append(character_image)

img_original = cv2.imread("./images/img_6.jpg")
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
        for character in template_characters:
            w, h = character.shape[::-1]
            res = cv2.matchTemplate(possible_plate, character, cv2.TM_CCOEFF_NORMED)
            threshold = 0.7
            loc = np.where(res >= threshold)
            for pt in zip(*loc[::-1]):
                if (mask[pt[1] + h//2, pt[0] + w//2] != 255):
                    mask[pt[1]:pt[1]+h, pt[0]:pt[0]+w] = 255
                    cv2.rectangle(possible_plate, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)
                    characters_found += 1
        if (characters_found >= 3):
            possible_plates.append(possible_plate)
            cv2.rectangle(img_original, (x,y), (x+width,y+height), (0,255,0), 2)

for possible_plate in possible_plates:
    cv2.imshow("Rendszamtabla", possible_plate)
    
cv2.imshow("Eredeti kep", img_original)
cv2.waitKey(0)
cv2.destroyAllWindows()