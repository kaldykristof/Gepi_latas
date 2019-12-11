import cv2
import numpy as np
from glob import glob
from os import path
import timeit

start_time = timeit.default_timer()

# Karakterek betöltése egy listába
templates = []
template_characters = glob("characters/*.png")
for temp_char in template_characters:
    templates.append(temp_char)

# Bemeneti kép feldolgozása
img_original = cv2.imread("images/img_10.jpg", 1)
mask = np.zeros(img_original.shape[:2], np.uint8)
img_grayscale = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
img_contrast = cv2.convertScaleAbs(img_grayscale, alpha = 1.25, beta = 0)
(_, img_threshold) = cv2.threshold(img_contrast, 0, 255, cv2.THRESH_OTSU)
(contours, _) = cv2.findContours(img_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Üres lista, amibe később a talált karakterek kerülnek
found_characters = []

for i,contour in enumerate(contours):
    (x, y, width, height) = cv2.boundingRect(contour)
    area = width * height
    roi = img_contrast[y:y+height, x:x+width]
    # Kontúrok szűkítése rendszámtáblára hasonlító téglalapokra
    if ((2000 < area < 10000) and (width >= height * 2) and (width <= height * 6)):
        # Kontúr átméretezése fix méretre
        possible_plate = cv2.resize(roi, (225,50))
        characters_found = 0
        for template in templates:
            temp = cv2.imread(template, 0)
            (w, h) = temp.shape[::-1]
            res = cv2.matchTemplate(possible_plate, temp, cv2.TM_CCOEFF_NORMED)
            # Küszöbérték beállítása a keresett karakter méretei alapján
            if ((w <= 20) and (h >= 20)): # '1'-es és 'I' karakterek
                threshold = 0.9
            elif (h <= 20): # '-' karakter
                threshold = 0.8
            else: # Minden más karakter
                threshold = 0.8
            # A küszöbértéket megugró találatok elmentése
            loc = np.where(res > threshold)
            for pt in zip(*loc[::-1]):
                name = str(path.basename(template)).split(".")[0]
                top_left = pt
                bottom_right = (pt[0] + w, pt[1] + h)
                col = top_left[0]
                row = top_left[1]
                if (mask[row + h//2, col + w//2] != 255):
                    # Ha valahol talált egyezést megjelöli, hogy ott többször ne találhasson
                    mask[row:row+h, col:col+w] = 255
                    cv2.rectangle(possible_plate, top_left, bottom_right, (0,0,255), 1)
                    characters_found += 1
                    found_characters.append((name, col))
        # Ha pontosan 7 karaktert talált a kontúron mentse el és rajzolja ki az eredeti képre
        if (characters_found == 7):
            license_plate = possible_plate
            cv2.rectangle(img_original, (x,y), (x+width,y+height), (0,255,0), 2)

# Talált karakterek kiíratása (rendezetlen)
print("\nTalált karakterek (rendezetlen):")
for character in found_characters:
    print(character, end = " ")

# Karakterek sorba rendezése oszlop alapján
def sortBySecond(element): 
    return element[1]
found_characters.sort(key = sortBySecond)  

# Talált karakterek kiíratása (rendezett)
print("\n\nTalált karakterek (rendezett):")
for character in found_characters:
    print(character, end = " ")

# Formázott rendszám kiíratása
print("\n\nTalált rendszám:")
for character in found_characters:
    print(character[0], end = '')

if (len(found_characters) == 7):
    cv2.imshow("Eredeti kep", img_original)
    cv2.imshow("Talalt rendszam", license_plate)
else:
    print("\n\nA talált rendszám nem teljes!")

# Futásidő kiszámítása
stop_time = timeit.default_timer()
runtime = round(stop_time - start_time, 5)
print("\n\nFutásidő: {0} másodperc".format(runtime))

cv2.waitKey(0)
cv2.destroyAllWindows()