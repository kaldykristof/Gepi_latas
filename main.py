import cv2
import numpy as np
from glob import glob
from os import path
import timeit

# Template karakterek betöltése
characters = []
glob_characters = glob("characters/*.png")
for char in glob_characters:
    characters.append(char)

# Tesztképek betöltése
images = []
glob_images = glob("images/*.jpg")
for img in glob_images:
    images.append(img)

for image in images:
    start_time = timeit.default_timer()

    # Bemeneti kép feldolgozása
    img_original = cv2.imread(image, 1)
    mask = np.zeros(img_original.shape[:2], np.uint8)
    img_grayscale = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
    img_contrast = cv2.convertScaleAbs(img_grayscale, alpha = 1.25, beta = 0)
    (_, img_threshold) = cv2.threshold(img_contrast, 0, 255, cv2.THRESH_OTSU)
    (contours, _) = cv2.findContours(img_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Üres lista, amibe később a talált karakterek kerülnek
    found_characters = []

    for contour in contours:
        (x, y, width, height) = cv2.boundingRect(contour)

        area = width * height
        roi = img_contrast[y:y+height, x:x+width]

        if ((2000 < area < 50000) and (width >= height * 2) and (width <= height * 6)):
            possible_plate = cv2.resize(roi, (230,50))

            characters_found = 0

            for character in characters:
                char = cv2.imread(character, 0)
                (w, h) = char.shape[::-1]
                res = cv2.matchTemplate(possible_plate, char, cv2.TM_CCOEFF_NORMED)

                # Küszöbérték beállítása a keresett karakter méretei alapján
                if ((w <= 20) and (h >= 20)): # '1'-es és 'I' karakterek
                    threshold = 0.9
                elif (h <= 20): # '-' karakter
                    threshold = 0.92
                else: # Minden más karakter
                    threshold = 0.8

                loc = np.where(res > threshold)
                for pt in zip(*loc[::-1]):
                    name = str(path.basename(character)).split(".")[0]
                    top_left = pt
                    bottom_right = (pt[0] + w, pt[1] + h)
                    col = top_left[0]
                    row = top_left[1]
                    if (mask[row + h//2, col + w//2] != 255):
                        mask[row:row+h, col:col+w] = 255
                        cv2.rectangle(possible_plate, top_left, bottom_right, (0,0,255), 1)
                        characters_found += 1
                        found_characters.append((name, col))

            if (characters_found > 1):
                img_license_plate = possible_plate
                cv2.rectangle(img_original, (x,y), (x+width,y+height), (0,255,0), 2)

    expected_plate = str(path.basename(image)).split(".")[0]
    print("\nVárt rendszám:\n{0}".format(expected_plate))

    # Karakterek sorba rendezése oszlop alapján
    def sortBySecond(element): 
        return element[1]
    found_characters.sort(key = sortBySecond)  

    # Formázott rendszám kiíratása
    print("\nTalált rendszám:")
    license_plate = ""
    for character in found_characters:
        license_plate += character[0]
    print(license_plate)

    print("\nRendszám sikeresen felismerve!" if expected_plate == license_plate else "\nA rendszám felismerése sikertelen!")

    if (len(found_characters) > 0):
        cv2.imshow("Eredeti kep", img_original)
        cv2.imshow("Talalt rendszam", img_license_plate)
    else:
        print("\nNem található rendszám!")

    # Számítási idő kiszámítása
    stop_time = timeit.default_timer()
    runtime = round(stop_time - start_time, 5)
    print("\nSzámítási idő: {0} másodperc".format(runtime))

    cv2.waitKey(0)
    cv2.destroyAllWindows()