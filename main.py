import cv2
import numpy as np
from glob import glob
from os import path
import timeit

# True  => manuálisan kell lépkedni a képek között, a végén % összegzés
# False => automatikus, nem mutatja a képeket csak a végén a %-ot
STEP_BY_STEP = False

full_match = 0  # Hány képen ismerte fel az összes karaktert
half_match = 0  # Hány képen ismerte fel legalább a karakterek felét
no_match = 0    # Hány képen nem ismert fel egy karaktert sem

characters_not_found = [] # Karakterek, amiket nem ismert fel az egyes képeken

# Template karakterek betöltése
characters = glob("characters/*.png")

# Tesztképek betöltése
images = glob("images/*.jpg")

def secondElement(element): 
        return element[1]

for i, image in enumerate(images):
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

    # Várt rendszám
    expected_plate = str(path.basename(image)).split(".")[0]

    # Karakterek sorba rendezése oszlop alapján
    found_characters.sort(key = secondElement)

    # Karakterek stringé konvertálása
    license_plate = ""
    for character in found_characters:
        license_plate += character[0]

    if (len(license_plate) == 7):
        full_match += 1
    if (len(license_plate) > 3):
        half_match += 1
    if (len(license_plate) == 0):
        no_match += 1

    # Számítási idő kiszámítása
    stop_time = timeit.default_timer()
    runtime = round(stop_time - start_time, 2)

    # Fel nem ismert karakter helyére '_' karaktert rak
    formatted_plate = ""
    for i, found_char in enumerate(license_plate):
        if (found_char == expected_plate[i]):
            formatted_plate += found_char
        else:
            formatted_plate += '_' + found_char
    while (len(formatted_plate) != 7):
        formatted_plate += '_'

    if (STEP_BY_STEP):
        print("Várt rendszám:   {0}".format(expected_plate))
        print("Talált rendszám: {0}".format(formatted_plate, runtime))
        print("Rendszám sikeresen felismerve!" if (expected_plate == license_plate) else "A rendszám felismerése sikertelen!")
        print("Számítási idő: {0} másodperc".format(runtime))
        print("-------------------------------------------------")

        if (len(found_characters) > 0):
            cv2.imshow("Eredeti kep", img_original)
            cv2.imshow("Talalt rendszam", img_license_plate)
        else:
            print("\nNem található rendszám!")

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Fel nem ismert karakterek szűrése
    for i, exp_char in enumerate(expected_plate):
        if (exp_char != formatted_plate[i] and exp_char not in characters_not_found):
            characters_not_found.append(exp_char)

characters_not_found = sorted(characters_not_found)

print("-------------------------------------------------")
print("Összes kép száma: {0} db\nEbből:".format(len(images)))
print(" - 100 %-os találat: {0} db ({1}%)".format(full_match, round(full_match / len(images) * 100, 2)))
print(" - >50 %-os találat: {0} db ({1}%)".format(half_match, round(half_match / len(images) * 100, 2)))
print(" - <50 %-os találat: {0} db ({1}%)".format(len(images)-half_match, round((len(images)-half_match) / len(images) * 100, 2)))
print(" -   0 %-os találat: {0} db ({1}%)".format(no_match, round(no_match / len(images) * 100, 2)))
print("Fel nem ismert karakterek:")
print(*characters_not_found, sep = ", ")  
print("-------------------------------------------------")