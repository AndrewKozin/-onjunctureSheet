"""
Чертим таблицу на рисунке
https://habr.com/ru/articles/546824/
https://translated.turbopages.org/proxy_u/en-ru.ru.cb1bae16-6761250f-90c910e0-74722d776562/https/stackoverflow.com/questions/32941689/how-to-extract-white-region-in-an-image
https://habr.com/ru/articles/577776/
ВАЖНО
    PyMuPDF                   1.21.1
"""
import cv2
import numpy as np
# from main import *
import pyperclip, os, subprocess, sys
import pytesseract
from PIL import ImageGrab, Image
from pdf2image import convert_from_path
import json
# from frontend import *
import tools
import fitz



# 0. Получение изображения из буфера обмена
image = ImageGrab.grabclipboard()

# Проверка, что получено изображение
if image is None:
    print("Ошибка: Не удалось получить изображение из буфера обмена.")

    # Открытие PDF-файла
    doc = fitz.open('simple.pdf')

    # Получение первой страницы
    page = doc[0]

    # Преобразование страницы в изображение
    pix = page.get_pixmap(dpi=100)

    # Сохранение изображения в файл
    pix.save('table_image.png')
    # 1. Загрузка изображения
    image = cv2.imread('table_image.png')

    # Проверка, что получено изображение
    if image is None:
        print("Ошибка: Не удалось загрузить изображение.")
        sys.exit()
    print('Размерность из файла', image.shape)
else:
    # 2. Предобработка изображения, полученного из буфера обмена

    image = image.convert('RGB')
    image.save('./tmp.jpg')
    image = cv2.imread('./tmp.jpg')
    if os.path.exists('./tmp.jpg'):
        os.remove('./tmp.jpg')
    print('Размерность из буфера обмена', image.shape)


# Создание объекта CLAHE
clahe = cv2.createCLAHE(clipLimit=50, tileGridSize=(50, 50))

# Преобразование изображения в LAB
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

# Разделение LAB на каналы
l, a, b = cv2.split(lab)

# Применение CLAHE к каналу L
l2 = clahe.apply(l)

# Объединение каналов обратно в LAB
lab = cv2.merge((l2, a, b))

# Преобразование из LAB обратно в BGR
image2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
# cv2.imshow('image2', image2)
# cv2.waitKey(0)

# Преобразование изображения в оттенки серого
gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray', gray)
# cv2.waitKey(0)

# Применение бинарного порога
ret, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY_INV)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 11)
# thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 51)

print(cv2.mean(gray)[0]) 

# Создание ядра для операции эрозии
kernel = np.ones((1, 1), np.uint8)

# Применение операции эрозии
obr_img = cv2.erode(thresh, kernel, iterations=1)
cv2.imshow('obr_img', obr_img)
cv2.waitKey(0)

# Применение гауссовой размытия
obr_img = cv2.GaussianBlur(obr_img, (3,3), 0)
# cv2.imshow('obr_img', obr_img)
# cv2.waitKey(0)

# 3. Поиск контуров
contours, _ = cv2.findContours(obr_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 4. Фильтрация и рисование границ ячеек

for contour in contours:
    # print('Количество контуров: ', len(contours))
    # Получаем границы ячейки
    x, y, w, h = cv2.boundingRect(contour)
    
    # Фильтрация по размеру (например, минимальная ширина и высота)
    if w > 20 and h > 20:  # Настройте пороги по необходимости
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # img_simple = image[y:y+h, x:x+w]
        # cv2.imshow('img_simple', img_simple)
        # cv2.waitKey(0)
        
# 5. Отображение результата
cv2.imshow('Detected Table Borders', image)
cv2.imwrite('output_image.png', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
