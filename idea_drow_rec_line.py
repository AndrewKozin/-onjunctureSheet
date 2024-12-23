"""
Чертим таблицу на рисунке
https://habr.com/ru/articles/546824/
https://translated.turbopages.org/proxy_u/en-ru.ru.cb1bae16-6761250f-90c910e0-74722d776562/https/stackoverflow.com/questions/32941689/how-to-extract-white-region-in-an-image
"""
import cv2
import numpy as np
# from main import *
import pyperclip, os, subprocess, sys
import pytesseract
from PIL import ImageGrab, Image
from pdf2image import convert_from_path
import json

# 0. Получение изображения из буфера обмена
image = ImageGrab.grabclipboard()

# Проверка, что получено изображение
if image is None:
    print("Ошибка: Не удалось получить изображение из буфера обмена.")

    # 1. Загрузка изображения
    image = cv2.imread('table_image.png')

    # Проверка, что получено изображение
    if image is None:
        print("Ошибка: Не удалось загрузить изображение.")
        sys.exit()

# 2. Предобработка изображения

image = image.convert('RGB')
image.save('./tmp.jpg')
image = cv2.imread('./tmp.jpg')
if os.path.exists('./tmp.jpg'):
    os.remove('./tmp.jpg')

# 2. Предобработка изображения
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, cv2.mean(gray)[0], 255, cv2.THRESH_BINARY_INV)
cv2.imshow('binary', binary)
cv2.waitKey(0)

# Применение адаптивного порога
binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 11)

cv2.imshow('binary', binary)
cv2.waitKey(0)
# 3. Поиск контуров
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

img_contours = image.copy()
img_contours = cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
cv2.imshow('contours', img_contours)
cv2.waitKey(0)

# 4. Фильтрация и рисование границ ячеек
for contour in contours:
    # Получаем границы ячейки
    x, y, w, h = cv2.boundingRect(contour)
    
    # Фильтрация по размеру (например, минимальная ширина и высота)
    if w > 20 and h > 20:  # Настройте пороги по необходимости
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


# Поиск прямых линий на изображении
lines = cv2.HoughLinesP(binary, 1, np.pi/360, 255)

# Нанесение прямых линий на изображение
for line in lines:
    x1, y1, x2, y2 = line[0]
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    if length > 50:  # Отбор по длине линий
        cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
# 5. Отображение результата
cv2.imshow('Detected Table Borders', image)
cv2.imwrite('output_image.png', image)
cv2.waitKey(0)
cv2.destroyAllWindows()