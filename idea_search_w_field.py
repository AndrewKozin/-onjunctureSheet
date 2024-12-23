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
image_src = cv2.imread('./tmp.jpg')
if os.path.exists('./tmp.jpg'):
    os.remove('./tmp.jpg')

gray = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
# gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 11)
_, gray = cv2.threshold(gray, 250,255,0)
cv2.imshow('gray', gray)
cv2.waitKey(0)


contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
largest_area = sorted(contours, key=cv2.contourArea)[-1]
mask = np.zeros(image_src.shape, np.uint8)
cv2.drawContours(mask, [largest_area], 0, (255,255,255,255), -1)
cv2.imshow('mask', mask)
cv2.waitKey(0)

dst = cv2.bitwise_and(image_src, mask)
mask = 255 - mask
roi = cv2.add(dst, mask)
cv2.imshow('roi', roi)
cv2.waitKey(0)

roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
ret, gray = cv2.threshold(roi_gray, 250,255,0)
contours, _ = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

max_x = 0
max_y = 0
min_x = image_src.shape[1]
min_y = image_src.shape[0]

for c in contours:
    if 150 < cv2.contourArea(c) < 100000:
        x, y, w, h = cv2.boundingRect(c)
        min_x = min(x, min_x)
        min_y = min(y, min_y)
        max_x = max(x+w, max_x)
        max_y = max(y+h, max_y)

roi = roi[min_y:max_y, min_x:max_x]
cv2.imwrite("roi.png", roi)