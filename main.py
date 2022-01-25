from PIL import Image as imageMain
import os
import cv2
import numpy


def recognize_and_paint(imageCv):
    gray = cv2.cvtColor(imageCv, cv2.COLOR_BGR2GRAY)
    bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
    blur = cv2.GaussianBlur(bilateral, (5, 5), 0)
    edged = cv2.Canny(blur, 170, 200)

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    rectangleContours = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approximationAccuracy = 0.02 * perimeter
        approximation = cv2.approxPolyDP(contour, approximationAccuracy, True)

        if len(approximation) == 4:
            rectangleContours.append(contour)

    plateContour = rectangleContours[0]
    plateBackgroundColor = (255, 255, 255)

    tempContours3 = cv2.drawContours(imageCv.copy(), [plateContour], -1, plateBackgroundColor, -1)

    return tempContours3


directory = os.fsencode("./samples/")

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    imagePil = imageMain.open(directory + file)
    imageCv = cv2.cvtColor(numpy.array(imagePil), cv2.COLOR_RGB2BGR)
    result = recognize_and_paint(imageCv)
    cv2.imwrite("./results/" + os.fsdecode(file), result)
