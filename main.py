import numpy as np
from PIL import Image as imageMain
import os
import cv2
import numpy


def recognize_and_paint(imageCv):
    gray = cv2.cvtColor(imageCv, cv2.COLOR_BGR2GRAY)

    bilateral = cv2.bilateralFilter(gray, 31, 31, 31)
    blur = cv2.GaussianBlur(bilateral, (5, 5), 0)
    edged = cv2.Canny(blur, 0, 255, L2gradient=True)

    contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
    rectangleContours = []

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)

        if perimeter >= 300:
            approximationAccuracy = 0.000001 * perimeter
        else:
            approximationAccuracy = 0.02 * perimeter

        approximation = cv2.approxPolyDP(contour, approximationAccuracy, True)

        if len(approximation) == 4:
            rectangleContours.append(contour)

    if len(rectangleContours) > 0:
        plateContour = rectangleContours[0]
    else:
        return imageCv

    x, y, w, h = cv2.boundingRect(plateContour)
    plateImage = imageCv[y:y + h, x:x + w]

    plateBackgroundColor = findMostOccurringColor(plateImage)

    if checkBrighntess(plateBackgroundColor, 90):
        print("small brightness: %.1f" % (np.mean(plateBackgroundColor)))
        return imageCv

    tempContours3 = cv2.drawContours(imageCv.copy(), [plateContour], -1, (255, 0, 0), -1)

    return tempContours3


def findMostOccurringColor(cvImage) -> (int, int, int):
    width, height, channels = cvImage.shape
    colorCount = {}
    for y in range(0, height):
        for x in range(0, width):
            BGR = (int(cvImage[x, y, 0]), int(cvImage[x, y, 1]), int(cvImage[x, y, 2]))
            if BGR in colorCount:
                colorCount[BGR] += 1
            else:
                colorCount[BGR] = 1

    maxCount = 0
    maxBGR = (0, 0, 0)
    for BGR in colorCount:
        count = colorCount[BGR]
        if count > maxCount:
            maxCount = count
            maxBGR = BGR

    return maxBGR


def checkBrighntess (BGR, value):
    return BGR[0] <= value or BGR[1] <= value and BGR[2] <= value


directory = os.fsencode("./samples/")

for file in os.listdir(directory):
    filename = os.fsdecode(file)
    imagePil = imageMain.open(directory + file)
    imageCv = cv2.cvtColor(numpy.array(imagePil), cv2.COLOR_RGB2BGR)
    result = recognize_and_paint(imageCv)
    cv2.imwrite("./results/" + os.fsdecode(file), result)