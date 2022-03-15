import numpy as np
from PIL import Image as imageMain
import concurrent.futures
import os
import cv2
import numpy


def recognize_and_paint(file):
    try:
        imagePil = imageMain.open(directory + file)
        imageCv = cv2.cvtColor(numpy.array(imagePil), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(imageCv, cv2.COLOR_BGR2GRAY)
        bilateral = cv2.bilateralFilter(gray, 11, 17, 17)
        blur = cv2.GaussianBlur(bilateral, (3, 3), 0)
        edged = cv2.Canny(blur, 190, 255, L2gradient=True)

        contours, hierarchy = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        rectangleContours = []

        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            approximationAccuracy = 0.02 * perimeter
            approximation = cv2.approxPolyDP(contour, approximationAccuracy, True)
            x, y, w, h = cv2.boundingRect(contour)

            if (len(approximation) == 4) and (perimeter < 800):
                rectangleContours.append(contour)

        if len(rectangleContours) > 0:
            plateContour = rectangleContours[0]
        else:
            return imageCv

        x, y, w, h = cv2.boundingRect(plateContour)
        plateImage = imageCv[y:y + h, x:x + w]
        plateBackgroundColor = findMostOccurringColor(plateImage)

        if not checkBrighntess(plateBackgroundColor, 100):
            cv2.imwrite("./results/" + os.fsdecode(file), result, [int(cv2.IMWRITE_WEBP_QUALITY), 75])

        result = cv2.drawContours(imageCv.copy(), [plateContour], -1, plateBackgroundColor, -1)
        cv2.imwrite("./results/" + os.fsdecode(file), result, [int(cv2.IMWRITE_WEBP_QUALITY), 75])
    except:
        cv2.imwrite("./results/" + os.fsdecode(file), cv2.cvtColor(numpy.array(imagePil), cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_WEBP_QUALITY), 75])


def findMostOccurringColor(cvImage):
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
    return np.mean(BGR) >= value


directory = os.fsencode("./samples/")

with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
    futures = []
    for file in os.listdir(directory):
        futures.append(executor.submit(recognize_and_paint, file))


