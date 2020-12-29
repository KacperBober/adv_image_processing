import cv2 as cv
import numpy as np


def main():

    original = cv.imread("images/org.jpg")
    edited = cv.imread("images/edited.jpg")

    grinch = cv.absdiff(original, edited)
    grey_grinch = cv.cvtColor(grinch, cv.COLOR_BGR2GRAY)
    ret, thresh1 = cv.threshold(grey_grinch, 30, 255, cv.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    closing = cv.morphologyEx(thresh1, cv.MORPH_CLOSE, kernel)
    canny_contour = cv.Canny(closing, 150, 200)

    contours, hierarchy = cv.findContours(canny_contour, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    cnt = contours[5]


    valid_contours = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area > 500:
            valid_contours.append(contour)

    cv.drawContours(grinch, valid_contours, -1, (0, 255, 0), 1)
    cv.imshow("Output", grinch)
    cv.waitKey(0)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
