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

    contours, hierarchy = cv.findContours(canny_contour, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

    cnt = contours[5]

    valid_contours = []
    for contour in contours:
        area = cv.contourArea(contour)

        if area > 10:
            arc_length = cv.arcLength(contour, True)
            valid_contours.append(contour)
            # approx = cv.approxPolyDP(cnt, arc_length*0.02, True)

            # x, y, w, h = cv.boundingRect(approx)
        # cv.rectangle(grinch, (x, y), (x+w, y+h),  (0, 255, 0), 1)

    contours_poly = [None] * len(valid_contours)
    boundRect = [None] * len(valid_contours)
    for i, c in enumerate(valid_contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        boundRect[i] = cv.boundingRect(contours_poly[i])

    for i in range(len(valid_contours)):
        color = (0, 0, 255)
        cv.rectangle(grinch, (int(boundRect[i][0] - 3), int(boundRect[i][1]) - 3),
                     (int(boundRect[i][0] + boundRect[i][2] + 3), int(boundRect[i][1] + boundRect[i][3]) + 3), color, 1)

    cv.drawContours(grinch, valid_contours, -1, (0, 255, 0), 1)
    cv.imshow("Output", grinch)
    cv.waitKey(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
