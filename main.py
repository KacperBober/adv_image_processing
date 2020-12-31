import cv2 as cv
import numpy as np


def filter_contours(contours):

    valid_contours = []
    for contour in contours:
        arc_length = cv.arcLength(contour, True)
        print(arc_length)
        if arc_length > 50:
            valid_contours.append(contour)
    return valid_contours


def main():
    original = cv.imread("images/org.jpg")
    edited = cv.imread("images/edited.jpg")

    grinch = cv.absdiff(original, edited)
    grey_grinch = cv.cvtColor(grinch, cv.COLOR_BGR2GRAY)
    ret, thresh1 = cv.threshold(grey_grinch, 27, 255, cv.THRESH_BINARY)

    #all credid to https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
    kernel = np.ones((5, 5), np.uint8)
    closing = cv.morphologyEx(thresh1, cv.MORPH_CLOSE, kernel)
    canny_contour = cv.Canny(closing, 150, 200)


    contours, hierarchy = cv.findContours(canny_contour, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    valid_contours = filter_contours(contours)

    cv.drawContours(edited, valid_contours, -1, (255, 255, 255), thickness=1)

    colored_grinch = cv.bitwise_and(edited, edited, mask=closing)

    b, g, r = cv.split(colored_grinch)
    rgba = [b, g, r, closing]
    no_background = cv.merge(rgba, 4)

    #https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.html
    contours_poly = [None] * len(valid_contours)
    bounding_rect = [None] * len(valid_contours)
    for i, c in enumerate(valid_contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        bounding_rect[i] = cv.boundingRect(contours_poly[i])

    grinches = []
    for i in range(len(valid_contours)):
        color = (0, 0, 255)
        cv.rectangle(edited, (int(bounding_rect[i][0] - 3), int(bounding_rect[i][1]) - 3),
                     (int(bounding_rect[i][0] + bounding_rect[i][2] + 3), int(bounding_rect[i][1] + bounding_rect[i][3]) + 3), color, 2)
       # roi = no_background[bounding_rect[i][0]: bounding_rect[i][0] + bounding_rect[i][2], bounding_rect[i][1]: bounding_rect[i][1] + bounding_rect[i][3]]
        #cv.imwrite(str(i) + '.jpg', roi)

    cv.imwrite("test.png", colored_grinch)
    resized = cv.resize(colored_grinch, (1440, 960))
    cv.imshow("Output", edited)
    cv.waitKey(0)
    cv.imshow("Output", no_background)
    cv.waitKey(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
