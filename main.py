import cv2 as cv
import numpy as np


def main():

    original = cv.imread("images/org.jpg")
    edited = cv.imread("images/edited.jpg")

    grinch = cv.absdiff(original, edited)
    grey_grinch = cv.cvtColor(grinch, cv.COLOR_BGR2GRAY)

    cv.imshow("Output", grey_grinch)
    cv.waitKey(0)




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
