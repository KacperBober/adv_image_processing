# Do poprawnego działania programu zdjęcia powinny znajdować się w tym samym folderze co plik .py
# W tym projekcie udało mi się zrealizować

import cv2 as cv
import numpy as np
import glob


def filter_contours(contours):
    valid_contours = []
    for contour in contours:
        arc_length = cv.arcLength(contour, True)
        if arc_length > 50:
            valid_contours.append(contour)
    return valid_contours


def show_image(image, window_name, x_pos, y_pos):
    cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
    cv.moveWindow(window_name, x_pos, y_pos)  # Move it to (40,30)
    cv.imshow(window_name, image)


def main():

    # https://stackoverflow.com/questions/4568580/python-glob-multiple-filetypes
    types = ('*.jpeg', '*.jpg')  # the tuple of file types
    files_grabbed = []
    for files in types:
        files_grabbed.extend(glob.glob(files))

    original = None
    edited = None
    images = []
    for file in files_grabbed:
        im = cv.imread(file)
        if file == 'org.jpg':
            original = im
        elif file == 'edited.jpg':
            edited = im
        else:
            images.append(im)

    # compare two images, they are different where grinches are
    # Then convert RGB to Gray and apply threshold to obtain white pixels mask
    grinch = cv.absdiff(original, edited)
    grey_grinch = cv.cvtColor(grinch, cv.COLOR_BGR2GRAY)
    ret, thresh1 = cv.threshold(grey_grinch, 27, 255, cv.THRESH_BINARY)

    # all credit to https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
    # remove single (white/black) pixels
    kernel = np.ones((5, 5), np.uint8)
    closed_shapes = cv.morphologyEx(thresh1, cv.MORPH_CLOSE, kernel)

    # https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    canny_contour = cv.Canny(closed_shapes, 150, 200)
    contours, hierarchy = cv.findContours(canny_contour, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    valid_contours = filter_contours(contours)

    # https://docs.opencv.org/master/d0/d86/tutorial_py_image_arithmetics.html
    # apply white pixel mask to image with grinches to get them colored
    colored_grinch = cv.bitwise_and(edited, edited, mask=closed_shapes)

    # https://stackoverflow.com/questions/40527769/removing-black-background-and-make-transparent-from-grabcut-output-in-python-ope
    # remove black background from image
    b, g, r = cv.split(colored_grinch)
    rgba = [b, g, r, closed_shapes]
    no_background = cv.merge(rgba, 4)

    # https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.html
    # get bounding-box from contours approximation and crop those boxes from image to get grinches
    # also draw those boxes on edited image and write grinches as images for further usage
    contours_poly = [None] * len(valid_contours)
    bounding_rect = [None] * len(valid_contours)
    for i, c in enumerate(valid_contours):
        contours_poly[i] = cv.approxPolyDP(c, 3, True)
        bounding_rect[i] = cv.boundingRect(contours_poly[i])

    for i in range(len(valid_contours)):
        color = (0, 0, 255)
        cv.rectangle(edited, (int(bounding_rect[i][0]), int(bounding_rect[i][1])),
                     (int(bounding_rect[i][0] + bounding_rect[i][2]), int(bounding_rect[i][1] + bounding_rect[i][3])), color, 2)
        roi = no_background[bounding_rect[i][1]: bounding_rect[i][1] + bounding_rect[i][3], bounding_rect[i][0]: bounding_rect[i][0] + bounding_rect[i][2]]
        show_image(roi, f"grinch {i}", 100, 100)
        # Uncomment to write to png images and see that there is no background
        # cv.imwrite(str(i) + '.png', roi)

    resized = cv.resize(edited, (1440, 960))
    show_image(resized, "BOUNDING BOXES", 200, 100)
    cv.waitKey(0)


if __name__ == '__main__':
    main()
