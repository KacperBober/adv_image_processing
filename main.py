import cv2
from PIL import Image, ImageChops
import numpy as np


def main():
    original: Image.Image = Image.open("images/edited.jpg")
    edited: Image.Image = Image.open("images/org.jpg")

    diff = ImageChops.difference(original, edited)
    gray_scale = diff.convert('LA')

    threshold = 25
    mask = gray_scale.point(lambda p: p > threshold and 255)
    mask.show()



if __name__ == '__main__':
    main()
