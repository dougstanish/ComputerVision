import os

import cv2
import numpy as np


def main():

    image = cv2.imread(os.path.realpath('./pics/flower.bmp'))

    print(os.path.realpath('./pics/flower.bmp'))


    lenna = cv2.imread('./pics\\lenna.jpg')

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    red, green, blue = cv2.split(image)

    light = cv2.add(image, 128)
    dark = cv2.subtract(image, 128)

    h = image.shape[0]
    w = image.shape[1]

    image = image + 128

    image[image < 128] = 255

    # loop over the image, pixel by pixel
    # for y in range(0, h):
    #     for x in range(0, w):
    #         # threshold the pixel
    #         image[y, x][0] = 128 + image[y, x][0]
    #         image[y, x][1] = 128 + image[y, x][1]
    #         image[y, x][2] = 128 + image[y, x][2]



    cv2.imshow('Original', image)
    # cv2.imshow('Gray', gray)
    cv2.imshow('Negative', cv2.bitwise_not(image))
    # cv2.imshow('Light', light)
    # cv2.imshow('Dark', dark)
    # cv2.imshow('Red', red)
    # cv2.imshow('Green', gray)
    # cv2.imshow('Blue', blue)
    #
    # gaus = cv2.GaussianBlur(image, (3,3), 5)
    # cv2.imshow('Gausian', gaus)

    # cv2.imshow('Normal', lenna)
    #
    # grayscale = cv2.cvtColor(lenna, cv2.COLOR_BGR2GRAY)
    #
    # cv2.imshow('Gray', grayscale)
    #
    # noise = random




    cv2.waitKey(0)
    cv2.destroyAllWindows()


def image_read_write():

    for file in os.listdir("./pics"):

        image = cv2.imread('./pics/' + file)  # Reads in image

        red = image.copy()  # Copies image
        red[:, :, 0] = 0  # Sets green and blue channels to 0
        red[:, :, 1] = 0

        green = image.copy()  # Copies image
        green[:, :, 0] = 0  # Sets red and blue channels to 0
        green[:, :, 2] = 0

        blue = image.copy()  # Copies image
        blue[:, :, 1] = 0  # Sets red and green channels to 0
        blue[:, :, 2] = 0

        greyscale = image.copy()

        h = greyscale.shape[0]
        w = greyscale.shape[1]

        # loop over the image, pixel by pixel
        for y in range(0, h):
            for x in range(0, w):

                # Get average of the values

                average = np.mean(greyscale[y, x])

                greyscale[y, x][0] = average
                greyscale[y, x][1] = average
                greyscale[y, x][2] = average

        function_greyscale = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

        cv2.imshow('Original', image)
        cv2.imshow('Red', red)
        cv2.imshow('Blue', blue)
        cv2.imshow('Green', green)
        cv2.imshow('Greyscale from layers', greyscale)
        cv2.imshow('Greyscale from function', function_greyscale)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

def pixel_operations():

    image = cv2.imread('./pics/flower.bmp')  # Reads in image

    negative = 255 - image.copy()

    light = image.copy() + 128  # Adds 128 to all pixels
    light[light < 128] = 255  # If the value has overflown, returns it to 255

    dark = image.copy() - 128  # Subtracts 128 to all pixels
    dark[dark > 128] = 0  # If the value has overflown, returns it to 0

    low_contrast = image.copy()
    # TODO Low Contrast


    high_contrast = image.copy()
    high_contrast[high_contrast >= 128] = 255
    high_contrast[high_contrast < 128] * 2





    cv2.imshow('Original', image)
    cv2.imshow('Negative', negative)
    cv2.imshow('Light', light)
    cv2.imshow('Dark', dark)
    cv2.imshow('Low Contrast', low_contrast)
    cv2.imshow('High Contrast', high_contrast)


    cv2.waitKey(0)
    cv2.destroyAllWindows()

def image_noise():

    image = cv2.imread('./pics/lenna.jpg')  # Reads in image

    # what amount of salt
    # what amount of pepper
    # what sigma for gausian

    cv2.imshow('Original', image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # cv2.boxFilter() for average
    # median = cv2.medianBlur(img,5)
    # blur = cv2.GaussianBlur(img,(5,5),0)


if __name__ == '__main__':

    # image_read_write()
    # pixel_operations()
    image_noise()

    # main()
