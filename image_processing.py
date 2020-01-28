import math
import os
import random

import cv2
import numpy as np
from skimage.util import random_noise


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
    """
    Reads all images, and performs the read/write operations specified in part 1 of the homework
    """

    for file in os.listdir("./pics"):  # For each image

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

        greyscale = image.copy()  # Copies image

        h = greyscale.shape[0]  # Gets size of the image
        w = greyscale.shape[1]

        # loop over the image, pixel by pixel
        for y in range(0, h):
            for x in range(0, w):

                # Get average of the values
                average = np.mean(greyscale[y, x])

                greyscale[y, x][0] = average  # Sets R G and B channels to the average
                greyscale[y, x][1] = average
                greyscale[y, x][2] = average

        # Copies image, and uses built in function to convert to grayscale
        function_greyscale = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

        # Displays all images
        cv2.imshow('Original', image)
        cv2.imshow('Red', red)
        cv2.imshow('Blue', blue)
        cv2.imshow('Green', green)
        cv2.imshow('Greyscale from layers', greyscale)
        cv2.imshow('Greyscale from function', function_greyscale)

        # Waits for key press, then closes all windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def pixel_operations():
    """
    Runs pixel operations on flower.bmp, part 2 of the homework
    """

    image = cv2.imread('./pics/flower.bmp')  # Reads in image

    negative = 255 - image.copy()  # Subtracts the value of each pixel from 255, resulting in negative

    light = image.copy() + 128  # Adds 128 to all pixels
    light[light < 128] = 255  # If the value has overflown, returns it to 255

    dark = image.copy() - 128  # Subtracts 128 to all pixels
    dark[dark > 128] = 0  # If the value has overflown, returns it to 0

    low_contrast = image.copy()  # Copies the image for manipulation
    low_contrast = np.floor_divide(low_contrast, 2)  # Divides by 2 to lower contrast, uses floor to avoid decimals

    high_contrast = image.copy()  # Copies the image for manipulation
    high_contrast[high_contrast >= 128] = 255  # If the number would overflow, sets it to max
    high_contrast[high_contrast < 128] * 2  # Otherwise, multiplies pixel by 2

    # Displays all images
    cv2.imshow('Original', image)
    cv2.imshow('Negative', negative)
    cv2.imshow('Light', light)
    cv2.imshow('Dark', dark)
    cv2.imshow('Low Contrast', low_contrast)
    cv2.imshow('High Contrast', high_contrast)

    cv2.waitKey(0)  # Waits for key press, then closes all windows
    cv2.destroyAllWindows()


def image_noise():
    """
    Adds noise to images, both salt and pepepr as well as gaussian
    @return: The unaltered image and the image with added salt and pepper
    """

    image = cv2.imread('./pics/lenna.jpg')  # Reads in image

    salt_and_pepper = np.zeros(image.shape, np.uint8)  # Creates empty array to hold salt and pepper noise

    probability = 0.01  # Probability of noise, used for pepper

    threshold = 1 - probability  # Value used to determine salt

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):

            random_number = random.random()  # Get random number between 0 and 1
            if random_number < probability:  # If the random number is below the probability
                salt_and_pepper[x][y] = 0  # Adds pepper
            elif random_number > threshold:  # If the random number is above the threshold
                salt_and_pepper[x][y] = 255  # Adds salt
            else:
                salt_and_pepper[x][y] = image[x][y]  # Otherwise just adds correct pixel

    gaussian = image.copy()  # Copies image for gaussian noise

    gaus_mean = 0  # Sets mean and Std for random noise
    gaus_sigma = 0.01**0.5
    gaussian_noise = np.random.normal(gaus_mean, gaus_sigma, image.shape)  # Calculates the random noise

    # Normalizes the image from -1-1, to match the noise
    gaussian = cv2.normalize(gaussian, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    gaussian = gaussian + gaussian_noise  # Adds the noise

    # Displays all images
    cv2.imshow('Original', image)
    cv2.imshow('Salt and Pepper', salt_and_pepper)
    cv2.imshow('Gaussian', gaussian)

    cv2.waitKey(0)  # Waits for key press, then closes all windows
    cv2.destroyAllWindows()

    return image, salt_and_pepper


def psnr(image, noisy_image):
    """
    Computes the peak signal-to-noise ratio of an image and its filtered counterpart
    @param image: the original image
    @param noisy_image: the filtered image
    @return: The PSNR
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Converts images to grayscale
    noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY)

    mse = np.mean((image - noisy_image) ** 2)

    psnr_value = 10 * math.log10((255 ** 2) / mse)

    return psnr_value



def image_denoising():
    """
    Function that takes a salt-pepper noised image, and attempts to denoise it
    """

    image, salt_and_pepper = image_noise()

    moving_average = cv2.blur(salt_and_pepper, (3, 3))
    median = cv2.medianBlur(salt_and_pepper, 3)

    print('PSNR for moving average is ' + str(psnr(image, moving_average)))
    print('PSNR for median is ' + str(psnr(image, median)))


    cv2.imshow('Salt and Pepper', salt_and_pepper)
    cv2.imshow('Moving Average', moving_average)
    cv2.imshow('Median', median)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def image_filters():

    temp=matlab_style_gauss2D(sigma=2)



    # temp = cv2.getGaussianKernel(3, 2)

    print(temp)

    # test = np.outer(temp, temp)

    image_filtering(temp)

    print(np.sum(temp))

    print(temp)

    pass


def image_filtering(filter):

    image = cv2.imread('./pics/gulfstream.jpg')  # Reads in image

    filtered_image = image.copy()

    h = image.shape[0]
    w = image.shape[1]

    filter_size = filter.shape[0]

    for y in range(0, h):
        for x in range(0, w):
            # threshold the pixel

            average_weighted_pixel = np.ndarray(shape=3)

            for filter_y in range(0, filter_size):
                for filter_x in range(0, filter_size):

                    print(f'{x}, {y}')

                    if filter_x > w or filter_x < 0 or filter_y > h or filter_y < 0:
                        pass
                    else:
                        # pixel = np.ndarray(shape=3)
                        #
                        # pixel[0] = image[y, x][0] * filter[filter_y, filter_x]
                        # pixel[1] = image[y, x][1] * filter[filter_y, filter_x]
                        # pixel[2] = image[y, x][2] * filter[filter_y, filter_x]


                        # image[y, x][0] = 128 + image[y, x][0]
                        # image[y, x][1] = 128 + image[y, x][1]
                        # image[y, x][2] = 128 + image[y, x][2]

                        average_weighted_pixel += filter[filter_y, filter_x] * image[y,x]

            filtered_image[y, x] = average_weighted_pixel

            # image[y, x][0] = 128 + image[y, x][0]
            # image[y, x][1] = 128 + image[y, x][1]
            # image[y, x][2] = 128 + image[y, x][2]

    print("DOne!")

    cv2.imshow('Before', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imshow('After', filtered_image)


    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    # image_read_write()
    # pixel_operations()
    image_noise()
    # image_denoising()
    # image_filters()

    # main()
