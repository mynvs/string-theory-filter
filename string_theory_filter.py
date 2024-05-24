from imageio.v3 import imread, imwrite
import numpy as np
from os import system, mkdir
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt2d

def median(x):
    return medfilt2d(x, kernel_size=MEDIAN_SIZE)

def to_uint8(x):
    return np.round(x*255).clip(0, 255).astype(np.uint8)

def normalized(x):
    s = int(np.ceil(MEDIAN_SIZE*0.5))
    height, width = x.shape[:2]
    min_, max_ = np.min(x[s:height-s, s:width-s]), np.max(x[s:height-s, s:width-s])
    return (x - min_)/(max_ - min_)



MEDIAN_SIZE = 9

GAUSS_SIZE_MAX = 38
DIFF_GAUSS_SIZE_MAX = 17
DIFF_AMOUNT = 0.2

POWER_VAL = 0.4

GRAYSCALE = True
SUBPIXEL = False



if SUBPIXEL:
    GAUSS_SIZE_MAX *= 3
    DIFF_GAUSS_SIZE_MAX *= 3

if GRAYSCALE:
    input_img = imread('input.png', mode='F', pilmode='RGB').astype(np.float64)
else:
    input_img = imread('input.png', mode='L', pilmode='RGB').astype(np.float64)

try:
    mkdir('outputs')
except FileExistsError:
    pass
system(f'cd outputs && del *.png')


for i, val in enumerate(np.linspace(0.005, 1, 200)):

    gauss_size = GAUSS_SIZE_MAX*val
    diff_gauss_size = DIFF_GAUSS_SIZE_MAX*val


    if GRAYSCALE:
        filtered_img1 = gaussian_filter(input_img[:,:,0], gauss_size)
        filtered_img11 = gaussian_filter(input_img[:,:,0], diff_gauss_size)
    else:              
        filtered_img1 = np.dstack((gaussian_filter(input_img[:,:,0], gauss_size),
                                   gaussian_filter(input_img[:,:,1], gauss_size),
                                   gaussian_filter(input_img[:,:,2], gauss_size)))
        filtered_img11 = np.dstack((gaussian_filter(input_img[:,:,0], diff_gauss_size),
                                    gaussian_filter(input_img[:,:,1], diff_gauss_size),
                                    gaussian_filter(input_img[:,:,2], diff_gauss_size)))

    filtered_img1 = np.abs(filtered_img1 - filtered_img11*DIFF_AMOUNT)
    filtered_img1 = normalized(filtered_img1)

    if GRAYSCALE:
        filtered_img2 = median(filtered_img1)
    else:
        filtered_img2 = np.dstack((median(filtered_img1[:,:,0]),
                                   median(filtered_img1[:,:,1]),
                                   median(filtered_img1[:,:,2])))

    filtered_img3 = np.abs(filtered_img2 - filtered_img1)
    filtered_img3 = np.power(normalized(filtered_img3), POWER_VAL)


    imwrite(f'outputs/{i}.png', to_uint8(filtered_img3))
    if SUBPIXEL:
        system(f'subpixeler "outputs/{i}.png"')
    print(i)


system('gifski --quality=100 --lossy-quality=100 --motion-quality=100 --extra --fps 50 -o output.gif outputs/*.png')
