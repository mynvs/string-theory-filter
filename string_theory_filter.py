from os import system, mkdir
from imageio.v3 import imread, imwrite
import numpy as np
from scipy.ndimage import gaussian_filter as gaussian
from scipy.ndimage import median_filter as median
from scipy.ndimage import sobel
from scipy.signal import medfilt2d
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from scipy.signal import resample




#-------- settings -----------------------------------
GRAYSCALE = True
SUBPIXEL = False
ANTI_PAD_RADIUS = 0
TIME_LOWER_BOUND = 0.1
TIME_UPPER_BOUND = 0.75

NUM_FRAMES = 60
QUALITY = 100
LOSSY_QUALITY = 100
MOTION_QUALITY = 100
FPS = 36

# GRADIENT_MAP = False  # no gradient map
GRADIENT_MAP = 'colormap5'

#-------- filter settings ----------------------------
GAUSS_SIZE_MAX = 38*4.2*0.7

DIFF_GAUSS_SIZE_MAX = 17*4.2*0.7
DIFF_AMOUNT = 0.15

POWER_VAL = 1
ENABLE_SOBEL = False
PRE_SOBEL = True
POST_SOBEL = True


MEDIAN_SIZE = 9
# MEDIAN_TIERS = 3

FAST_MEDIAN = False
CIRCLE_WEIGHTS = True
FILL_CIRCLE = True
CIRCLE_THICKNESS = 1
#-----------------------------------------------------




# float array -> unint8 array;
def to_uint8(x):
    return np.round(x*255).clip(0, 255).astype(np.uint8)

# array -> float array; normalizes input array between 0 and 1.
def normalized(x):
    min_ = np.min(x) # np.percentile(x, 0.1)
    max_ = np.max(x) # np.percentile(x, 0.9)
    return (x - min_)/(max_ - min_)

# 2d array, int -> 2d array; opposite of numpy.pad() function.
def anti_pad(x, r):
    return x[r:-r, r:-r]

# def lightness_map(x):
#     # return x
#     a = x*x
#     return 1.19040961067*x - 1.56545604886*a + 2.74955818196*a*x - 1.38838695488*a*a

def draw_antialiased_circle(size):
    ceil_size = int(np.ceil(size))
    image = np.zeros((ceil_size, ceil_size, 1), dtype=np.float32)

    center = (ceil_size/2 - 0.5, ceil_size/2 - 0.5)
    radius = size/2 - 0.5

    if FILL_CIRCLE:
        for y in range(ceil_size):
            for x in range(ceil_size):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)

                if dist <= radius:
                    image[y, x] = 1
                elif dist <= radius + 1:
                    alpha = 1 - (dist - radius)
                    image[y, x] = alpha
    else:
        for y in range(ceil_size):
            for x in range(ceil_size):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)

                if dist < radius-CIRCLE_THICKNESS:
                    image[y, x] = 0
                elif dist < radius-CIRCLE_THICKNESS+1:
                    alpha = (dist - radius)%1
                    image[y, x] = alpha
                elif radius-CIRCLE_THICKNESS+1 <= dist <= radius:
                    image[y, x] = 1
                elif dist <= radius + 1:
                    alpha = 1 - (dist - radius)
                    image[y, x] = alpha
        
    return image[:,:,0]/np.max(image[:,:,0])


# INV_GAMMA = 1/2.36
# def srgb_mix(a, b, mix=0.5, gamma=2.36):
#     a = np.power(a, gamma)
#     b = np.power(b, gamma)
#     c = a*(1-mix) + b*mix
#     return np.power(c, INV_GAMMA)

# 2d array -> 2d array; applies median filter to input array.



# MEDIAN

if FAST_MEDIAN:
    def median_filter(x):
            return medfilt2d(x, kernel_size=MEDIAN_SIZE)
else:
    if CIRCLE_WEIGHTS:
        FOOTPRINT = draw_antialiased_circle(MEDIAN_SIZE)
        def median_filter(x):
            return median(x, footprint=FOOTPRINT)
    else:
        def median_filter(x):
            return median(x, size=MEDIAN_SIZE)


# def median_filter_2(x):
#     pass


if GRADIENT_MAP:
    GRADIENT = imread('colormaps/'+GRADIENT_MAP+'.png', mode='L', pilmode='RGB')[0,:,:]/255
    GRADIENT = ListedColormap(GRADIENT)
    def gradient_map(x):
        aximg = plt.imshow(x, cmap=GRADIENT)
        return aximg.make_image(renderer=None, unsampled=True)[0]


if ENABLE_SOBEL:
    def gaussian_filter(x, radius):
        image_ = x
        if PRE_SOBEL:
            sobel_h = sobel(x, 0)
            sobel_v = sobel(x, 1)
            x = np.sqrt(sobel_h**2 + sobel_v**2)
        x = gaussian(image_, radius, mode='constant')
        if POST_SOBEL:
            sobel_h = sobel(x, 0)
            sobel_v = sobel(x, 1)
            x = np.sqrt(sobel_h**2 + sobel_v**2)
        return x
else:
     def gaussian_filter(x, radius):
        return gaussian(x, radius, mode='constant')



# scale gaussian blur radii to compensate for 1/3x scaling factor
if SUBPIXEL:
    GAUSS_SIZE_MAX *= 3
    DIFF_GAUSS_SIZE_MAX *= 3

ANTI_PAD_RADIUS += int(np.ceil(MEDIAN_SIZE*0.5))


# read 'input.png' as numpy array
if GRAYSCALE:
    input_img = imread('input.png', mode='F', pilmode='RGB').astype(np.float64)
else:
    input_img = imread('input.png', mode='L', pilmode='RGB').astype(np.float64)

# initialize directory for output images
try:
    mkdir('outputs')
except FileExistsError:
    pass
system(f'cd outputs && del *.png')


if GRAYSCALE:
    for i, val in enumerate(np.linspace(TIME_LOWER_BOUND, TIME_UPPER_BOUND, NUM_FRAMES)):
        # determine linear gaussian blur radius
        gauss_size = (np.power(2, val)-1)*GAUSS_SIZE_MAX+1
        diff_gauss_size = (np.power(2, val)-1)*DIFF_GAUSS_SIZE_MAX+1

        # apply gaussian blur
        filtered_img1 = gaussian_filter(input_img[:,:,0], gauss_size)
        filtered_img11 = gaussian_filter(input_img[:,:,0], diff_gauss_size)

        filtered_img1 = np.abs(filtered_img1 - filtered_img11*DIFF_AMOUNT)
        
        filtered_img1 = normalized(filtered_img1)

        filtered_img2 = median_filter(filtered_img1)

        filtered_img3 = anti_pad(np.abs(filtered_img2 - filtered_img1), ANTI_PAD_RADIUS)
        filtered_img3 = np.power(normalized(filtered_img3), POWER_VAL)

        if GRADIENT_MAP:
            filtered_img3 = gradient_map(filtered_img3)
        else:
            filtered_img3 = to_uint8(filtered_img3)
        print(i)

        imwrite(f'outputs/{i}.png', filtered_img3)
        if SUBPIXEL:
            system(f'subpixeler "outputs/{i}.png"')


system(f'gifski \
       --quality={QUALITY} \
       --lossy-quality={LOSSY_QUALITY} \
       --motion-quality={MOTION_QUALITY} \
       --extra \
       --fps {FPS} \
       -o output.gif \
       outputs/*.png \
       ')

# gifski --quality=100 --lossy-quality=100 --motion-quality=100 --extra --fps 36 -o output.gif outputs/*.png





# ARBITRARY RGB IMAGES (outdated)

# else:
#     for i, val in enumerate(np.linspace(0.005, 1, NUM_FRAMES)):
#         gauss_size = (np.power(2, val)-1)*GAUSS_SIZE_MAX+1
#         diff_gauss_size = (np.power(2, val)-1)*DIFF_GAUSS_SIZE_MAX+1

#         filtered_img1 = np.dstack((gaussian_filter(input_img[:,:,0], gauss_size),
#                                    gaussian_filter(input_img[:,:,1], gauss_size),
#                                    gaussian_filter(input_img[:,:,2], gauss_size)))
#         filtered_img11 = np.dstack((gaussian_filter(input_img[:,:,0], diff_gauss_size),
#                                     gaussian_filter(input_img[:,:,1], diff_gauss_size),
#                                     gaussian_filter(input_img[:,:,2], diff_gauss_size)))
        
#         filtered_img1 = np.abs(filtered_img1 - filtered_img11*DIFF_AMOUNT)

#         filtered_img2 = np.dstack((median_filter(filtered_img1[:,:,0]),
#                                    median_filter(filtered_img1[:,:,1]),
#                                    median_filter(filtered_img1[:,:,2])))

#         filtered_img3 = np.abs(filtered_img2 - filtered_img1)
#         filtered_img3 = np.power(normalized(filtered_img3), POWER_VAL)

#         imwrite(f'outputs/{i}.png', to_uint8(filtered_img3))
#         if SUBPIXEL:
#             system(f'subpixeler "outputs/{i}.png"')
#         print(i)