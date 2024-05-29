from os import system, makedirs
from imageio.v3 import imread, imwrite
import numpy as np
from scipy.ndimage import gaussian_filter as gaussian
from scipy.ndimage import median_filter as median
from scipy.ndimage import sobel
from scipy.signal import medfilt2d
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.pyplot as plt
from time import time
start = time()



#-------- settings ----------------------
GRAYSCALE = True
SUBPIXEL = False
ANTI_PAD_RADIUS = 0

TIME_START = 0.6
TIME_END = 0.18

NUM_FRAMES = 100
FPS = 33
QUALITY = 100
LOSSY_QUALITY = 100
MOTION_QUALITY = 100

#-------- filter settings ---------------
POWER_VAL = 1
GRADIENT_MAP = False  # no gradient map
# GRADIENT_MAP = 'colormap11'

MOIRE_MAP = False
ORDER = 10
LERP = False


GAUSS_SIZE_MAX = 50
DIFF_GAUSS_SIZE_MAX = 30
DIFF_AMOUNT = 0.2

ENABLE_SOBEL = False

MEDIAN_SIZE = 11
FAST_MEDIAN = False
CUSTOM_KERNEL = False
KERNEL_COLOR = False
CIRCLE_WEIGHTS = True
FILL_CIRCLE = False
CIRCLE_THICKNESS = 1
#----------------------------------------



# float array -> unint8 array;
def to_uint8(x):
    return np.round(x*255).clip(0, 255).astype(np.uint8)

# array -> float array; normalize input array between 0 and 1.
def normalized(x):
    min_ = np.min(x) # np.percentile(x, 0.1)
    max_ = np.max(x) # np.percentile(x, 0.9)
    if max_ - min_ == 0:
        return x*0
    else:
        return (x - min_)/(max_ - min_)

# array -> array; perceptual lerp between two rgb arrays
# GAMMA = 2.36
# INV_GAMMA = 1/GAMMA
# def srgb_mix(a, b, mix=0.5, gamma=GAMMA):
#     a = np.power(a, gamma)
#     b = np.power(b, gamma)
#     c = np.add(a*(1-mix), b*mix)
#     return np.power(c, INV_GAMMA)

# 2d array, int -> 2d array; opposite of numpy.pad() function.
def anti_pad(x, r):
    return x[r:-r, r:-r]

# float -> 2d float array; generate antialiased circle.
if KERNEL_COLOR:
    def draw_antialiased_circle(size):
        system('draw_antialiased_circle.py')
        circle = imread('circle.png', mode='L', pilmode='RGB')
        return circle
else:
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
            
        return image

# 2d array -> 2d array; apply median filter.
if FAST_MEDIAN:
    def median_filter(x):
            return medfilt2d(x, kernel_size=MEDIAN_SIZE)
else:
    if CUSTOM_KERNEL:
        FOOTPRINT = imread('circle.png', mode='L', pilmode='RGB')
        def median_filter(x):
            if KERNEL_COLOR:
                return np.dstack((median(x[:,:,0], footprint=FOOTPRINT[:,:,0]),
                                  median(x[:,:,1], footprint=FOOTPRINT[:,:,1]),
                                  median(x[:,:,2], footprint=FOOTPRINT[:,:,2])))
            else:
                return median(x, footprint=FOOTPRINT)
    elif CIRCLE_WEIGHTS:
        FOOTPRINT = draw_antialiased_circle(MEDIAN_SIZE)
        def median_filter(x):
            if KERNEL_COLOR:
                return np.dstack((median(x[:,:,0], footprint=FOOTPRINT[:,:,0]),
                                  median(x[:,:,1], footprint=FOOTPRINT[:,:,1]),
                                  median(x[:,:,2], footprint=FOOTPRINT[:,:,2])))
            else:
                return median(x, footprint=FOOTPRINT)
    else:
        def median_filter(x):
            return median(x, size=MEDIAN_SIZE)

# def median_filter_2(x):
#     pass

def generate_cdict(image):
    flattened_data = image.flatten()
    length = len(flattened_data)
    cdict = {'red': [], 'green': [], 'blue': []}
    for i in range(length):
        value = flattened_data[i]
        if i == 0:
            cdict['red'].append((0.0, value, value))
            cdict['green'].append((0.0, value, value))
            cdict['blue'].append((0.0, value, value))
        elif i == length - 1:
            cdict['red'].append((1.0, value, value))
            cdict['green'].append((1.0, value, value))
            cdict['blue'].append((1.0, value, value))
        else:
            x = i / (length - 1)
            cdict['red'].append((x, value, value))
            cdict['green'].append((x, value, value))
            cdict['blue'].append((x, value, value))
    return cdict

# 2d array -> 2d uint8 array; apply gradient map.
if MOIRE_MAP:
    GRADIENT = imread('colormaps/colormap10.png', mode='L', pilmode='RGB')[0,:2*ORDER,:]/255
    if LERP:
        cdict = generate_cdict(GRADIENT)
        colormap = LinearSegmentedColormap('custom_colormap', cdict)
    else:
        colormap = ListedColormap(GRADIENT)
elif GRADIENT_MAP:
    GRADIENT = imread('colormaps/'+GRADIENT_MAP+'.png', mode='L', pilmode='RGB')[0,:,:]/255
    if LERP:
        cdict = generate_cdict(GRADIENT)
        colormap = LinearSegmentedColormap('custom_colormap', cdict)
    else:
        colormap = ListedColormap(GRADIENT)
else:
    colormap = None
def gradient_map(x):
    aximg = plt.imshow(x, cmap=colormap)
    return aximg.make_image(renderer=None, unsampled=True)[0]

# 2d array -> 2d array; apply gaussian blur with optional sobel filters.
if ENABLE_SOBEL:
    def gaussian_filter(x, radius):
        sobel_h = sobel(x, 0)
        sobel_v = sobel(x, 1)
        x = np.sqrt(sobel_h**2 + sobel_v**2)
        x = gaussian(x, radius, mode='constant')
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

# initialize directory for .png sequence

makedirs('outputs', exist_ok=True)
system(f'cd outputs && del *.png 2>NUL')


print(f'initalized. t={time()-start}')
total_time = 0

if GRAYSCALE:
    for i, time_value in enumerate(np.linspace(TIME_START, TIME_END, NUM_FRAMES)):
        # determine linear radius
        gauss_size = (np.power(2, time_value)-1)*GAUSS_SIZE_MAX+1
        diff_gauss_size = (np.power(2, time_value)-1)*DIFF_GAUSS_SIZE_MAX+1

        # apply difference of gaussians
        filtered_img1 = gaussian_filter(input_img[:,:,0], gauss_size)
        diff = gaussian_filter(input_img[:,:,0], diff_gauss_size)
        filtered_img1 = np.abs(filtered_img1 - diff*DIFF_AMOUNT)
        filtered_img1 = normalized(filtered_img1)
        filtered_img1 = np.repeat(filtered_img1[:, :, np.newaxis], 3, axis=2)

        # apply median
        filtered_img2 = median_filter(filtered_img1)
        filtered_img3 = anti_pad(np.abs((filtered_img2 - filtered_img1)), ANTI_PAD_RADIUS)
        filtered_img3 = np.power(normalized(filtered_img3), POWER_VAL)

        if GRADIENT_MAP:
            if KERNEL_COLOR:
                pass
                # filtered_img3 = gradient_map(filtered_img3[:,:,0])
            else:
                filtered_img3 = gradient_map(filtered_img3[:,:,0])
        else:
            filtered_img3 = to_uint8(filtered_img3)
        
        imwrite(f'outputs/{i}.png', filtered_img3)
        if SUBPIXEL:
            system(f'subpixeler "outputs/{i}.png"')

        t = time()-start
        start = time()
        total_time += t
        print(f'frame {i} done. t={t}')
        
start = time()
# convert .png sequence to .gif
if NUM_FRAMES > 1:
    system(f'gifski \
        --quality={QUALITY} \
        --lossy-quality={LOSSY_QUALITY} \
        --motion-quality={MOTION_QUALITY} \
        --extra \
        --fps {FPS} \
        -o output.gif \
        outputs/*.png \
        ')

t = time()-start
total_time += t
print(f'total={total_time}')

# gifski --quality=100 --lossy-quality=100 --motion-quality=100 --extra --fps 36 -o output.gif outputs/*.png

#         filtered_img1 = np.dstack((gaussian_filter(input_img[:,:,0], gauss_size),
#                                    gaussian_filter(input_img[:,:,1], gauss_size),
#                                    gaussian_filter(input_img[:,:,2], gauss_size)))
#         filtered_img11 = np.dstack((gaussian_filter(input_img[:,:,0], diff_gauss_size),
#                                     gaussian_filter(input_img[:,:,1], diff_gauss_size),
#                                     gaussian_filter(input_img[:,:,2], diff_gauss_size)))
#         filtered_img2 = np.dstack((median_filter(filtered_img1[:,:,0]),
#                                    median_filter(filtered_img1[:,:,1]),
#                                    median_filter(filtered_img1[:,:,2])))





# def apply_filters(x, t):
#         gauss_size = (np.power(2, t)-1)*GAUSS_SIZE_MAX+1
#         diff_gauss_size = (np.power(2, t)-1)*DIFF_GAUSS_SIZE_MAX+1

#         if PRE_SOBEL:
#             x = sobel_filter(x)

#         # apply difference of gaussians
#         a = gaussian_filter(x, gauss_size)
#         b = gaussian_filter(x, diff_gauss_size)
#         x = np.abs(a - b*DIFF_AMOUNT)

#         if POST_SOBEL:
#             x = sobel_filter(x)

#         x = normalized(x)

#         # apply median
#         x = anti_pad(np.abs((median_filter(x) - x)), ANTI_PAD_RADIUS)
#         x = np.power(normalized(x), POWER_VAL)

#         if GRADIENT_MAP:
#             return gradient_map(x)
#         return x

# if GRAYSCALE:
#     for i, time in enumerate(np.linspace(TIME_START, TIME_END, NUM_FRAMES)):
#         # determine linear radius

#         output_img = apply_filters(input_img[:,:,0], time)
        
#         imwrite(f'outputs/{i}.png', to_uint8(output_img))
#         if SUBPIXEL:
#             system(f'subpixeler "outputs/{i}.png"')
#         print(f'frame {i} done')