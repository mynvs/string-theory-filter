import numpy as np
from imageio.v3 import imread, imwrite
from cv2 import resize, COLOR_RGB2BGR, cvtColor, INTER_LANCZOS4 
from os import system

def linearize(x):
    return np.where(x <= 0.04045, x * 0.07739938080495356, np.power((x + 0.055) * 0.94786729857819905, 2.4))

def rgb_to_oklab_Luminance(rgb):
    rgb = linearize(rgb * 0.00392156862745098)
    L = np.cbrt(0.4122214708*rgb[..., 0] + 0.5363325363*rgb[..., 1] + 0.0514459929*rgb[..., 2])
    M = np.cbrt(0.2119034982*rgb[..., 0] + 0.6806995451*rgb[..., 1] + 0.1073969566*rgb[..., 2])
    S = np.cbrt(0.0883024619*rgb[..., 0] + 0.2817188376*rgb[..., 1] + 0.6299787005*rgb[..., 2])
    return 0.2104542553*L + 0.7936177850*M - 0.0040720468*S

def sort_colors_by_luminance(image):
    reshaped_image = image.reshape(-1, 3)
    luminance = rgb_to_oklab_Luminance(reshaped_image)
    indices = np.argsort(luminance)
    sorted_colors = reshaped_image[indices]

    return sorted_colors

input_img = imread('colormap.png', mode='L', pilmode='RGB').astype(np.uint8)
image = np.rot90(sort_colors_by_luminance(input_img))
image = np.dstack((image[2,:], image[1,:], image[0,:]))
length = int(round(np.sqrt(image.shape)[1])*8)
imwrite('colormaps/colormapi.png', image)
# system(f'ffmpeg -i colormap2.png -vf scale={length}:1:flags="bilinear" colormapipng -y')
# system('del "colormap2.png"')
