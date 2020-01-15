import numpy as np
import pandas as pd
import os, sys
import operator # for sorting arrays with two criteria
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from skimage import io, restoration, exposure
from tqdm import tqdm # for loading bars
import itertools as it
import glob
import re
from os.path import join, basename
from PIL import Image

def get_pic(im):
    """
    Load an image

    Parameters
    ----------
    im: str
        Path to the pictures

    Returns
    -------
    pic: Numpy `ndarray`
        Image
    """
    pic = np.asarray(Image.open(im))
    return pic


def unwrap_im(im):
    """
    Unwrap a np.uint8 phase image. Attention : it doubles the phase jumps.

    Parameters
    ----------

    im: numpy.uint8
        8-bit grayscale phase image to be unwrapped

    Returns
    ----------
    numpy.uint8 unwrapped image

    """

    im1 = im

    # ax1 = plt.subplot(241)
    # ax1.imshow(im1,'gray')
    # ax1.set_title('[0, 255]')
    # plt.xlim(1,900)
    # plt.ylim(1,900)
    # plt.xticks([1, 900])
    # plt.yticks([1, 900])

    im2 = exposure.rescale_intensity(1.0*im1, in_range=(0, 255), out_range=(0, 4*np.pi))

    # ax2 = plt.subplot(242)
    # ax2.imshow(im2,'gray')
    # ax2.set_title(r'$[0, 4\pi]$')
    # plt.xlim(1,900)
    # plt.ylim(1,900)
    # plt.xticks([1, 900])
    # plt.yticks([1, 900])

    im3 = np.angle(np.exp(1j * im2))

    # ax3 = plt.subplot(243)
    # ax3.imshow(im3,'gray')
    # ax3.set_title(r'$[-\pi, +\pi]$')
    # plt.xlim(1,900)
    # plt.ylim(1,900)
    # plt.xticks([1, 900])
    # plt.yticks([1, 900])

    im4 = restoration.unwrap_phase(im3)

    # ax4 = plt.subplot(244)
    # ax4.imshow(im4,'gray')
    # ax4.set_title(r'Modified image: ??')
    # plt.xlim(1,900)
    # plt.ylim(1,900)
    # plt.xticks([1, 900])
    # plt.yticks([1, 900])

    return im4

def rescale_unwrap_im(imw, real_per):
    """
    Rescale an unwrap image, given the real value of a phase jump.

    Parameters
    ----------

    imw: numpy.uint8
        8-bit grayscale unwrapped phase image
    unwrap_per: float
        period of a phase jump
        Default: 2*np.pi
    real_per: float
        real scale corresponding to a phase jump
        Default: 333/2 (half the half-wavelength of the laser)

    Returns
    ----------
    numpy.uint8 rescaled unwrapped image

    """
    # unwrap_per=4*np.pi
    # ran = np.amax(imw)-np.amin(imw)
    # nper = np.floor(ran/unwrap_per)
    # rest = (ran - nper*unwrap_per)/ran

    # yoyo = exposure.rescale_intensity(imw - np.amin(imw),
    #                                   in_range=(0, ran),
    #                                   out_range=(0, real_per*(nper + rest)))

    im4 = imw

    unwrap_per=4*np.pi
    ran = np.amax(im4)-np.amin(im4)
    nper = np.floor(ran/unwrap_per)
    rest = (ran - nper*unwrap_per)/ran

    im5 = exposure.rescale_intensity(im4 - np.amin(im4),
                                      in_range=(0, ran),
                                      out_range=(0, real_per*(nper + rest)))

    # ax5 = plt.subplot(245)
    # ax5.margins(0.05)           # Default margin is 0.05, value 0 means fit
    # ax5.imshow(im5,'gray')
    # ax5.set_title(r'Modified image: ??')
    # plt.xlim(1,900)
    # plt.ylim(1,900)
    # plt.xticks([1, 900])
    # plt.yticks([1, 900])
    #
    # plt.show()

    return im5
