from matplotlib import pyplot as plt
import numpy as np
import os
import glob
import cv2
import sys
from scipy import stats
from scipy.signal import argrelextrema

##########################################################################
#DHM background image analysis
def dhm_background_image_analysis(f, img1, img2, img3, conv):

    def_files = sorted(glob.glob('*.txt'), key=os.path.getmtime)
    n = np.shape(def_files)[0]                      #n = 2000

    x_px = np.shape(np.loadtxt(def_files[0]))[0]    #x_px = 900
    y_px = np.shape(np.loadtxt(def_files[0]))[1]    #y_px = 900

    #def_img = np.zeros((img3,x_px,y_px))           #Shows memory error. Why??
    avg_def = np.zeros((img3+1,2))
    avg_back = np.zeros((x_px,y_px))

    count = 0

    for i in range(0,img3+1):
        print(def_files[i])
        def_img = np.loadtxt(def_files[i])
        avg_def[i,0] = i
        avg_def[i,1] = def_img.mean()
        if i<img1:
            count = count + 1
            avg_back = avg_back + def_img

    avg_back = avg_back/(count)
    image = avg_back - np.loadtxt(def_files[img2+1])

    return image, avg_def, avg_back

##########################################################################

##########################################################################
#Deformation center at (img2+1)th image
def deformation_center(image):

    x_px = np.shape(image)[0]
    y_px = np.shape(image)[1]

    s = int(round((x_px + y_px)/4))                                 #s = 450

    delta_theta = round(np.degrees(np.arcsin(4/s)),1)   #delta_theta = 0.5
    n = int(round(360/delta_theta))                     #n = 720

    ph = np.zeros((n,s))
    s_max = np.zeros(n)
    x_max = np.zeros(n)
    y_max = np.zeros(n)

    for k in range(0,n):
        theta = k*(360/n)
        for j in range(0,s):
            xx = s+(j*np.cos(np.deg2rad(theta)))
            yy = s+(j*np.sin(np.deg2rad(theta)))
            ph[k,j] = image[int(round(yy)),int(round(xx))]

        s_max[k] = np.argmax(ph[k,:])
        x_max[k] = s + (s_max[k]*np.cos(np.deg2rad(theta)))
        y_max[k] = s + (s_max[k]*np.sin(np.deg2rad(theta)))

    xc = (max(x_max) + min(x_max))/2
    yc = (max(y_max) + min(y_max))/2

    return x_max, y_max, xc, yc, delta_theta
##########################################################################

##########################################################################
#read info file
def read_info_file(f):

    os.chdir(f)
    t = [x.split(' ')[3] for x in open('timestamps.txt').readlines()]

    os.chdir(os.getcwd() + r'/info')

    avg_back = np.loadtxt('avg_background.txt')
    avg_def = np.loadtxt('avg_deformation.txt')

    with open('details.txt') as f_lines:
        content = f_lines.readlines()

    img1 = int(content[1])
    img2 = int(content[2])
    img3 = int(content[3])

    xc = int(content[4])
    yc = int(content[5])

    return avg_back, avg_def, img1, img2, img3, xc, yc, t

##########################################################################

##########################################################################

def smooth(y, box_pts):

    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')

    return y_smooth

##########################################################################

##########################################################################

def power_law_fit(x, y):

    xx = np.log(x)
    yy = np.log(y)

    print()

    slope, intercept, *_ = stats.linregress(xx, yy)

    initial = np.exp(intercept)
    power = slope

    y_fit = initial*(x**power)

    return initial, power, y_fit

##########################################################################

##########################################################################

def average_profile(theta_start, theta_end, delta_theta, xc, yc, s, img):

    k1 = len(np.arange(theta_start, theta_end, delta_theta))
    k2 = len(range(0,s))

    haha = np.zeros(s)
    haha1 = np.zeros(s)
    count = -1

    for k in np.arange(theta_start, theta_end, delta_theta):
        count = count + 1
        theta = k
        b = image_profile(img, xc, yc, theta, s)
        haha = haha + b

    haha = haha/count

    haha1 = haha-np.mean(haha)

    theta_n = len(np.arange(theta_start, theta_end, delta_theta))
    radius_n = s

    output = np.zeros((theta_n,radius_n), dtype = int)

    for i in range(theta_n):
        theta = theta_start + (i*delta_theta)
        for j in range(radius_n):
            xx = int(round(xc+(j*np.cos(np.deg2rad(-theta)))))
            yy = int(round(yc+(j*np.sin(np.deg2rad(-theta)))))
            output[i,j] = img[yy,xx]

    index_minima_horizontal = np.array(argrelextrema(output, np.less, axis=0)).T

    print(theta_n, radius_n)
    plt.imshow(output, cmap='gray')
    plt.scatter(index_minima_horizontal[:,1], index_minima_horizontal[:,0], marker='.', color='red')
    plt.show()

    return haha1

##########################################################################

##########################################################################

def image_profile(img_in, xc, yc, theta, s):

    profile_out = np.zeros(s)

    for j in range(0,s):
        xx = int(round(xc+(j*np.cos(np.deg2rad(-theta)))))
        yy = int(round(yc+(j*np.sin(np.deg2rad(-theta)))))
        profile_out[j] = img_in[yy,xx]

    return profile_out

################################################################################

################################################################################

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

################################################################################

################################################################################

def sphere_to_cylinder(D,H,sigma, rho, g):
    lc = (sigma/(rho*g))**(1/2)
    k1 = (D*H)/(lc**2)
    k2 = D/lc

    coeff_4 = 9
    coeff_3 = 0
    coeff_2 = -((3*k1)+18)
    coeff_1 = 12
    coeff_0 = k2**2

    coeff = [coeff_4, coeff_3, coeff_2, coeff_1, coeff_0]
    d_ratio = np.roots(coeff)
    h_ratio = (2/3)*(d_ratio**-2)

    D_max = d_ratio
    h_min_cm = h_ratio/2

    return D_max, h_min_cm

################################################################################

################################################################################

def sphere_to_pancake(D,delta_h,rho,g,sigma):
    lc = (sigma/(rho*g))**(1/2)
    k = (D*delta_h)/(lc**2)

    coeff_4 = 1
    coeff_3 = 0
    coeff_2 = 0
    coeff_1 = (np.pi)/6
    coeff_0 = -((k+6)/12)**2

    coeff = [coeff_4, coeff_3, coeff_2, coeff_1, coeff_0]
    poly_roots = np.roots(coeff)

    for i in range(0,5):
        if poly_roots[i].real>0 and poly_roots[i].imag==0:
            R_cap = poly_roots[i].real
            r_cap = (((k+6)/12) - (poly_roots[i].real**2))/(np.pi*(poly_roots[i].real))
            break

    D_max_ratio = 2*(R_cap + r_cap)                                             #maximum pancake lateral length normalised with the sherical drop diameter
    h_max_ratio = 2*(r_cap)                                                     #maximum pancake height normalised with the sherical drop diameter

    return D_max_ratio, h_max_ratio

################################################################################

################################################################################

def max_ray_pixels(xc,yc,x_px,y_px):

    s = min(xc,yc,x_px-xc-1,y_px-yc-1)-1

    return s

################################################################################

################################################################################

def binning_data(data,bins):
    n = len(data)
    data_output = np.zeros(n)
    for i in range(0,n):
        if i < (bins-1):
            data_output[i] = sum(data[i:i+bins])/bins
        elif i > (n-bins):
            data_output[i] = sum(data[i-bins:i])/bins
        else:
            data_output[i] = sum(data[i-int(bins/2):i+int(bins/2)])/bins

    return data_output

################################################################################
