"""
Created on Wed Dec  6 02:25:08 2017

@author: costalongam
"""

import argparse # for command line execution
import os, sys
import numpy as np
import pandas as pd

# Custom modules
from dhm_time import *

def get_params(argv):
    """
    Function called to read and parse command line arguments

    Arguments
    ----------
    argv: list
        Command line arguments as passed by sys.argv[1:]

    Returns
    --------
    args: Namespace object returned by `argparse`
            Contains the values for all necessary parameters

    Possible flags
    ----------------
    -p = PATH ; --path = path
    -ns = SAMPLE REFRACTIVE INDEX ; --ns = ns
    -nref = REFERENCE REFRACTIVE INDEX ; --nref = nref
    -trans = DHM-T USED ; --trans == true/false
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', help='Folder containing the datas, Holograms, Intensity and Phase',
                        default='.', type=str)
    parser.add_argument('-ns', '--nsample', help='Sample refractive index',
                        default='1.403', type=float)
    parser.add_argument('-nref', '--nref', help='Reference refractive index',
                        default='1', type=float)
    parser.add_argument('-trans', '--transmission', action='store_true', help='DHM-T used')
    args = parser.parse_args(argv)
    args.path = os.path.normpath(os.path.expanduser(os.path.abspath(args.path)))

    return args

if __name__ == "__main__":
    """
    Note that the path should lead to a folder where there is at least a
    Phase/Image/ path containing the images to be stitched
    """
    cl_args = get_params(sys.argv[1:])

    # Make a stack
    phase_path = os.path.join(cl_args.path, 'Phase/Image/')

    # Calculate conversion and make the lists of images to be unwrapped
    if cl_args.transmission:
        conv = 666/(cl_args.nsample - cl_args.nref)
    else:
        conv = 666/(2*cl_args.nref)

    os.makedirs(os.path.join(cl_args.path, 'Unwrapped'))

    phase_images = os.listdir(phase_path)

    # Count the total number of files to be unwrapped
    total = len(phase_images)-1
    for file_name in phase_images:
        if file_name.startswith('.') and os.path.isfile(os.path.join(phase_path, file_name)):
            total -= 1

    # Unwrap images
    count = 0
    for file_name in phase_images:
        if not file_name.startswith('.') and os.path.isfile(os.path.join(phase_path, file_name)):
            count += 1
            im = get_pic(os.path.join(phase_path, file_name))

            # do not understand this part
            unwrapped = rescale_unwrap_im(unwrap_im(im), real_per=conv)
            ###############################

            file_name = file_name.replace('_phase.tif','')
            #np.savetxt(os.path.join(cl_args.path,'Unwrapped',file_name+'_unwrapped.txt'), unwrapped, delimiter='\t')
            np.savetxt(os.path.join(cl_args.path,'Unwrapped',file_name+'_unwrapped.txt'), unwrapped, fmt='%0.2f', delimiter='\t')

            print(count,'/',total)

    print('Done.')
