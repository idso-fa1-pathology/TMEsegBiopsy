import os
import argparse
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import math
from glob import glob

from predict_slide import generate_tme
from ss1_stich import ss1_stich
from ss1_post import postprocess_tme

# get arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', dest='data_dir', help='path to cws data')
parser.add_argument('-o', '--save_dir', dest='save_dir', help='path to save all output files', default=None)
parser.add_argument('-s', '--save_dir_ss1', dest='save_dir_ss1', help='path to save all ss1 files', default=None)
parser.add_argument('-sp', '--save_dir_ss1_post', dest='save_dir_ss1_post', help='path to save all files after post processing', default=None)
parser.add_argument('-p', '--pattern', dest='file_name_pattern', help='pattern in the files name', default='*.ndpi')
parser.add_argument('-c', '--color', dest='color_norm', help='color normalization', action='store_false')
parser.add_argument('-n', '--nfile', dest='nfile', help='the nfile-th file', default=150, type=int)
parser.add_argument('-ps', '--patch_size', dest='patch_size', help='the size of the patch', default=768, type=int)
parser.add_argument('-ins', '--input_size', dest='input_size', help='the size of the model input', default=384, type=int)
parser.add_argument('-nC', '--number_class', dest='nClass', help='how many classes to segment', default=6, type=int)
parser.add_argument('-sf', '--scale_factor', dest='scale', help='how many times to scale compared to x20', default=0.0625, type=float)
parser.add_argument('-nJ', '--number_pods', dest='nJob', help='how many pods to be used in K8s', default=32, type=int)
args = parser.parse_args()

datapath=args.data_dir
nfile=args.nfile
file_pattern=args.file_name_pattern
files = sorted(glob(os.path.join(datapath, file_pattern)))
njob = args.nJob


if len(files) <= 32:
        start_file = nfile
        end_file = nfile + 1
else:
        file_job = len(files) // njob +1
        start_file = nfile * file_job
        end_file = nfile * file_job + file_job

for i in range(start_file, end_file):
    ######step0: generate tiles
    ######step1: generate growth pattern for tiles
    generate_tme(datapath=args.data_dir, save_dir=args.save_dir, file_pattern=args.file_name_pattern, 
                color_norm=args.color_norm, nfile=i, patch_size=args.patch_size, 
                patch_stride=args.patch_size*0.5, input_size=args.input_size, nClass=args.nClass) #before is nfile=args.nfile

    #######step2: stich to ss1 level
    ss1_stich(cws_folder=args.data_dir, annotated_dir=args.save_dir, output_dir=args.save_dir_ss1, 
            nfile=i, file_pattern=args.file_name_pattern, downscale=args.scale) #before is nfile=args.nfile
    
    #######step3: refine ss1 mask
    postprocess_tme(cws_folder=args.data_dir, ss1_dir=args.save_dir_ss1, ss1_post_dir=args.save_dir_ss1_post, 
                    nfile=i, file_pattern=args.file_name_pattern)