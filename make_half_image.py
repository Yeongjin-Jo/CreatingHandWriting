#!/usr/bin/env python
# coding: utf-8

import argparse
from tqdm import tqdm
import os
from PIL import Image
import numpy as np

def make_half_image(experiment_dir, label):
    path = os.path.join(experiment_dir, 'infer', label)
    # path = './experiment_dir_399/infer/31'

    image_list = os.listdir(path)

    for image_name in tqdm(image_list):
        image = Image.open(os.path.join(path, image_name)).convert('L')
        half_image = np.array(image)[:, :256]
        half_image = Image.fromarray(half_image)
        half_image.save(os.path.join(path, image_name))


parser = argparse.ArgumentParser(description='make half image by cropping')
parser.add_argument('--experiment_dir', dest='experiment_dir', required=True, help='experiment_dir')
parser.add_argument('--label', required=True, help='label')

args = parser.parse_args()

if __name__ == "__main__":
    make_half_image(args.experiment_dir, args.label)
