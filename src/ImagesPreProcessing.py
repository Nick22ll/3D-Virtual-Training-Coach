from __future__ import print_function

import os

import argparse
import imutils
import cv2
from natsort import os_sorted


def convert_to_256(image_path):
    im = cv2.imread(image_path)
    height, width = im.shape[:2]
    if height > width:
        im = imutils.resize(im, height=256)
        width = im.shape[1]
        bordersize = int((256 - width) / 2)
        padded_im = cv2.copyMakeBorder(
            im,
            left=bordersize,
            right=bordersize,
            top=0,
            bottom=0,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )
    else:
        im = imutils.resize(im, width=256)
        height = im.shape[0]
        bordersize = int((256 - height) / 2)
        padded_im = cv2.copyMakeBorder(
            im,
            left=0,
            right=0,
            top=bordersize,
            bottom=bordersize,
            borderType=cv2.BORDER_CONSTANT,
            value=[0, 0, 0]
        )

    #cv2.imshow('padded_im', padded_im)
    return padded_im

def main():

    parser = argparse.ArgumentParser(description='Joint Dataset Converter', allow_abbrev=False)
    parser.add_argument('--directory', type=str, required=True)
    opts = parser.parse_args()
    directory = opts.directory
    for image in os_sorted(os.listdir(directory)):
        os.makedirs(directory.replace("Frames", "PaddedFrames"), exist_ok=True)
        cv2.imwrite(directory.replace("Frames", "PaddedFrames") + "/" + image, convert_to_256(opts.directory + "/" + image))

    print("Finita la cartella:" + directory)

if __name__ == "__main__":
    main()

# initialize the HOG descriptor/person detector



