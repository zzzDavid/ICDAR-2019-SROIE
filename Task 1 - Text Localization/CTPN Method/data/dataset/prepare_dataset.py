import csv
import glob
import os
import random
import shutil
#from PIL import Image
#from skimage import io
import cv2

def get_data():
    filenames = [os.path.splitext(f)[0] for f in glob.glob("original/*.jpg")]
    jpg_files = [s + ".jpg" for s in filenames]
    txt_files = [s + ".txt" for s in filenames]

    for file in txt_files:
        boxes = []
        with open(file, "r", encoding="utf-8", newline="") as lines:
            for line in csv.reader(lines):
                boxes.append([line[0], line[1], line[6], line[7]])
        with open('mlt/label/' + file.split('/')[1], "w+") as labelFile:
            wr = csv.writer(labelFile)
            wr.writerows(boxes)

    for jpg in jpg_files:
        shutil.copy(jpg, 'mlt/image/')


if __name__ == "__main__":
    get_data()