"""
Title: label
Main Author: Shengjie Xiu
Time: 2019/3/30
Purpose: To draw box on the ICDAR datasets
Environment: python3.5.6 pytorch1.0.1 cuda9.0
"""

import cv2
import os
import shutil
import numpy as np
from matplotlib import pyplot as plt


image_path = "../ICDAR_Dataset/0325updated.task1train(626p)/0325updated.task1train(626p)/"
box_path = image_path + 'box/'

# gain the image list


def ListFilesToTxt(dir, file, wildcard, recursion):
    exts = wildcard.split(" ")
    files = os.listdir(dir)
    for name in files:
        fullname = os.path.join(dir, name)
        if (os.path.isdir(fullname) & recursion):
            ListFilesToTxt(fullname, file, wildcard, recursion)
        else:
            for ext in exts:
                if (name.endswith(ext)):
                    file.write(name + "\n")
                    break


def list():
    outfile = box_path + "jpglist.txt"
    wildcard = ".jpg"
    file = open(outfile, "w")
    if not file:
        print("cannot open the file %s for writing" % outfile)
    ListFilesToTxt(image_path, file, wildcard, 1)   # list the img in the file
    file.close()



# draw box
def draw():
    f = open(box_path + 'jpglist.txt')

    rect_scale_pack=[]
    rect_ratio_pack=[]

    # read each image and its label
    line = f.readline()
    line_num =0
    while line:
        line_num=line_num+1
        print('Image:', line_num)
        name = line.strip('\n')
        img = cv2.imread(image_path + name)
        img_size = img.shape
        img_size = img_size[0]*img_size[1]

        # read each coordinate and draw box
        f_txt = open(image_path + name.strip('.jpg') + '.txt')
        line_txt = f_txt.readline()
        while line_txt:
            coor = line_txt.split(',')
            x1 = int(coor[0].strip('\''))
            y1 = int(coor[1].strip('\''))
            x3 = int(coor[4].strip('\''))
            y3 = int(coor[5].strip('\''))
            text = coor[8].strip('\n').strip('\'')

            rect_size = (x3-x1)*(y3-y1)
            rect_scale = np.sqrt(rect_size / img_size)
            rect_scale_pack.append(rect_scale)
            #print(rect_scale_pack)

            rect_ratio = (x3-x1)/(y3-y1)
            rect_ratio_pack.append(rect_ratio)


            #cv2.rectangle(img, (x1, y1), (x3, y3), (255, 0, 0), 1)
            #cv2.putText(img, text, (x1, y1 - 1),
                        #cv2.FONT_HERSHEY_TRIPLEX, 0.35, (0, 0, 255), 1)
            line_txt = f_txt.readline()
        #cv2.imwrite(box_path + name, img)
        line = f.readline()
        # img = cv2.imshow('image', img)
        # cv2.waitKey(0)
    return rect_scale_pack,rect_ratio_pack

if __name__ == '__main__':
    if os.path.exists(box_path):
        shutil.rmtree(box_path)
    os.mkdir(box_path)
    list()  # list all the image in image_path into a txt
    rect_scale_pack, rect_ratio_pack = draw()  # draw the box based on the list
    #np.histogram(rect_scale_pack)  # have a statistic of the scale distribution
    print(rect_scale_pack)
    plt.figure(1)
    plt.hist(rect_scale_pack, bins=100)
    plt.xlabel('scale')
    plt.ylabel('num of image')
    plt.show()

    print(rect_ratio_pack)
    plt.figure(1)
    plt.hist(rect_ratio_pack, bins=20,range=(0,5))
    plt.xlabel('ratio')
    plt.ylabel('num of image')
    plt.show()


