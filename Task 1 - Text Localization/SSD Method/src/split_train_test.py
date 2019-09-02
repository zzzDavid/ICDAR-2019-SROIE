"""
Title: split_train_test
Main Author: Michael Xiu
Time: 2019/3/31
Purpose: Split the dataset in ICDAR
Environment: python3.5.6 pytorch1.0.1 cuda9.0
"""

import os
import shutil
import random

image_path = "../ICDAR_Dataset/0325updated.task1train(626p)/0325updated.task1train(626p)/"
train_path = "../ICDAR_Dataset/0325updated.task1train(626p)/train1/"
test_path = "../ICDAR_Dataset/0325updated.task1train(626p)/test1/"
train_test_ratio = 24


def MoveFile(dir, file_train, file_test, wildcard, recursion):
    exts = wildcard.split(" ")
    files = os.listdir(dir)
    num = 0
    for name in files:
        fullname = os.path.join(dir, name)
        if (os.path.isdir(fullname) & recursion):
            MoveFile(fullname, file_train, file_test, wildcard, recursion)
        else:
            for ext in exts:
                if (name.endswith(ext)):
                    name = name.strip('.jpg')
                    srcfile_img = dir + name + '.jpg'
                    srcfile_label = dir + name + '.txt'
                    if (num + 1) % (train_test_ratio + 1) != 0:
                        dstfile_img = train_path + name + '.jpg'
                        dstfile_label = train_path + name + '.txt'
                        shutil.copyfile(srcfile_img, dstfile_img)
                        shutil.copyfile(srcfile_label, dstfile_label)
                        file_train.write(name + "\n")
                        num = num + 1
                    else:
                        dstfile_img = test_path + name + '.jpg'
                        dstfile_label = test_path + name + '.txt'
                        shutil.copyfile(srcfile_img, dstfile_img)
                        shutil.copyfile(srcfile_label, dstfile_label)
                        file_test.write(name + "\n")
                        num = num + 1
                    break


def run():
    outfile_train = train_path + "train.txt"
    outfile_test = test_path + "test.txt"
    wildcard = ".jpg"
    file_train = open(outfile_train, "w")
    file_test = open(outfile_test, "w")
    MoveFile(image_path, file_train, file_test, wildcard, 1)
    file_train.close()
    file_test.close()


if __name__ == '__main__':
    if os.path.exists(train_path):
        shutil.rmtree(train_path)
    os.mkdir(train_path)
    if os.path.exists(test_path):
        shutil.rmtree(test_path)
    os.mkdir(test_path)
    run()
