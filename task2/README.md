# Scanned Receipt OCR by Convolutional-Recurrent Neural Network

This is a `pytorch` implementation of CRNN, which is based on @meijieru's repository [here](https://github.com/meijieru/crnn.pytorch).

## Introduction

The CNN + RNN + CTC model visualisation:

<div align=center><img src="../Media/CTC.png" width="600"/></div>

We applied a modified CRNN in this task. CRNN is a conventional scene text recognition method including convolutional layers, bidirectional LSTM layers, and a transcription layer in sequence. 

<div align=center><img src="../Media/CRNN.png" width="300"/></div>

In scanned receipts each text usually contains several words. We add the blank space between words to the alphabet for LSTM prediction and thus improve the network from single word recognition to multiple words recognition. Moreover, we double the input image width to tackle the overlap problem of long texts after max-pooling and stack one more LSTM, enhancing the accuracy per character in the training set from 62% to 83%.

## Dependency

1. [warp-ctc-pytorch](https://github.com/SeanNaren/warp-ctc/tree/pytorch_bindings/pytorch_binding)
2. lmdb

## Prediction

As is shown in the introduction image, CRNN only accepts image that has single line of word. Therefore, we must provide bounding boxes for the whole reciept image.

1. Put image under folder `./data_test/` and bounding box text file under `./boundingbox/`, the name of image file and text file must correspond. The example test file can be found [here](https://drive.google.com/open?id=107WIMIzcD00EycMVy9VGvYiTGr_ySPOj)

2. Download pre-trained model from [here](https://drive.google.com/open?id=1X3_pNnLNEdwEcgiFrtwvc4uYXzkZ9Zjw) and put the weight file under `./expr/` folder 

3. To predict, just run `python main.py`. You can change the code inside to visualise output or prepare result for task 3.

example result:
```
tan chay yee
81750 masai johor
sales persor : fatin
tax invoice
total inclusive gst:
invoice no : pegiv1030765
email: ng@ojcgroup.com
bill to"
date
description
address
total:
cashier
:the peak quarry works
```

## Training

Training a CRNN requires converting data into `lmdb` format.

1. Divide training data into training and validating dataset, put each portion into `./data_tain/` and `./data_valid`. An example could be found [here](https://drive.google.com/open?id=1JKLh7Jq1VXVNW1InKJv6xUrc21zCQNpE)

2. Run `create_dataset.py` to create lmdb dataset. The created dataset could be found inside `./dataset`

3. After preparing dataset, just run:
   ```shell
   python train.py --adadelta --trainRoot {train_path} --valRoot {val_path} --cuda
   ```
   with desired options

4. Trained model output will be in `./expr/`