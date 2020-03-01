# A Simple Method for Key Information Extraction as Character-wise Classification with LSTM

## Introduction
This is a method that tackles the key information extraction problem as a character-wise classification problem with a simple stacked bidirectional LSTM. The method first formats the text from an image into a single sequence. The sequence is then fed into a two-layer bidirectional LSTM to produce a classification label from 5 classes - 4 key information category and one "others" - for each character. The method is simple enough with just a two-layer bidirectional LSTM implemented in PyTorch, and proves to sufficient in understanding the context of a receipt text and outputting highly accurate results.

## Dependency

- PyTorch 
- numpy
- colorama

## Training

Training data is available at `./data/`. To train a model, just run this command at root directory
```shell
$ python ./src/train.py
```
or to test the model:

```shell
$ python ./src/test.py
```



The original repo of task 3 is [here](https://github.com/patrick22414/sroie-task3).