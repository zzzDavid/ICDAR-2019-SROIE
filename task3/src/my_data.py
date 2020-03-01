import json
import os
import random
from os import path
from string import ascii_uppercase, digits, punctuation

import colorama
import numpy
import regex
import torch
from colorama import Fore
from torch.utils import data

from my_classes import TextBox, TextLine
from my_utils import robust_padding

VOCAB = ascii_uppercase + digits + punctuation + " \t\n"


class MyDataset(data.Dataset):
    def __init__(
        self, dict_path="data/data_dict.pth", device="cpu", val_size=76, test_path=None
    ):
        if dict_path is None:
            self.val_dict = {}
            self.train_dict = {}
        else:
            data_items = list(torch.load(dict_path).items())
            random.shuffle(data_items)

            self.val_dict = dict(data_items[:val_size])
            self.train_dict = dict(data_items[val_size:])

        if test_path is None:
            self.test_dict = {}
        else:
            self.test_dict = torch.load(test_path)

        self.device = device

    def get_test_data(self, key):
        text = self.test_dict[key]
        text_tensor = torch.zeros(len(text), 1, dtype=torch.long)
        text_tensor[:, 0] = torch.LongTensor([VOCAB.find(c) for c in text])

        return text_tensor.to(self.device)

    def get_train_data(self, batch_size=8):
        samples = random.sample(self.train_dict.keys(), batch_size)

        texts = [self.train_dict[k][0] for k in samples]
        labels = [self.train_dict[k][1] for k in samples]

        robust_padding(texts, labels)

        maxlen = max(len(t) for t in texts)

        text_tensor = torch.zeros(maxlen, batch_size, dtype=torch.long)
        for i, text in enumerate(texts):
            text_tensor[:, i] = torch.LongTensor([VOCAB.find(c) for c in text])

        truth_tensor = torch.zeros(maxlen, batch_size, dtype=torch.long)
        for i, label in enumerate(labels):
            truth_tensor[:, i] = torch.LongTensor(label)

        return text_tensor.to(self.device), truth_tensor.to(self.device)

    def get_val_data(self, batch_size=8, device="cpu"):
        keys = random.sample(self.val_dict.keys(), batch_size)

        texts = [self.val_dict[k][0] for k in keys]
        labels = [self.val_dict[k][1] for k in keys]

        maxlen = max(len(s) for s in texts)
        texts = [s.ljust(maxlen, " ") for s in texts]
        labels = [
            numpy.pad(a, (0, maxlen - len(a)), mode="constant", constant_values=0)
            for a in labels
        ]

        text_tensor = torch.zeros(maxlen, batch_size, dtype=torch.long)
        for i, text in enumerate(texts):
            text_tensor[:, i] = torch.LongTensor([VOCAB.find(c) for c in text])

        truth_tensor = torch.zeros(maxlen, batch_size, dtype=torch.long)
        for i, label in enumerate(labels):
            truth_tensor[:, i] = torch.LongTensor(label)

        return keys, text_tensor.to(self.device), truth_tensor.to(self.device)


def get_files(data_path="data/"):
    json_files = sorted(
        (f for f in os.scandir(data_path) if f.name.endswith(".json")),
        key=lambda f: f.path,
    )
    txt_files = sorted(
        (f for f in os.scandir(data_path) if f.name.endswith(".txt")),
        key=lambda f: f.path,
    )

    assert len(json_files) == len(txt_files)
    for f1, f2 in zip(json_files, txt_files):
        assert path.splitext(f1)[0] == path.splitext(f2)[0]

    return json_files, txt_files


def sort_text(txt_file):
    with open(txt_file, "r") as txt_opened:
        content = sorted([TextBox(line) for line in txt_opened], key=lambda box: box.y)

    text_lines = [TextLine(content[0])]
    for box in content[1:]:
        try:
            text_lines[-1].insert(box)
        except ValueError:
            text_lines.append(TextLine(box))

    return "\n".join([str(text_line) for text_line in text_lines])


def create_test_data():
    keys = sorted(
        path.splitext(f.name)[0]
        for f in os.scandir("tmp/task3-test(347p)")
        if f.name.endswith(".jpg")
    )

    files = ["tmp/text.task1&2-test(361p)/" + s + ".txt" for s in keys]

    test_dict = {}
    for k, f in zip(keys, files):
        test_dict[k] = sort_text(f)

    torch.save(test_dict, "data/test_dict.pth")


def create_data(data_path="tmp/data/"):

    json_files, txt_files = get_files(data_path)
    keys = [path.splitext(f.name)[0] for f in json_files]

    data_dict = {}

    for key, json_file, txt_file in zip(keys, json_files, txt_files):
        with open(json_file, "r", encoding="utf-8") as json_opend:
            key_info = json.load(json_opend)

        text = sort_text(txt_file)
        text_space = regex.sub(r"[\t\n]", " ", text)

        text_class = numpy.zeros(len(text), dtype=int)

        print()
        print(json_file.path, txt_file.path)
        for i, k in enumerate(iter(key_info)):
            v = key_info[k]
            if k == "total":
                s = regex.search(
                    r"(\bTOTAL[^C]*ROUND[^C]*)(" + v + r")(\b)", text_space
                )
                if s is None:
                    s = regex.search(r"(\bTOTAL[^C]*)(" + v + r")(\b)", text_space)
                    if s is None:
                        s = regex.search(r"(\b)(" + v + r")(\b)", text_space)
                        if s is None:
                            s = regex.search(r"()(" + v + r")()", text_space)
                v = s[2]
                text_class[range(*s.span(2))] = i + 1
            else:
                if not v in text_space:
                    s = None
                    e = 0
                    while s is None and e < 3:
                        e += 1
                        s = regex.search(
                            r"(\b" + v + r"\b){e<=" + str(e) + r"}", text_space
                        )
                    v = s[0]

                pos = text_space.find(v)
                text_class[pos : pos + len(v)] = i + 1

        data_dict[key] = (text, text_class)

        # print(txt_file.path)
        # color_print(text, text_class)

    return keys, data_dict


def color_print(text, text_class):
    colorama.init()
    for c, n in zip(text, text_class):
        if n == 1:
            print(Fore.RED + c, end="")
        elif n == 2:
            print(Fore.GREEN + c, end="")
        elif n == 3:
            print(Fore.BLUE + c, end="")
        elif n == 4:
            print(Fore.YELLOW + c, end="")
        else:
            print(Fore.WHITE + c, end="")
    print(Fore.RESET)
    print()


if __name__ == "__main__":
    create_test_data()

    # dataset = MyDataset("data/data_dict2.pth")
    # text, truth = dataset.get_train_data()
    # print(text)
    # print(truth)
    # dict3 = torch.load("data/data_dict3.pth")
    # for k in dict3.keys():
    #     text, text_class = dict3[k]
    #     color_print(text, text_class)

    # keys, data_dict = create_data()
    # torch.save(data_dict, "data/data_dict4.pth")

    # s = "START 0 TOTAL:1.00, START TOTAL: 1.00 END"
    # rs = regex.search(r"(\sTOTAL.*)(1.00)(\s)", s)
    # for i in range(len(rs)):
    #     print(repr(rs[i]), rs.span(i))
