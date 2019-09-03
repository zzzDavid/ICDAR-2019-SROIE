import numpy


class TextBox(object):
    def __init__(self, line):
        line_split = line.strip().split(",", maxsplit=8)
        self.xspan = (int(line_split[0]), int(line_split[4]))
        self.yspan = (int(line_split[1]), int(line_split[5]))
        self.x = (self.xspan[0] + self.xspan[1]) / 2
        self.y = (self.yspan[0] + self.yspan[1]) / 2
        self.text = line_split[8]

    def __repr__(self):
        return self.text


class TextLine(object):
    def __init__(self, text_box=None):
        if isinstance(text_box, TextBox):
            self.text = [text_box.text]
            self.xs = [text_box.x]
            self.y = text_box.y
            self.yspan = text_box.yspan
        else:
            self.text = []
            self.xs = []
            self.yspan = None

    def insert(self, text_box):
        if not (
            (text_box.yspan[0] < self.y < text_box.yspan[1])
            and (self.yspan[0] < text_box.y < self.yspan[1])
        ):
            raise ValueError

        try:
            at = next(i for i, v in enumerate(self.xs) if v > text_box.x)
            self.text.insert(at, text_box.text)
            self.xs.insert(at, text_box.x)
        except StopIteration:
            self.text.append(text_box.text)
            self.xs.append(text_box.x)

        self.y = text_box.y
        self.yspan = text_box.yspan

    def __str__(self):
        return "\t".join(self.text)

    def __repr__(self):
        if self.yspan is None:
            repr_yspan = "[    ,    ] "
        else:
            repr_yspan = "[{:4d},{:4d}] ".format(self.yspan[0], self.yspan[1])

        repr_text = "\t".join(self.text)

        return repr_yspan + repr_text


if __name__ == "__main__":
    pass
