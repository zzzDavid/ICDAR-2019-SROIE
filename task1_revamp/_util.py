def iou(box_1, box_2):
    """Intersection Over Union

    Input:
    box_1 -- a sequence like [x_0, y_0, x_1, y_1], representing top-left and bottom-right points of a box
    box_2 -- same as box_2
    """

    min_x_0 = min(box_1[0], box_2[0])
    min_y_0 = min(box_1[1], box_2[1])
    min_x_1 = min(box_1[2], box_2[2])
    min_y_1 = min(box_1[3], box_2[3])

    max_x_0 = max(box_1[0], box_2[0])
    max_y_0 = max(box_1[1], box_2[1])
    max_x_1 = max(box_1[2], box_2[2])
    max_y_1 = max(box_1[3], box_2[3])

    area_i = (min_x_1 - max_x_0) * (min_y_1 - max_y_0)
    area_o = (max_x_1 - min_x_0) * (max_y_1 - min_y_0)

    return area_i / area_o


if __name__ == "__main__":
    for i in range(10):
        print(11 // (0.7 ** i))
