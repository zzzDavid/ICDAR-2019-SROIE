import torch
from torch.nn import functional

from torch import Tensor


def outputs_to_boxes(
    y_1: Tensor, y_2: Tensor, y_3: Tensor, anchors: Tensor, threshold=0.75
):
    """Transform CTPN model outputs (see _model.CtpnModel.forward) to a tensor of shape [N x 4].

    Inputs:
        y_1 -- [N x H x W x k x 2], where `k` is the number of anchors. Predicted text/non-text scores.
        y_2 -- [N x H x W x k x 2]. Predicted vertical coordinates.
        y_3 -- [N x H x W x k]. Predicted side-refinement offsets.
        anchors -- list(tensor) of height of each anchor
        threshold
    """

    # apply softmax on text/non-text socres so that they lie in the range [0, 1] and sum to 1
    # and take the text scores only
    y_1 = functional.softmax(y_1, dim=4)[..., 0]
    # y_1: [N x H x W x k]

    # transform y_2 from [v_c, v_h] to [c_y, h]
    center_y_of_anchors = (torch.arange(y_2.size(1)) * 16 + 8).view(1, -1, 1, 1)
    y_2[..., 0].mul_(anchors).add_(center_y_of_anchors)
    y_2[..., 1].exp_().mul_(anchors)


def _vertical_nms(text_scores: Tensor, coordinates: Tensor):
    """Apply NMS to each column of anchors.

    Inputs:
        text_scores: [N x H x k]
        coordinates: [N x H x k x 2]
    """
    pass


def _iou(box_1, box_2):
    """Intersection Over Union

    Input:
        box_1 -- a sequence like [x_0, y_0, x_1, y_1], representing top-left and bottom-right corners of a box
        box_2 -- same as box_1
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
    x = torch.rand(1, 4, 3)
    w = torch.rand(1, 4, 3, 2)
    y = x > 0
    # y = x.unsqueeze(3).expand(1, 4, 3, 2) > 0

    print(w)
    print(y)
    print(w[y].view(-1, 2))
    # print(y > 0)
