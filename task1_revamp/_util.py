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

    # TODO: delete anchors with a text score below threshold
    # wait, can we do that?


def _vertical_nms(
    text_scores: Tensor,
    coordinates: Tensor,
    confidence_threshold=0.75,
    suppression_threshold=0.5,
):
    """Apply NMS to each column of anchors.

    Inputs:
        text_scores: [Na]
        coordinates: [Na x 2]
    """

    is_active = torch.ones_like(text_scores, dtype=torch.bool)
    is_active[text_scores < confidence_threshold] = False

    indices = torch.argsort(text_scores)

    for i in indices:
        if not is_active[i]:
            continue

        iou = _vertical_iou(coordinates[i], coordinates)

        is_active[iou > suppression_threshold] = False
        is_active[i] = True

    return is_active


def _vertical_iou(anchor_0: Tensor, anchors: Tensor):
    """Intersection Over Union for vertical anchors
    """
    half_0 = anchor_0[1] / 2
    halfs = anchors[:, 1] / 2

    top_0 = anchor_0[0] - half_0
    tops = anchors[:, 0] - halfs

    bottom_0 = anchor_0[0] + half_0
    bottoms = anchors[:, 0] + halfs

    min_top = torch.min(top_0, tops)
    max_top = torch.max(top_0, tops)

    min_bottom = torch.min(bottom_0, bottoms)
    max_bottom = torch.max(bottom_0, bottoms)

    area_i = min_bottom - max_top
    area_o = max_bottom - min_top

    return area_i / area_o


# def _iou(box_1, box_2):
#     """Intersection Over Union

#     Input:
#         box_1 -- a sequence like [x_0, y_0, x_1, y_1], representing top-left and bottom-right corners of a box
#         box_2 -- same as box_1
#     """

#     min_x_0 = min(box_1[0], box_2[0])
#     min_y_0 = min(box_1[1], box_2[1])
#     min_x_1 = min(box_1[2], box_2[2])
#     min_y_1 = min(box_1[3], box_2[3])

#     max_x_0 = max(box_1[0], box_2[0])
#     max_y_0 = max(box_1[1], box_2[1])
#     max_x_1 = max(box_1[2], box_2[2])
#     max_y_1 = max(box_1[3], box_2[3])

#     area_i = (min_x_1 - max_x_0) * (min_y_1 - max_y_0)
#     area_o = (max_x_1 - min_x_0) * (max_y_1 - min_y_0)

#     return area_i / area_o


if __name__ == "__main__":
    x = torch.randn(2, 3)
    y = torch.argsort(x)

    print(x)
