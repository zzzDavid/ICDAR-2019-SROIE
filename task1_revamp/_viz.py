import torch
from PIL import ImageDraw
from torchvision import transforms


def viz_boxes(img, boxes):
    """Visualize boxes

    Input:
    - img: image tensor
    - boxes: a tensor of [N x 4]. N is the number of boxes. Each row is a box like [x_0, y_0, x_1, y_1]
    """
    with transforms.functional.to_pil_image(img) as im:
        draw = ImageDraw.Draw(im)
        outline = (255, 0, 0)  # outline color in RGB
        for box in boxes:
            xy = box.round().type(torch.long).numpy()
            draw.rectangle(xy, outline=outline)


# TEST
if __name__ == "__main__":
    import random
    from _data import Task1Dataset

    dataset = Task1Dataset("data/img", "data/box", 1)

    img, tgt_1, tgt_2, idx_2, tgt_3, idx_3 = random.choice(dataset)
