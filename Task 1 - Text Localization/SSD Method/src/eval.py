import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # to specify the GPU_id in the remote server

from utils import *
from datasets import ICDARDataset
from tqdm import tqdm
from pprint import PrettyPrinter
import numpy
import torch

# Good formatting when printing the APs for each class and mAP
pp = PrettyPrinter()

# Parameters
data_folder = '../ICDAR_Dataset/0325updated.task1train(626p)/'
keep_difficult = False  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 64
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = 'best4.17.tar' #''BEST_checkpoint_ssd300.pth.tar'
test_img_num = 25

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()

# Load test data
test_dataset = ICDARDataset(data_folder,
                                split='test')
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                          collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)

def evaluate(test_loader, model, test_img_num):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():

        f1_max = 0
        ap_max = 0
        ar_max = 0

        for min_score in [0.1]:

            for max_overlap in [0.8]:

                # Batches
                f1 = 0
                ap = 0
                ar = 0
                images_num = 0

                for i, (images, boxes, labels) in enumerate(tqdm(test_loader, desc='Evaluating')):

                    if i < test_img_num:

                        images = images.to(device)  # (N, 3, 300, 300)
                        print(images)
                        print(images.size)

                        # Forward prop.
                        predicted_locs, predicted_scores = model(images)

                        # Detect objects in SSD output
                        det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,
                                                                                                   min_score=min_score, max_overlap=max_overlap,
                                                                                                   top_k=200, original_image = images, max_OCR_overlap=0.2, max_OCR_pixel=245)
                        # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

                        # Store this batch's results for mAP calculation
                        boxes = [b.to(device) for b in boxes]
                        labels = [l.to(device) for l in labels]

                        det_boxes.extend(det_boxes_batch)
                        det_labels.extend(det_labels_batch)
                        det_scores.extend(det_scores_batch)
                        true_boxes.extend(boxes)
                        true_labels.extend(labels)

                        f1_0, ap_0, ar_0 = calc_f1(det_boxes_batch[0], boxes[0], iou_thresh=0.5)
                        # print()
                        # print("F1: ", f1_0)
                        # print("AP: ", ap_0)
                        # print("AR: ", ar_0)
                        f1 += f1_0
                        ap += ap_0
                        ar += ar_0
                        images_num += 1

                if f1 / images_num > f1_max:
                    f1_max = f1/ images_num
                    f1_max_par = [min_score, max_overlap]
                print('f1 max:' ,f1_max)
                print('f1 max par:', f1_max_par )
                if ap / images_num > ap_max:
                    ap_max = ap/ images_num
                    ap_max_par = [min_score, max_overlap]
                print('ap max:' ,ap_max)
                print('ap max par:', ap_max_par )
                if ar / images_num > ar_max:
                    ar_max = ar/ images_num
                    ar_max_par = [min_score, max_overlap]
                print('ar max:' ,ar_max)
                print('ar max par:', ar_max_par )



        # Calculate mAP
        # APs, mAP = calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels)
        # print("Final F1: ", f1 / images_num)
        # print("Final AP: ", ap / images_num)
        # print("Final AR: ", ar / images_num)

    # Print AP for each class
    # pp.pprint(APs)

    #print('\nMean Average Precision (mAP): %.3f' % mAP)


if __name__ == '__main__':
    evaluate(test_loader, model, test_img_num)
