#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 15:45:16 2019

@author: viswanatha
"""

from utils import *
from datasets import PascalVOCDataset
from tqdm import tqdm
from pprint import PrettyPrinter
from mobilenet_ssd_priors import priors

pp = PrettyPrinter()
voc_labels = ('0','27', '28', '44', '46', '47', '48', '49', '50', '51', '62', '63', '72', '73', '74', '75', '76', '77', '78', '84')
label_map = {k: v for v, k in enumerate(voc_labels)}



# Parameters
data_folder = "dataset"
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
batch_size = 64
workers = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = "./05-BEST_checkpoint_ssd300.pth.tar"
priors_cxcy = priors
priors_cxcy = priors_cxcy.to(device)

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint["model"]
model = model.to(device)

# Switch to eval mode
model.eval()

# Load test data
test_dataset = PascalVOCDataset(
    data_folder, label_map, split="test")
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=test_dataset.collate_fn,
    num_workers=workers,
    pin_memory=True,
)


def evaluate(test_loader, model):
    """
    Evaluate.
    :param test_loader: DataLoader for test data
    :param model: model
    """
    global priors_cxcy
    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes, det_labels, det_scores, true_boxes, true_labels = (
        [],
        [],
        [],
        [],
        []
    )
    # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels) in enumerate(
            tqdm(test_loader, desc="Evaluating")
        ):
            images = images.to(device)  # (N, 3, 300, 300)

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = detect_objects(
                priors_cxcy,
                predicted_locs,
                predicted_scores,
                min_score=0.01,
                max_overlap=0.45,
                top_k=200,
                n_classes=20
            )
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos

            # Store this batch's results for mAP calculation
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]
            # difficulties = [d.to(device) for d in difficulties]

            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.extend(boxes)
            true_labels.extend(labels)
            # true_difficulties.extend(difficulties)

        # Calculate mAP
        APs, mAP = calculate_mAP(
            det_boxes,
            det_labels,
            det_scores,
            true_boxes,
            true_labels,
            # true_difficulties,
        )

    # Print AP for each class
    pp.pprint(APs)

    print("\nMean Average Precision (mAP): %.3f" % mAP)


if __name__ == "__main__":
    evaluate(test_loader, model)
