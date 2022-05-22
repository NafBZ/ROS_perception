import torch
from torch.utils.data import Dataset
import json
import os
from PIL import Image
from utils import transform


class PascalVOCDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, label_map, split):
        """
        :param data_folder: folder where data files are stored
        :param split: split, one of 'TRAIN' or 'TEST'
        :param keep_difficult: keep or discard objects that are considered difficult to detect?
        """
        self.split = split.upper()
        self.width = 300
        self.height = 300
        self.label_map = label_map

        assert self.split in {"TRAIN", "TEST"}

        self.data_folder = data_folder + '/' + self.split.lower()
        self.images = [self.data_folder + '/'+ image for image in sorted(os.listdir(self.data_folder)) if image[-4:]=='.jpg']

        # Read data files
#         with open(os.path.join(data_folder, self.split + "_images.json"), "r") as j:
#             self.images = json.load(j)
#         with open(os.path.join(data_folder, self.split + "_objects.json"), "r") as j:
#             self.objects = json.load(j)

#         assert len(self.images) == len(self.objects)

    def __getitem__(self, i):
        # Read image
        # print('--- I AM I', i, self.images[i])
        image = Image.open(self.images[i], mode="r")
        image = image.convert("RGB")
        

        # image_path = os.path.join(self.data_folder, self.split, image)
        
        annot_filename = self.images[i][:-4] + '.txt'
        annot_file_path = annot_filename

        boxes = []
        labels = []
        r_width = image._size[0]
        r_height = image._size[1]

        # box coordinates for txt files are extracted and corrected for image size given
        with open(annot_file_path) as f:
            for line in f:
                parsed = [float(x) for x in line.split(' ')]
                # if parsed[0] not in self.label_map:
                #     continue
                # labels.append(self.label_map[str(int(parsed[0]))]) # parsed the category id
                # x_left = parsed[1]*(self.width/r_width) # x coordinate
                # y_left = parsed[2]*(self.height/r_height) # y coordinate from top
                # box_width = parsed[3]*(self.width/r_width) # length along x axis
                # box_height = parsed[4]*(self.height/r_height) # length along y axis
                labels.append(self.label_map[str(int(parsed[0]))]) # parsed the category id
                x_left = parsed[1] # x coordinate
                y_left = parsed[2] # y coordinate from top
                box_width = parsed[3] # length along x axis
                box_height = parsed[4] # length along y axis
                xmin = int(x_left)
                xmax = int(x_left + (box_width+1))
                ymin = int(y_left)
                ymax = int(y_left + (box_height+1))

                boxes.append([xmin, ymin, xmax, ymax])
        
        # convert boxes into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

#         # Read objects in this image (bounding boxes, labels, difficulties)
#         objects = self.objects[i]
#         boxes = torch.FloatTensor(objects["boxes"])  # (n_objects, 4)
#         labels = torch.LongTensor(objects["labels"])  # (n_objects)
#         difficulties = torch.ByteTensor(objects["difficulties"])  # (n_objects)

#         # Discard difficult objects, if desired
#         if not self.keep_difficult:
#             boxes = boxes[1 - difficulties]
#             labels = labels[1 - difficulties]
#             difficulties = difficulties[1 - difficulties]

        # Apply transformations
        image, boxes, labels = transform(
            image, boxes, labels, self.split)

        return image, boxes, labels

    def __len__(self):
        print(len(self.images))
        return len(self.images)
    
    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
        This describes how to combine these tensors of different sizes. We use lists.
        Note: this need not be defined in this Class, can be standalone.
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels, and difficulties
        """

        images, boxes, labels = [], [], []

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return (
            images,
            boxes,
            labels
        )  # tensor (N, 3, 300, 300), 2 lists of N tensors each
