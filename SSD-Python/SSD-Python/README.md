

### Dataset

Each image can contain one or more ground truth objects.

Each object is represented by –

- a bounding box in absolute boundary coordinates

- a label (one of the object types mentioned above)

### Inputs to model

We will need three inputs.

#### Images

* For SSD300 variant, the images would need to be sized at `300, 300` pixels and in the RGB format.
* PyTorch follows the NCHW convention, which means the channels dimension (C) must precede the size dimensions(1, 3, 300, 300).

Therefore, **images fed to the model must be a `Float` tensor of dimensions `N, 3, 300, 300`**, and must be normalized by the aforesaid mean and standard deviation. `N` is the batch size.


#### Objects' Bounding Boxes

For each image, the bounding boxes of the ground truth objects follows (x_min, y_min, x_max, y_max) format`.

# Training
* In config.json change the paths. 
* "backbone_network" : "MobileNetV2" or "MobileNetV1"
* For training run
  ```
  python train.py config.json
  ```
# Inference 
  ```
  python inference.py image_path checkpoint
  ```
 
