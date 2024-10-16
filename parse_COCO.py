import json
import os
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Path to the dataset
annotation_file = 'img/valid/_annotations.coco.json'  # Update this with the actual path to your annotation file
image_folder = 'img/valid'  # Update this with the actual path to your images

# Load the COCO annotations
coco = COCO(annotation_file)

# Get all category IDs and corresponding names
cat_ids = coco.getCatIds()
categories = coco.loadCats(cat_ids)

print("Categories:", categories)

# Get all image IDs
img_ids = coco.getImgIds()

# Loop through and process the images
# for img_id in img_ids[:5]:  # For demonstration, only show the first 5 images

img_id = 22
# Load image metadata
img_metadata = coco.loadImgs(img_id)[0]
# print('metadata', img_metadata)
img_path = os.path.join(image_folder, img_metadata['file_name'])

# Load the image
image = np.array(Image.open(img_path))

# Load annotations (segmentations, bounding boxes, etc.)
ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=None)
print('ann_ids', ann_ids)
annotations = coco.loadAnns(ann_ids)
print('annotations', annotations)

# Display the image
plt.imshow(image)
plt.axis('off')

# Display segmentation masks or bounding boxes
coco.showAnns(annotations)
plt.show()

# Clear the plot after showing
# plt.clf()  # Clears the current figure
# plt.close()  # Closes the figure window
