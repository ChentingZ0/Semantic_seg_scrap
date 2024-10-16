import os
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from PIL import Image

# Path to your COCO annotation file
annotation_file = 'project_2.v15i.coco-segmentation/train/_annotations.coco.json'

# output_mask_dir
output_mask_dir = 'project_2.v15i.coco-segmentation/train_mask'

def get_labels():
    """Load the mapping that associates classes with label colors.
       Our electrical substation dataset has 3 objects + background.
    Returns:
        np.ndarray with dimensions (16, 3)
    """
    return np.asarray([(0, 0, 0),  # category0_background: black
                       (162, 0, 255),  # category1_copper: purple
                       (255, 0, 0),  # category2_meatball:
                       (81, 162, 0),  # category3_trash: green
                       ])

# Create the output directory if it doesn't exist
os.makedirs(output_mask_dir, exist_ok=True)

# Load COCO annotations
coco = COCO(annotation_file)

# Get image ids
image_ids = coco.getImgIds()

for img_id in image_ids:
    img_info = coco.loadImgs(img_id)[0]
    img_height = img_info['height']
    img_width = img_info['width']

    # Initialize a blank single-channel mask (single channel for category ids)
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # # 3-channel mask initilization
    # mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    # Get all annotations for the image
    ann_ids = coco.getAnnIds(imgIds=img_id)
    annotations = coco.loadAnns(ann_ids)

    for ann in annotations:
        segmentation = ann['segmentation']
        category_id = ann['category_id']

        # If segmentation is in polygon format
        if isinstance(segmentation, list):
            rle = maskUtils.frPyObjects(segmentation, img_height, img_width)
            rle = maskUtils.merge(rle)
            m = maskUtils.decode(rle)
        else:
            # If segmentation is already in RLE format
            m = maskUtils.decode(segmentation)

        # Update the single-channel mask with category_id
        mask[m > 0] = category_id

        # # 3-channel mask with color label
        # class_colors = get_labels()
        # mask[m > 0] = class_colors[category_id]

    # Save the mask with the exact same name as the original image (with .png extension)
    filename = img_info["file_name"]
    name_without_extension, _ = os.path.splitext(filename)

    # Create the new filename with .png extension
    new_filename = name_without_extension + ".png"

    # Convert the single-channel mask to a PIL image and save
    mask_image = Image.fromarray(mask)
    mask_image.save(os.path.join(output_mask_dir, new_filename))

print("Single-channel segmentation masks saved successfully.")
