import albumentations as A
import numpy as np
import glob
import os
from PIL import Image

# Define augmentation pipeline
transform = A.Compose([
    A.HorizontalFlip(p=0.5),  # Horizontal flip
    A.VerticalFlip(p=0.5),  # Vertical flip
    A.RandomRotate90(p=0.5),  # Rotate 90 degrees
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.5),  # Random scaling, rotation
    A.RandomBrightnessContrast(p=0.5),  # Adjust brightness and contrast
])

# Function to apply augmentation to both image and mask
def augment_image_and_mask(image, mask):
    # Apply the transformation
    augmented = transform(image=image, mask=mask)
    aug_image = augmented['image']
    aug_mask = augmented['mask']

    return aug_image, aug_mask

# Function to augment the entire dataset
def augment_dataset(train_images_dir, train_masks_dir, augmented_images_dir, augmented_masks_dir):
    # Get list of image and mask files
    train_images = glob.glob(os.path.join(train_images_dir, '*'))
    train_masks = glob.glob(os.path.join(train_masks_dir, '*'))

    # Sort to ensure images and masks match in order
    train_images.sort()
    train_masks.sort()

    # Create directories for augmented data if they don't exist
    os.makedirs(augmented_images_dir, exist_ok=True)
    os.makedirs(augmented_masks_dir, exist_ok=True)

    # Loop through each image and mask pair
    for i, (image_file, mask_file) in enumerate(zip(train_images, train_masks)):
        # Load image and mask
        image = np.array(Image.open(image_file).convert('RGB'))  # Ensure the image is RGB
        mask = np.array(Image.open(mask_file))  # Load mask (grayscale or RGB)

        # Perform augmentation
        augmented_image, augmented_mask = augment_image_and_mask(image, mask)

        # Save augmented images and masks
        aug_image_filename = os.path.join(augmented_images_dir, f'aug_image_round2_{i}.jpg')
        aug_mask_filename = os.path.join(augmented_masks_dir, f'aug_image_round2_{i}.png')

        # Save using PIL
        Image.fromarray(augmented_image).save(aug_image_filename)
        Image.fromarray(augmented_mask).save(aug_mask_filename)

        print(f"Saved augmented image and mask: {aug_image_filename}, {aug_mask_filename}")

# Example usage of the function
train_images_dir = '../segmentation_thesis_project/Iron_dataset/Iron_material/JPEGImages'
train_masks_dir = '../segmentation_thesis_project/Iron_dataset/Iron_material/SegmentationClass'
augmented_images_dir = '../segmentation_thesis_project/Iron_augmented/augmented_images'
augmented_masks_dir = '../segmentation_thesis_project/Iron_augmented/augmented_masks'

# Augment the dataset
augment_dataset(train_images_dir, train_masks_dir, augmented_images_dir, augmented_masks_dir)
