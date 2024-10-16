import numpy as np
import glob
from PIL import Image
from collections import defaultdict

def count_pixels_per_class(mask_dir, num_classes):
    # Initialize a dictionary to hold the pixel counts for each class
    pixel_counts = defaultdict(int)
    # Get all mask file paths
    mask_files = glob.glob(mask_dir + "/*")

    # Loop through each mask
    for mask_file in mask_files:
        # Load the mask as a numpy array
        mask = np.array(Image.open(mask_file))

        # Count the number of pixels for each class
        for class_id in range(num_classes):
            pixel_counts[class_id] += np.sum(mask == class_id)

    return dict(pixel_counts)

# Example usage:
mask_dir = 'Iron_dataset/Iron_material/SegmentationClass'  # Path to your masks
num_classes = 4  # Example: number of classes in your dataset

pixel_counts = count_pixels_per_class(mask_dir, num_classes)
pixel_all = int(np.sum(list(pixel_counts.values())))


# Calculate percentages for each class
pixel_percentages = {class_id: (count / pixel_all) * 100 for class_id, count in pixel_counts.items()}

print("Pixel counts per class:", pixel_counts)
print("Pixel percentage counts per class:", pixel_percentages)
# Result: Pixel counts per class: {background(0): 583924202, copper(1): 84414114, meatball(2): 63487299, trash(3): 75559217}
# percentage: Pixel percentage counts per class: {0: 72.3229%, 1: 10.4553%, 2: 7.8633%, 3: 9.3585%}