import os

def count_images_in_folder(folder_path):
    # Define common image file extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff'}

    # Initialize a counter for images
    image_count = 0

    # Traverse through the directory and count the image files
    for filename in os.listdir(folder_path):
        # Get file extension and check if it's an image
        if os.path.splitext(filename)[1].lower() in image_extensions:
            image_count += 1

    return image_count

# Example usage
folder_path = 'Iron_dataset/Iron_material/SegmentationClass'
num_images = count_images_in_folder(folder_path)
print(f'Total number of images: {num_images}')
