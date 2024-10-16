from PIL import Image

def print_image_matrix(image_path):
    # Open the image
    img = Image.open(image_path)

    # Convert image to RGB (or other mode if needed)
    # img = img.convert('RGB')

    # Get the size of the image
    width, height = img.size

    # Load pixel data
    pixels = img.load()

    # Loop through the pixels and print their values in matrix form
    for y in range(height):
        row = []
        for x in range(width):
            row.append(pixels[x, y])  # pixels[x, y] returns the RGB values of the pixel at (x, y)
        print(row)


# Example usage
print_image_matrix('VOCdevkit/VOC2007/SegmentationClass/2007_000032.png')
