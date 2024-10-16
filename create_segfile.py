import os

def save_filenames_to_txt(folder_path, output_txt_file):
    with open(output_txt_file, 'w') as f:
        # Iterate over all files in the folder
        for filename in os.listdir(folder_path):
            # Get the file name without the extension
            name_without_extension, extension = os.path.splitext(filename)
            # Write the file name to the text file
            # print(extension)
            if extension != '.json':
                f.write(name_without_extension + '\n')

    print(f"File names saved to {output_txt_file}")


# Example usage
folder_path = 'project_2.v15i.coco-segmentation/train'  # Replace with your folder path
output_txt_file = 'project_2.v15i.coco-segmentation/seg.txt'  # Replace with your desired output path
save_filenames_to_txt(folder_path, output_txt_file)
