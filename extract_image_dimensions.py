import os
import csv
from tqdm import tqdm

# Folder path containing images
folder_path = 'Frames/'

# CSV file to store image details
csv_filename = 'image_sizes.csv'

# Open the CSV file for writing
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Image', 'Width', 'Height'])  # Write header row

    # Traverse through the folder
    for img_name in tqdm(os.listdir(folder_path), desc="Getting Image Sizes"):
        img_path = os.path.join(folder_path, img_name)
        if os.path.isfile(img_path):
            # Get image size
            img_size = os.path.getsize(img_path)

            # Get image width and height (if it's an image file)
            try:
                from PIL import Image
                with Image.open(img_path) as img:
                    width, height = img.size
            except Exception as e:
                width, height = None, None

            # Display image size
            # print(f"Image: {img_name}, Size: {img_size} bytes, Width: {width}, Height: {height}")

            # Write image details to CSV
            writer.writerow([img_name, width, height])
