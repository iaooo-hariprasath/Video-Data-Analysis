import os

folder = "Video Dataset"

extensions = set()

def get_extensions(folder_path):
    for filename in os.listdir(folder_path):
        if os.path.isdir(os.path.join(folder_path, filename)):
            subfolder_path = os.path.join(folder_path, filename)
            get_extensions(subfolder_path)
            continue
        extensions.add(os.path.splitext(filename)[1])


get_extensions(folder)

print(extensions)
