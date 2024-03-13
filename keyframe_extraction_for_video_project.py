"""
Script to extract keyframes from videos using a 3 stage process
"""

import os
import subprocess
import torch
from torchvision import models
from PIL import Image
import torchvision.transforms as T
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from collections import defaultdict
from pathlib import Path
import shutil
from tqdm import tqdm

#### Model to detect presence of objects
class EfficientNet:
    def __init__(self):
        self.model = models.get_model('efficientnet_b0')
        self.model.eval()
        self.preprocess = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

    def __call__(self, image: Image):
        image_tensor = self.preprocess(image).unsqueeze(0)
        with torch.no_grad():
            out = self.model(image_tensor).cpu().detach().numpy()
        return out


#### Model to detect presence of objects
class DINOv2:
    def __init__(self):
        self.image_transforms = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()
        self.model = self.model.to(self.device)

    def __call__(self, image: Image):
        transformed_image = self.image_transforms(image).unsqueeze(0)
        return self.model(transformed_image.to(self.device)).cpu().detach().numpy()

#### function to return the name of the directory to which the next run of the program should direct the outputs to
def create_next_run(dir_path):
    if os.path.isdir(dir_path):
        existing_runs = [ int(dir_name[4:]) for dir_name in os.listdir(dir_path) if dir_name.startswith("run_") ]
        if existing_runs:
            max_run = max(existing_runs)
            if any(os.listdir(os.path.join(dir_path, f"run_{max_run:03d}"))):
                next_number = max_run + 1
            else:
                next_number = max_run
        else:
            next_number = 1 
    else:
        next_number = 1
    return os.path.join(dir_path, f"run_{next_number:03d}")


#### script to itertate through folders containing videos and extract keyframes from videos and put them in a common folder "key_frames_path"
def iterate_through_folder(folder_path, key_frames_path):
    for filename in tqdm(os.listdir(folder_path), desc=f"Iterating through - {os.path.basename(folder_path)}"):
        filepath = os.path.join(folder_path, filename)
        if os.path.isdir(filepath):
            iterate_through_folder(filepath, key_frames_path)
            continue
        video_path = filepath

        output_file_format = f"{key_frames_path}/{os.path.basename(video_path)}-keyframes_%03d.jpg"
        ffmpeg_command = ["ffmpeg", "-i", video_path, "-vf","select=eq(pict_type\,I)","-vsync", "vfr", output_file_format]
        subprocess.run(ffmpeg_command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

#### Model to cluster all the keyframes stored in keyframes path, and by defining the number of clusters, narrow down the keyframes to a
#### limited set amd store the ultimate keyframes in "informative_frames_path"
def k_means_clustering(output_path, key_frames_path, informative_frames_path):
    print("Performing k-means clustering")
    d = EfficientNet()
    embeddings = []
    pathlist = Path(key_frames_path).glob('**/*.jpg')
    for path in tqdm(pathlist, desc="Extracting embeddings"):
        embeddings.append([str(path), d(Image.open(str(path)))])

    #Perform k-means clustering
    k = 50  # Number of clusters
    kmeans = KMeans(n_clusters=k, n_init='auto')
    image_paths = np.array(list(embedding[0] for embedding in embeddings))
    image_embeddings = np.array(list(embedding[1][0] for embedding in embeddings))
    cluster_labels = kmeans.fit_predict(image_embeddings)

    centroids = kmeans.cluster_centers_
    distances = cdist(image_embeddings, centroids)
    centroid_indices = np.argmin(distances, axis=0)

    selected_images = []
    for i in range(k):
        selected_images.append(image_paths[centroid_indices[i]])  # Change this to select a different image from each cluster

    for image in selected_images:
        shutil.copy(image, f"{informative_frames_path}/")

    for i, label in tqdm(enumerate(cluster_labels), desc="Copying informative frames"):
        tmp_path = os.path.join(output_path,"informative_frames_debug",str(label))
        os.makedirs(tmp_path, exist_ok=True)
        shutil.copy(image_paths[i], tmp_path)


if __name__ == "__main__":
    folder = "Applicable videos"

    output_directory = os.path.join(os.getcwd(), f"output/multi_step/")
    output_path = create_next_run(output_directory)
    os.makedirs(output_path, exist_ok=True) #create output directory

    key_frames_path = os.path.join(output_path, "key_frames")
    informative_frames_path = os.path.join(output_path, "informative_frames")

    os.makedirs(key_frames_path, exist_ok=False) #create keyframes directory
    os.makedirs(informative_frames_path, exist_ok=True) #create informative keyframes directory

    print("Starting iteration through folder")
    iterate_through_folder(folder, key_frames_path)

    k_means_clustering(output_path, key_frames_path, informative_frames_path)
