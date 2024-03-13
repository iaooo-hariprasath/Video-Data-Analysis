import os
import csv
import cv2
import random

#possible video extensions
video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']

#create a csv file and create the head row
def create_csv_file(csv_filename):
    # Open the CSV file for writing
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Video Path', 'Size (bytes)', 'Width', 'Height', 'FPS', 'Total Frames'])  # Write header row

#given a list of video paths, process each video and save the video dims in given csv file
def get_video_details(csv_filename, video_path_list):
    with open(csv_filename, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Traverse through the folder
        for video_path in video_path_list:
            if os.path.isfile(video_path):
                # Get video size
                video_size = os.path.getsize(video_path)

                # Get video dimensions and frame rate
                cap = cv2.VideoCapture(video_path)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Display video details
                print(f"Video: {video_path}, Size: {video_size} bytes, Dimensions: {width}x{height}, FPS: {fps}, Total Frames: {total_frames}")

                # Write video details to CSV
                writer.writerow([video_path, video_size, width, height, fps, total_frames])


#given a folder, iterate through all files and similar subfolders, while in each folder of videos, 
#select only a particular percentage of videos for getting dimension inference
def traverse_folders(folder_path, csv_filename, percentage_of_videos):
    percentage = percentage_of_videos
    videos_list = []
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isdir(filepath):
            traverse_folders(filepath, csv_filename, percentage_of_videos)
            continue
        if os.path.splitext(filename)[1] in video_extensions:
            videos_list.append(filepath)
    
    total_videos = len(videos_list)
    num_samples = int(total_videos * (percentage/100))

    video_path_list = random.sample(videos_list, num_samples)
    get_video_details(csv_filename, video_path_list)


if __name__ == "__main__":
    # Folder path containing videos
    folder_path = '../../../Downloads/Video Dataset'

    # CSV file to store video details
    csv_filename = 'video_details.csv'

    create_csv_file(csv_filename)

    percentage_of_videos = 10 #percentage of videos to select form a directory of videos

    traverse_folders(folder_path, csv_filename, percentage_of_videos)
    print("Video details saved to", csv_filename)
